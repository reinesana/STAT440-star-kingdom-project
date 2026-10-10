"""Takito-style rich features with year-start availability and frozen observations."""
from pathlib import Path
from collections import defaultdict
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / 'ensemble_preparation'
DATA = ROOT / 'github-main/data'
OUT = HERE / 'data'
BASE = ['scenario_year','material','surface','material_surface','pipe_length_m','log1p_pipe_length_m',
        'midpoint_x_m','midpoint_y_m','age_at_year_start','log1p_age_at_year_start','age_missing']


def topology(pipes):
    path = ROOT/'github-main/src/enrich_cost_features.py'
    spec = importlib.util.spec_from_file_location('reviewed_topology',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route_features(pipes)[0]


class FeatureBuilder:
    def __init__(self):
        self.pipes = pd.read_csv(DATA/'pipes.csv')
        self.raw = pd.read_csv(PREP/'prepared/training_features_all.csv')
        self.ids = pd.Series(np.arange(len(self.pipes)),index=self.pipes['Pipe ID'])
        self.a = self.pipes[['GPS x1','GPS y1']].to_numpy(float)
        self.b = self.pipes[['GPS x2','GPS y2']].to_numpy(float)
        self.mid = (self.a+self.b)/2
        self.length = np.linalg.norm(self.b-self.a,axis=1)
        self.lay = pd.to_datetime(self.pipes['Lay date'])
        assert self.lay.dropna().lt(pd.Timestamp('2019-01-01')).all()
        self.material = self.pipes.Material.fillna('__unknown__').to_numpy()
        self.surface = self.pipes.Surface.fillna('__unknown__').to_numpy()
        self.repair_date = self.pipes['Pipe ID'].map(
            pd.read_csv(DATA/'train.csv').set_index('Pipe ID').Date)
        self.repair_date = pd.to_datetime(self.repair_date).to_numpy(dtype='datetime64[ns]')
        self.routes = topology(self.pipes).set_index('pipe_id')
        self.neighbors = {}
        valid = np.flatnonzero(self.lay.notna())
        tree = cKDTree(self.mid[valid])
        for radius in [50,100,250]:
            self.neighbors[radius] = [valid[np.asarray(z,int)][valid[np.asarray(z,int)]!=i]
                for i,z in enumerate(tree.query_ball_point(self.mid,radius))]
        self.static = self.build_static()
        self.events = self.add_keys(self.raw.copy())

    def add_keys(self, query):
        q=query.copy()
        route=self.routes.reindex(q.pipe_id)
        q['route_key']=route.route_id.fillna('__unknown__').to_numpy()
        for size in [250,500,1000]:
            q[f'grid{size}_key']=(np.floor(q.midpoint_x_m/size).astype(int).astype(str)+'_'+
                                  np.floor(q.midpoint_y_m/size).astype(int).astype(str))
        q['age_band_key']=pd.cut(q.age_at_year_start,[-np.inf,20,40,60,80,np.inf],labels=False).fillna(-1).astype(int).astype(str)
        q['length_band_key']=pd.cut(q.pipe_length_m,[-np.inf,10,25,50,100,np.inf],labels=False).astype(str)
        return q

    def build_static(self):
        rows=[]
        for i in range(len(self.pipes)):
            dx,dy=self.b[i]-self.a[i]
            r={'pipe_id':self.pipes['Pipe ID'].iloc[i], 'span_x_m':abs(dx),'span_y_m':abs(dy),
               'orientation_sin2':2*dx*dy/self.length[i]**2,
               'orientation_cos2':(dx*dx-dy*dy)/self.length[i]**2}
            for radius in [50,100,250]:
                j=self.neighbors[radius][i]; n=len(j); tag=f'inventory_midpoint_{radius}m_'
                r[tag+'count']=n
                r[tag+'length_sum_m']=self.length[j].sum()
                r[tag+'same_original_material_fraction']=float((self.material[j]==self.material[i]).mean()) if n else 0
                for surface in ['road','structure','water','swamp']:
                    r[tag+surface+'_count']=int((self.surface[j]==surface).sum())
            rows.append(r)
        static=pd.DataFrame(rows).set_index('pipe_id')
        eligible=np.flatnonzero(self.lay.notna())
        for label in ['road','structure','water']:
            valid=eligible[self.surface[eligible]==label]
            tree=cKDTree(self.mid[valid])
            ds,ix=tree.query(self.mid,k=2)
            same=valid[ix[:,0]]==np.arange(len(self.pipes))
            static[f'inventory_nearest_{label}_midpoint_distance_m']=np.where(same,ds[:,1],ds[:,0])
        for col in self.routes:
            if col not in ['route_id','straight_section_id']:
                static['topology_'+col]=self.routes[col].reindex(static.index)
        # Inventory is an original-design snapshot. These are geometric covariates,
        # not observed excavation lengths or proof of physical connectivity.
        return static

    def build(self, query, origin_year, events=None):
        q=self.add_keys(query)
        e=self.events if events is None else events
        h=e.loc[e.event_year<origin_year]
        assert h.empty or h.event_year.max()<origin_year
        pieces=[q[BASE].reset_index(drop=True),self.static.reindex(q.pipe_id).reset_index(drop=True)]
        out={}
        keys=['material','surface','material_surface','grid250_key','grid500_key','grid1000_key',
              'age_band_key','length_band_key','route_key']
        threshold=h.cost.quantile(.99)
        for window, first in [('all',2019),('last2',origin_year-2)]:
            hh=h.loc[h.event_year>=first]
            mean=hh.cost.mean()
            rate=(hh.cost>=threshold).mean() if len(hh) else np.nan
            out[f'hist_{window}_global_mean']=np.full(len(q),mean)
            for key in keys:
                z=hh.assign(tail=(hh.cost>=threshold).astype(int)).groupby(key).agg(
                    n=('cost','size'),total=('cost','sum'),median=('cost','median'),max=('cost','max'),
                    std=('cost','std'),tail=('tail','sum'))
                count=q[key].map(z.n).fillna(0).to_numpy()
                total=q[key].map(z.total).fillna(0).to_numpy()
                tag=f'hist_{window}_{key.removesuffix("_key")}_'
                out[tag+'n']=count
                out[tag+'mean']=(total+20*mean)/(count+20)
                out[tag+'tail_rate']=(q[key].map(z['tail']).fillna(0).to_numpy()+50*rate)/(count+50)
                for stat in ['median','max','std']:
                    out[tag+stat]=q[key].map(z[stat]).to_numpy()
        for radius in [50,100,250]:
            rows=[]
            for i in self.ids.reindex(q.pipe_id).to_numpy():
                j=self.neighbors[radius][i]
                prior=self.repair_date[j]<np.datetime64(f'{origin_year}-01-01')
                recently=prior&(self.repair_date[j]>=np.datetime64(f'{origin_year-2}-01-01'))
                iron=np.isin(self.material[j],['cast iron','gray iron','wrought iron'])&~prior
                rows.append([prior.sum(),prior.mean() if len(j) else 0,recently.sum(),iron.sum(),self.length[j][iron].sum()])
            for ix,stat in enumerate(['prior_repairs','prior_repair_fraction','last2_repairs','remaining_iron_count','remaining_iron_length_m']):
                out[f'state_{radius}m_{stat}']=np.array(rows)[:,ix]
        positions=q[['midpoint_x_m','midpoint_y_m']].to_numpy()
        if len(h):
            tree=cKDTree(h[['midpoint_x_m','midpoint_y_m']].to_numpy())
            costs=h.cost.to_numpy()
            for radius in [50,100,250,500]:
                rows=[]
                for group in tree.query_ball_point(positions,radius):
                    z=costs[group]; n=len(z)
                    rows.append([n,(z.sum()+20*h.cost.mean())/(n+20),np.median(z) if n else np.nan,
                                 z.max() if n else np.nan,((z>=threshold).sum()+50*(h.cost>=threshold).mean())/(n+50)])
                for ix,stat in enumerate(['n','mean','median','max','tail_rate']):
                    out[f'hist_radius{radius}m_{stat}']=np.array(rows)[:,ix]
            _,near=tree.query(positions,k=min(25,len(h)))
            if near.ndim==1: near=near[:,None]
            for k in [5,10,25]:
                values=costs[near[:,:k]]
                for stat,values_agg in [('mean',values.mean(axis=1)),('median',np.median(values,axis=1)),('max',values.max(axis=1))]:
                    out[f'hist_nearest{k}_{stat}']=values_agg
        else:
            for radius in [50,100,250,500]:
                for stat in ['n','mean','median','max','tail_rate']:
                    out[f'hist_radius{radius}m_{stat}']=np.full(len(q),0 if stat=='n' else np.nan)
            for k in [5,10,25]:
                for stat in ['mean','median','max']:out[f'hist_nearest{k}_{stat}']=np.full(len(q),np.nan)
        history=pd.DataFrame(out)
        logs={col+'_log1p':np.log1p(history[col]) for col in history
              if col.endswith(('_mean','_median','_max','_std'))}
        history=pd.concat([history,pd.DataFrame(logs)],axis=1)
        pieces.append(history)
        x=pd.concat(pieces,axis=1).copy()
        x['length_x_structure']=x.log1p_pipe_length_m*np.log1p(x.inventory_midpoint_100m_structure_count)
        x['road_x_structure']=x.surface.eq('road').astype(int)*np.log1p(x.inventory_midpoint_100m_structure_count)
        x['age_x_wet']=x.age_at_year_start*np.log1p(x.inventory_midpoint_100m_water_count+x.inventory_midpoint_100m_swamp_count)
        x=x.copy()  # consolidate feature blocks
        assert x.columns.is_unique and not np.isinf(x.select_dtypes(include=np.number).to_numpy()).any()
        assert not any(token in c for c in x for token in ['hour','weekday','event_month'])
        return x

    def future(self, year):
        if year<2027: raise ValueError('Future forecasts start in 2027.')
        q=pd.read_csv(PREP/'prepared/forecast_features_2027_supported.csv')
        q['scenario_year']=year
        lay=self.pipes.set_index('Pipe ID')['Lay date'].reindex(q.pipe_id)
        q['age_at_year_start']=(pd.Timestamp(f'{year}-01-01')-pd.to_datetime(lay)).dt.days.to_numpy()/365.25
        q['log1p_age_at_year_start']=np.log1p(q.age_at_year_start)
        return q,self.build(q,origin_year=2027)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    builder=FeatureBuilder()
    frames=[];checks=[]
    for year in range(2019,2027):
        q=builder.events.loc[builder.events.event_year==year]
        x=builder.build(q,year)
        frame=pd.concat([q[['event_id','pipe_id','event_year','cost']].reset_index(drop=True),x],axis=1)
        frames.append(frame)
        sample=q.head(8)
        before=builder.build(sample,year)
        changed=builder.events.copy()
        mask=changed.event_year>=year
        changed.loc[mask,'cost']=1e12
        changed.loc[mask,'material']='future-mutated'
        changed.loc[mask,['midpoint_x_m','midpoint_y_m']]+=1e6
        pd.testing.assert_frame_equal(before,builder.build(sample,year,changed))
        pd.testing.assert_frame_equal(before,builder.build(sample,year,builder.events.loc[builder.events.event_year<year]))
        checks.append({'year':year,'future_cost_location_material_mutation':'passed','future_event_removal':'passed'})
        print(f'features year={year} rows={len(q)} cols={len(x.columns)}',flush=True)
    d=pd.concat(frames,ignore_index=True)
    d.to_csv(OUT/'training_rich.csv',index=False)
    q,x=builder.future(2027)
    pd.concat([q[['pipe_id']].reset_index(drop=True),x],axis=1).to_csv(OUT/'forecast_features_2027.csv',index=False)
    (OUT/'feature_contract.json').write_text(json.dumps({'features':list(x),
        'base_features':BASE,'categorical':['material','surface','material_surface'],
        'rows':len(d),'future_rows':len(q),'history_origin_rule':'strictly earlier calendar years; observed future origin fixed at 2027',
        'inventory_rule':'known lay dates precede 2019; original attributes assumed valid before first leak',
        'topology_rule':'exact endpoint chains, geometric proxy only; no planar crossing connectivity',
        'missing_rule':'retain six missing ages; training-split preprocessing',
        'future_invariance_checks':checks},indent=2),encoding='utf-8')
    print(f'Prepared {len(d)} events, {len(x.columns)} deployable features, {len(q)} supported future pipes',flush=True)


if __name__=='__main__':main()
