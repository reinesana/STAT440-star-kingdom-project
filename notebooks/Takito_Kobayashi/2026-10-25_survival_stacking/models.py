"""Ten distinct survival estimators with a common survival-curve interface.

MTLR and DeepHit implement censored discrete PMF likelihoods in torch.
The hierarchical hazard model is penalized MAP, not posterior MCMC.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, expit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from data import SEED

MODELS = ['01_elastic_net_cox', '02_weibull_aft', '03_lognormal_aft', '04_spline_survival',
          '05_random_survival_forest', '06_xgboost_aft', '07_catboost_hazard', '08_mtlr',
          '09_deephit', '10_hierarchical_hazard']
CATS = ['material', 'surface', 'lay_decade', 'material_cohort']


class Transform:
    def fit(self, frame, columns):
        self.columns = columns
        self.cat = [c for c in columns if c in CATS]
        self.num = [c for c in columns if c not in CATS]
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.scale = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        if self.num:
            self.scale.fit(self.imputer.fit_transform(frame[self.num]))
        if self.cat:
            self.ohe.fit(frame[self.cat].fillna('__unknown__').astype(str))
        return self

    def apply(self, frame):
        parts = []
        if self.num:
            parts.append(self.scale.transform(self.imputer.transform(frame[self.num])))
        if self.cat:
            parts.append(self.ohe.transform(frame[self.cat].fillna('__unknown__').astype(str)))
        return np.column_stack(parts).astype(np.float32)


def ranking(frame, duration, event, groups):
    """Proposal ranking is fit solely on the current inner/outer training data."""
    columns = sum(groups.values(), [])
    columns = list(dict.fromkeys(c for c in columns if frame[c].nunique(dropna=False) > 1))
    tr = Transform().fit(frame, columns)
    x = tr.apply(frame)
    y = event & (duration <= 1)
    forest = ExtraTreesClassifier(n_estimators=64, max_depth=6, min_samples_leaf=50,
                                 max_features=.7, random_state=SEED, n_jobs=4)
    forest.fit(x, y)
    values = dict(zip(tr.num, forest.feature_importances_[:len(tr.num)]))
    start = len(tr.num)
    for col, cat in zip(tr.cat, tr.ohe.categories_ if tr.cat else []):
        values[col] = forest.feature_importances_[start:start + len(cat)].sum()
        start += len(cat)
    return sorted(columns, key=lambda c: (-values.get(c, 0), c))


def select(frame, config, groups, ranked):
    mode = config['features']
    if mode == 'all':
        cols = sum(groups.values(), [])
    elif mode == 'ranked':
        cols = ranked[:config['top_k']]
    elif mode == 'groups':
        cols = sum([v for k, v in groups.items() if config.get('group_' + k, False)], [])
    else:
        cols = list(groups['basic'])
    # Model-defining covariates are mandatory only for these two estimators.
    if config['model'] == '04_spline_survival':
        cols += ['age', 'material']
    if config['model'] == '10_hierarchical_hazard':
        cols += ['material', 'lay_year', 'age']
    cols = list(dict.fromkeys(cols))
    cols = [c for c in cols if frame[c].nunique(dropna=False) > 1 or c in ['age', 'material', 'lay_year']]
    if not cols:
        cols = ['age']
    return cols


class Estimator:
    def __init__(self, name, config, columns, cutoff):
        self.name, self.c, self.columns, self.cutoff = name, config, columns, cutoff
        self.support = cutoff - 2018 if name in ['01_elastic_net_cox', '05_random_survival_forest', '08_mtlr', '09_deephit'] else 25

    def fit(self, f, duration, event):
        self.tr = Transform().fit(f, self.columns)
        x = self.tr.apply(f)
        c = self.c
        if self.name == '01_elastic_net_cox':
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            from sksurv.util import Surv
            self.model = CoxnetSurvivalAnalysis(alphas=[c['alpha']], l1_ratio=c['l1_ratio'],
                                               fit_baseline_model=True, max_iter=30000, tol=1e-6)
            self.model.fit(x, Surv.from_arrays(event, duration))
        elif self.name in ['02_weibull_aft', '03_lognormal_aft']:
            xx = np.column_stack([np.ones(len(x)), x]).astype(float)
            logt = np.log(duration)
            penalty = c['penalty']
            weibull = self.name == '02_weibull_aft'

            def objective(theta):
                mu = xx @ theta[:-1]
                scale = np.exp(theta[-1])
                z = (logt - mu) / scale
                if weibull:
                    ez = np.exp(np.clip(z, -100, 80))
                    loss = ez - event * (z - theta[-1] - logt)
                    dmu = (event - ez) / scale
                    ds = event * (1 + z) - ez * z
                else:
                    logsf = log_ndtr(-z)
                    ratio = np.exp(np.clip(-.5*z*z - .5*np.log(2*np.pi) - logsf, -100, 80))
                    loss = np.where(event, .5*z*z + theta[-1] + logt + .5*np.log(2*np.pi), -logsf)
                    dmu = np.where(event, -z / scale, -ratio / scale)
                    ds = np.where(event, 1-z*z, -ratio*z)
                ridge = theta[:-1].copy(); ridge[0] = 0
                value = loss.mean() + .5 * penalty * (ridge @ ridge)
                grad = np.r_[xx.T @ dmu / len(xx) + penalty * ridge, ds.mean()]
                return value, grad

            theta = np.zeros(xx.shape[1] + 1)
            theta[0] = np.log(np.median(duration))
            result = minimize(objective, theta, method='L-BFGS-B', jac=True,
                              bounds=[(None, None)] * xx.shape[1] + [(-3, 3)],
                              options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-6})
            if not result.success:
                raise RuntimeError('AFT convergence: ' + str(result.message))
            self.theta = result.x
        elif self.name == '05_random_survival_forest':
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv
            self.model = RandomSurvivalForest(n_estimators=c['trees'], max_depth=c['depth'],
                min_samples_leaf=c['leaf'], max_features=c['max_features'], max_samples=c['sample'],
                n_jobs=4, random_state=SEED)
            self.model.fit(x, Surv.from_arrays(event, duration))
        elif self.name == '06_xgboost_aft':
            import xgboost as xgb
            d = xgb.DMatrix(x, label_lower_bound=duration, label_upper_bound=np.where(event, duration, np.inf))
            self.model = xgb.train(dict(objective='survival:aft', eval_metric='aft-nloglik', tree_method='hist',
                aft_loss_distribution=c['distribution'], aft_loss_distribution_scale=c['aft_scale'],
                max_depth=c['depth'], eta=c['lr'], min_child_weight=c['child'],
                subsample=c['sample'], colsample_bytree=c['colsample'], reg_lambda=c['l2'], reg_alpha=c['l1'],
                nthread=4, seed=SEED), d, num_boost_round=c['trees'])
        elif self.name in ['04_spline_survival', '07_catboost_hazard', '10_hierarchical_hazard']:
            # First-year outcomes of every historical landmark are the annual risk set.
            y = (event & (duration <= 1)).astype(int)
            if self.name == '04_spline_survival':
                age = f[['age']].fillna(f.age.median()).to_numpy()
                self.age_fill = float(f.age.median())
                self.spline = SplineTransformer(n_knots=c['knots'], degree=3, extrapolation='linear', include_bias=False)
                z = self.spline.fit_transform(age)
                # Material-specific age curves permit non-proportional hazards.
                self.materials = sorted(f.material.unique())
                interaction = np.column_stack([z * f.material.eq(m).to_numpy()[:,None] for m in self.materials])
                self.model = LogisticRegression(C=c['C'], max_iter=1200, solver='lbfgs', tol=1e-6)
                self.model.fit(np.column_stack([x, interaction]), y)
                if self.model.n_iter_[0] >= 1200:
                    raise RuntimeError('Spline logistic did not converge')
            elif self.name == '07_catboost_hazard':
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier(iterations=c['trees'], depth=c['depth'], learning_rate=c['lr'],
                    l2_leaf_reg=c['l2'], random_strength=c['random_strength'],
                    loss_function='Logloss', random_seed=SEED, thread_count=4, verbose=False,
                    allow_writing_files=False)
                self.model.fit(self.catframe(f), y, cat_features=[v for v in self.columns if v in CATS])
            else:
                self.fit_hierarchy(f, x, y)
        else:
            self.fit_neural(x, duration, event)
        return self

    def catframe(self, f):
        x = f[self.columns].copy()
        for col in self.columns:
            x[col] = x[col].fillna('__unknown__').astype(str) if col in CATS else x[col].fillna(-999.)
        return x

    def hierarchy_indices(self, f, fit=False):
        width = self.c['cohort_width']
        cohort = (f.lay_year.fillna(-999) // width).astype(int).astype(str)
        levels = [f.material.astype(str), cohort, f.material.astype(str) + '_' + cohort]
        if fit:
            self.maps = [{v:i+1 for i,v in enumerate(sorted(level.unique()))} for level in levels]
        return np.column_stack([level.map(mapping).fillna(0).to_numpy(int) for level,mapping in zip(levels,self.maps)])

    def fit_hierarchy(self, f, x, y):
        import torch
        torch.manual_seed(SEED); torch.set_num_threads(4)
        indices = self.hierarchy_indices(f, fit=True)
        # Fixed covariates + shrunk material, cohort and nested material/cohort effects.
        xt = torch.tensor(x); yt = torch.tensor(y, dtype=torch.float32)
        it = torch.tensor(indices, dtype=torch.long)
        self.linear = torch.nn.Linear(x.shape[1], 1)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(len(m)+1, 1, padding_idx=0) for m in self.maps])
        with torch.no_grad():
            self.linear.weight.zero_(); self.linear.bias.fill_(float(np.log((y.mean()+1e-5)/(1-y.mean()+1e-5))))
            for emb in self.embeddings: emb.weight.zero_()
        params = list(self.linear.parameters()) + list(self.embeddings.parameters())
        optim = torch.optim.Adam(params, lr=self.c['lr'])
        rng = np.random.default_rng(SEED)
        for epoch in range(self.c['epochs']):
            for ix in np.array_split(rng.permutation(len(x)), max(1, len(x)//8192)):
                logits = self.linear(xt[ix]).squeeze(1)
                for j, emb in enumerate(self.embeddings): logits = logits + emb(it[ix,j]).squeeze(1)
                penalty = self.c['fixed_penalty'] * self.linear.weight.square().mean()
                for j,emb in enumerate(self.embeddings): penalty = penalty + self.c['group_penalty'] * (self.c['nested_multiplier'] if j==2 else 1) * emb.weight.square().mean()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[ix]) + penalty
                optim.zero_grad(); loss.backward(); optim.step()

    def fit_neural(self, x, duration, event):
        import torch
        torch.manual_seed(SEED); torch.set_num_threads(4)
        c = self.c; k = self.support
        layers = []; dim = x.shape[1]
        for _ in range(c['layers']):
            layers += [torch.nn.Linear(dim, c['width']), torch.nn.ReLU(), torch.nn.Dropout(c['dropout'])]
            dim = c['width']
        layers += [torch.nn.Linear(dim, k+1)]
        self.net = torch.nn.Sequential(*layers)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c['lr'], weight_decay=c['decay'])
        xt = torch.tensor(x); dt = torch.tensor(duration, dtype=torch.float32)
        et = torch.tensor(event); idx = torch.tensor(np.minimum(np.floor(duration).astype(int), k))
        rng = np.random.default_rng(SEED)
        self.net.train()
        for epoch in range(c['epochs']):
            for ix in np.array_split(rng.permutation(len(x)), max(1, len(x)//c['batch'])):
                logits = self.net(xt[ix])
                if self.name == '08_mtlr':
                    logits = torch.flip(torch.cumsum(torch.flip(logits, [1]), 1), [1])
                logpmf = torch.log_softmax(logits, dim=1)
                ii = idx[ix]; ee = et[ix]
                ll_event = logpmf.gather(1, ii[:,None]).squeeze(1)
                allowed = torch.arange(k+1)[None,:] >= ii[:,None]
                ll_censor = torch.logsumexp(logpmf.masked_fill(~allowed, -torch.inf), dim=1)
                loss = -torch.where(ee, ll_event, ll_censor).mean()
                if self.name == '09_deephit':
                    # Stochastic comparable-pair DeepHit ranking penalty.
                    pmf = logpmf.exp(); cdf = pmf.cumsum(1)
                    j = torch.randperm(len(ix))
                    comparable = ee & (dt[ix] < dt[ix[j]])
                    if comparable.any():
                        own = cdf.gather(1, ii[:,None]).squeeze(1)
                        other = cdf[j].gather(1, ii[:,None]).squeeze(1)
                        rank = torch.exp(torch.clamp((other-own)/c['sigma'], max=10))[comparable].mean()
                        loss = c['alpha'] * loss + (1-c['alpha']) * rank
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()

    def annual_hazard(self, f):
        x = self.tr.apply(f)
        if self.name == '04_spline_survival':
            z = self.spline.transform(f[['age']].fillna(self.age_fill).to_numpy())
            interaction = np.column_stack([z*f.material.eq(m).to_numpy()[:,None] for m in self.materials])
            return self.model.predict_proba(np.column_stack([x,interaction]))[:,1]
        if self.name == '07_catboost_hazard':
            return self.model.predict_proba(self.catframe(f))[:,1]
        import torch
        with torch.no_grad():
            logits = self.linear(torch.tensor(x)).squeeze(1)
            it = torch.tensor(self.hierarchy_indices(f))
            for j,emb in enumerate(self.embeddings): logits += emb(it[:,j]).squeeze(1)
            return torch.sigmoid(logits).numpy()

    def survival(self, f, times):
        times = np.asarray(times, float)
        x = self.tr.apply(f)
        if self.name in ['04_spline_survival', '07_catboost_hazard', '10_hierarchical_hazard']:
            columns=[]; s=np.ones(len(f))
            for year in range(1, int(times.max())+1):
                future=f.copy(); future['age']=future.age+year-1
                future['origin_year']=future.origin_year+year-1
                # Observed history and group rates are frozen at forecast origin.
                s *= 1-self.annual_hazard(future)
                columns.append(s.copy())
            out=np.column_stack(columns)[:,times.astype(int)-1]
        elif self.name in ['02_weibull_aft', '03_lognormal_aft']:
            mu=np.column_stack([np.ones(len(x)),x])@self.theta[:-1]
            z=(np.log(times)[None,:]-mu[:,None])/np.exp(self.theta[-1])
            out=np.exp(-np.exp(np.clip(z,-100,80))) if self.name=='02_weibull_aft' else np.exp(log_ndtr(-z))
        elif self.name=='06_xgboost_aft':
            import xgboost as xgb
            from scipy.special import ndtr
            mu=self.model.predict(xgb.DMatrix(x), output_margin=True)
            z=(np.log(times)[None,:]-mu[:,None])/self.c['aft_scale']
            dist=self.c['distribution']
            out=1-ndtr(z) if dist=='normal' else (expit(-z) if dist=='logistic' else np.exp(-np.exp(np.clip(z,-100,80))))
        elif self.name in ['01_elastic_net_cox','05_random_survival_forest']:
            chunks=[]
            for start in range(0,len(x),2048):
                functions=self.model.predict_survival_function(x[start:start+2048])
                values=[]
                for fun in functions:
                    bounded=np.clip(times,fun.x[0],fun.x[-1])
                    value=np.atleast_1d(fun(bounded)); value[times<fun.x[0]]=1
                    values.append(value)
                chunks.append(np.array(values))
            out=np.vstack(chunks)
        else:
            import torch
            with torch.no_grad():
                parts=[]
                for start in range(0,len(x),4096):
                    logits=self.net(torch.tensor(x[start:start+4096]))
                    if self.name=='08_mtlr': logits=torch.flip(torch.cumsum(torch.flip(logits,[1]),1),[1])
                    pmf=torch.softmax(logits,dim=1).numpy()
                    s=1-np.cumsum(pmf[:,:self.support],axis=1)
                    parts.append(s[:,np.minimum(times.astype(int),self.support)-1])
                out=np.vstack(parts)
        out=np.clip(out,0,1)
        out[:,times>self.support]=np.nan
        return out


def propose(trial, name):
    c={'model':name, 'features':trial.suggest_categorical('features',['basic','all','ranked','groups'])}
    if c['features']=='ranked': c['top_k']=trial.suggest_categorical('top_k',[8,16,24,40,64,96])
    if c['features']=='groups':
        for group in ['basic','cohort','space','topology','neighbors','history','calendar']:
            c['group_'+group]=trial.suggest_categorical('group_'+group,[False,True])
    if name=='01_elastic_net_cox':
        c.update(alpha=trial.suggest_float('alpha',1e-5,.1,log=True),l1_ratio=trial.suggest_float('l1_ratio',.05,1.))
    elif name in ['02_weibull_aft','03_lognormal_aft']:
        c['penalty']=trial.suggest_float('penalty',1e-5,1.,log=True)
    elif name=='04_spline_survival':
        c.update(knots=trial.suggest_int('knots',3,10),C=trial.suggest_float('C',1e-4,10,log=True))
    elif name=='05_random_survival_forest':
        c.update(trees=trial.suggest_categorical('trees',[100,200,350]),depth=trial.suggest_int('depth',4,12),
            leaf=trial.suggest_categorical('leaf',[25,60,150,400]),max_features=trial.suggest_categorical('max_features',['sqrt',.3,.7]),
            sample=trial.suggest_float('sample',.4,1.))
    elif name=='06_xgboost_aft':
        c.update(distribution=trial.suggest_categorical('distribution',['normal','logistic','extreme']),
            aft_scale=trial.suggest_float('aft_scale',.25,3.,log=True),trees=trial.suggest_int('trees',150,800,step=50),
            depth=trial.suggest_int('depth',2,7),lr=trial.suggest_float('lr',.015,.15,log=True),
            child=trial.suggest_float('child',5,100,log=True),sample=trial.suggest_float('sample',.6,1),
            colsample=trial.suggest_float('colsample',.5,1),l2=trial.suggest_float('l2',.1,100,log=True),
            l1=trial.suggest_float('l1',1e-5,10,log=True))
    elif name=='07_catboost_hazard':
        c.update(trees=trial.suggest_int('trees',200,1000,step=100),depth=trial.suggest_int('depth',3,7),
            lr=trial.suggest_float('lr',.015,.15,log=True),l2=trial.suggest_float('l2',1,100,log=True),
            random_strength=trial.suggest_float('random_strength',0,3))
    elif name in ['08_mtlr','09_deephit']:
        c.update(width=trial.suggest_categorical('width',[32,64,128]),layers=trial.suggest_int('layers',1,3),
            dropout=trial.suggest_float('dropout',0,.4),lr=trial.suggest_float('lr',1e-4,.01,log=True),
            decay=trial.suggest_float('decay',1e-6,.01,log=True),epochs=trial.suggest_categorical('epochs',[30,60,100]),
            batch=trial.suggest_categorical('batch',[2048,8192]))
        if name=='09_deephit': c.update(alpha=trial.suggest_float('alpha',.5,1),sigma=trial.suggest_float('sigma',.1,1))
    else:
        c.update(cohort_width=trial.suggest_categorical('cohort_width',[1,2,5,10]),
            lr=trial.suggest_float('lr',.003,.05,log=True),epochs=trial.suggest_categorical('epochs',[30,60,100]),
            fixed_penalty=trial.suggest_float('fixed_penalty',1e-5,.1,log=True),
            group_penalty=trial.suggest_float('group_penalty',1e-5,.1,log=True),
            nested_multiplier=trial.suggest_float('nested_multiplier',1,20,log=True))
    return c
