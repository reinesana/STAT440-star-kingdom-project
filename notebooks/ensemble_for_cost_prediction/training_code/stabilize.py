"""Budget/annual-error balanced weights within an RMSE guardrail.

All historical-year weights use preceding-year out-of-time predictions only.
Fixed rules: 1% minimum / 70% maximum weight, balanced loss uses annual total
bias and variation in annual RMSE relative to the surface baseline; RMSE cap
is 1% above the best feasible blend using the same diversification bounds.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from train import CORE, OUT, MODELS, diagnostics, dump
from features import HERE


def balanced_weights(past):
    x=past[CORE].to_numpy(); y=past.cost.to_numpy(); years=past.event_year.to_numpy()
    scale=float(np.sqrt(np.mean(y*y)))
    xx=x/scale; yy=y/scale
    annual=[years==year for year in np.unique(years)]
    baseline=np.array([np.sqrt(np.mean((xx[m,0]-yy[m])**2)) for m in annual])
    uniform=np.full(len(CORE),1/len(CORE))
    bounds=[(.01,.70)]*len(CORE)
    equal={'type':'eq','fun':lambda w:w.sum()-1}
    def primary(w):
        err=xx@w-yy
        return float(np.mean([np.sqrt(np.mean(err[m]**2)) for m in annual]))
    optimal=minimize(primary,uniform,method='SLSQP',bounds=bounds,constraints=[equal],
                     options={'ftol':1e-11,'maxiter':1500})
    assert optimal.success,optimal.message
    limit=primary(optimal.x)*1.01
    def stability(w):
        p=xx@w; err=p-yy
        bias=np.array([p[m].sum()/yy[m].sum()-1 for m in annual])
        relative=np.array([np.sqrt(np.mean(err[m]**2)) for m in annual])/baseline
        return float(np.mean(abs(bias))+np.std(relative)+.05*np.sum((w-uniform)**2))
    balanced=minimize(stability,optimal.x,method='SLSQP',bounds=bounds,
        constraints=[equal,{'type':'ineq','fun':lambda w:limit-primary(w)}],
        options={'ftol':1e-10,'maxiter':2500})
    if not balanced.success or primary(balanced.x)>limit+1e-7:
        # Fall back to the feasible diversified RMSE optimum, with explicit status.
        w=optimal.x; status='fallback_to_feasible_rmse_optimum'
    else:w=balanced.x;status='balanced_optimization_passed'
    assert np.isclose(w.sum(),1) and min(w)>=.01-1e-7 and max(w)<=.70+1e-7
    return w,{'status':status,'best_feasible_past_mean_annual_RMSE':primary(optimal.x)*scale,
              'selected_past_mean_annual_RMSE':primary(w)*scale,'RMSE_cap':limit*scale,
              'past_stability_objective':stability(w)}


def main():
    oof=pd.read_csv(OUT/'out_of_time_predictions.csv')
    rows=[];checks=[]
    for year in range(2023,2027):
        past=oof[oof.event_year<year]; current=oof.event_year==year
        assert past.event_year.max()<year
        w,status=balanced_weights(past)
        oof.loc[current,'ensemble_balanced']=oof.loc[current,CORE].to_numpy()@w
        checks.append({'test_year':year,'weight_training_end':year-1,**status})
        for name,weight in zip(CORE,w):rows.append({'test_year':year,'candidate':name,'weight':weight,'weight_training_end':year-1})
    oof.to_csv(OUT/'out_of_time_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'balanced_past_only_weights.csv',index=False)
    dump(OUT/'balanced_optimization_checks.json',checks)
    # Preserve the original comparison, then append the balanced candidate using
    # the exact same held-out populations and diagnostic definitions.
    metrics=pd.read_csv(OUT/'annual_metrics.csv')
    extra=[]
    d=pd.read_csv(HERE/'data/training_rich.csv')
    for year,q in oof[oof.event_year>=2023].groupby('event_year'):
        threshold=d.loc[d.event_year<year,'cost'].quantile(.99)
        p=q.ensemble_balanced;err=p-q.cost; tail=q.cost>=threshold
        extra.append({'model':'ensemble_balanced','year':int(year),'n_test':len(q),
              'RMSE':np.sqrt(np.mean(err**2)),'MAE':np.mean(abs(err)),
              'actual_total':q.cost.sum(),'predicted_total':p.sum(),
              'total_bias_pct':100*(p.sum()/q.cost.sum()-1),
              'high_cost_threshold_prior_only':threshold,'high_cost_RMSE':np.sqrt(np.mean(err[tail]**2))})
    metrics=pd.concat([metrics[metrics.model!='ensemble_balanced'],pd.DataFrame(extra)],ignore_index=True)
    metrics.to_csv(OUT/'annual_metrics.csv',index=False)
    summary=metrics.groupby('model').agg(mean_annual_RMSE=('RMSE','mean'),annual_RMSE_sd=('RMSE','std'),
        worst_annual_RMSE=('RMSE','max'),mean_absolute_annual_total_bias_pct=('total_bias_pct',lambda x:abs(x).mean()),
        worst_absolute_annual_total_bias_pct=('total_bias_pct',lambda x:abs(x).max()))
    q=oof[oof.event_year>=2023]
    summary['pooled_RMSE']=[np.sqrt(np.mean((q[name]-q.cost)**2)) for name in summary.index]
    summary['pooled_total_bias_pct']=[100*(q[name].sum()/q.cost.sum()-1) for name in summary.index]
    baseline=metrics[metrics.model=='prior_surface_mean'].set_index('year').RMSE
    relative=metrics.assign(relative_RMSE=metrics.RMSE/metrics.year.map(baseline))
    summary['annual_relative_RMSE_sd']=relative.groupby('model').relative_RMSE.std()
    summary.sort_values('mean_annual_RMSE').to_csv(OUT/'summary_metrics.csv')
    final,status=balanced_weights(oof)
    weights=json.loads((MODELS/'ensemble_weights.json').read_text())
    weights['ensemble_balanced']=dict(zip(CORE,final.tolist()))
    dump(MODELS/'ensemble_weights.json',weights)
    contract=json.loads((MODELS/'training_contract.json').read_text())
    contract.update(default_ensemble='ensemble_balanced',default_is_provisional=False,
        balanced_rule='1% minimum and 70% maximum per model; past mean annual RMSE <= 1.01x best blend within those bounds; minimize absolute annual total bias + relative annual RMSE SD + .05 uniform shrinkage',
        final_balanced_fit=status,adoption_reason='User requests proactive diversification and both budget and annual-error stability; no promise of future improvement.')
    dump(MODELS/'training_contract.json',contract)
    forecast=pd.read_csv(OUT/'forecast_costs_2027.csv')
    forecast['ensemble_balanced']=forecast[CORE].to_numpy()@final
    forecast.to_csv(OUT/'forecast_costs_2027.csv',index=False)
    print(summary.sort_values('mean_annual_RMSE').round(3).to_string())
    print('Final balanced weights:',dict(zip(CORE,final.round(4))))


if __name__=='__main__':main()
