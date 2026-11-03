"""Lookup known inventory IDs, or score a feature CSV using a saved estimator."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from data import BASE, Dataset, raw
from models import MODELS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--pipe-id',nargs='+')
    parser.add_argument('--features-csv',help='Override full feature rows; supply pipe_id and selected features, plus age/origin_year for hazard models')
    parser.add_argument('--output',default='lookup_predictions.csv')
    args=parser.parse_args()
    if args.features_csv:
        from experiment import export_prediction
        frame=pd.read_csv(args.features_csv)
        model=joblib.load(BASE/'models'/args.model/'2027.joblib')
        missing=set(model.columns+['pipe_id','age','origin_year','lay_missing','material'])-set(frame)
        if missing:raise ValueError(f'Missing feature columns: {sorted(missing)}')
        out=BASE/'cache'/'custom_prediction';out.mkdir(parents=True,exist_ok=True)
        pred=export_prediction(model,frame,2027,out)
    else:
        if not args.pipe_id:parser.error('Supply --pipe-id or --features-csv')
        path=BASE/args.model/'2027'/'predictions.csv.gz'
        saved=pd.read_csv(path).set_index('pipe_id')
        p,h=raw();inventory=p.set_index('Pipe ID');historical=set(h['Pipe ID'])
        records=[]
        for pipe_id in args.pipe_id:
            if pipe_id in saved.index:
                row=saved.loc[pipe_id].to_dict();row['status']='predicted'
            elif pipe_id not in inventory.index:row={'status':'unknown_pipe_id'}
            elif pipe_id in historical:row={'status':'already_replaced_after_historical_leak'}
            elif inventory.loc[pipe_id,'Material']=='polyurethane':row={'status':'native_polyurethane_no_leak_under_rules'}
            else:row={'status':'not_supported'}
            records.append({'pipe_id':pipe_id,**row})
        pred=pd.DataFrame(records)
    pred.to_csv(args.output,index=False)
    columns=[c for c in ['pipe_id','status','failure_year_q50','failure_year_q10','failure_year_q90','p_failure_next_year','median_status'] if c in pred]
    print(pred[columns].to_string(index=False))


if __name__=='__main__':main()
