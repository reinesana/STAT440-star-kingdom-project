"""Apply the actual fitted bases and meta-model to prepared year-start features."""
from pathlib import Path
import argparse
import joblib
from threadpoolctl import threadpool_limits
from deployment import model_predict

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True,help='Prepared year-start feature CSV, with pipe_id/material/surface and the model feature columns')
    p.add_argument('--model',type=Path,default=HERE/'fitted_model_2027.joblib')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import pandas as pd
    model=joblib.load(args.model)
    features=pd.read_csv(args.features)
    if 'scenario_year' in features and not features.scenario_year.eq(2027).all():
        p.error('This frozen model expects scenario_year=2027')
    result,_=model_predict(model,features)
    result.to_csv(args.output,index=False)
    print(f'Predicted {len(result)} rows with 3 fitted bases trained through 2026 and the conditional stack.')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
