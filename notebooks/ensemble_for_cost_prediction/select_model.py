"""Show or select a registered severity model without retraining or overwriting artifacts."""
from pathlib import Path
import argparse,json

HERE=Path(__file__).resolve().parent

def validate(entry):
    for field in ['config','predictions','fitted_model']:
        path=(HERE/entry[field]).resolve()
        if not path.is_relative_to(HERE.resolve()) or not path.is_file():
            raise ValueError('Invalid or missing model artifact: '+field)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select',help='Registered model ID, for example CMS001')
    args=parser.parse_args()
    registry=json.loads((HERE/'model_registry.json').read_text(encoding='utf-8'))['models']
    if args.select:
        if args.select not in registry:parser.error('Unknown ID. Available: '+', '.join(registry))
        selected=registry[args.select]
        validate(selected)
        (HERE/'latest_model.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    current=json.loads((HERE/'latest_model.json').read_text(encoding='utf-8'))
    validate(current)
    print(current['registry_id']+' - '+current['display_name']+' v'+current['version'])
    print('Recipe: '+current['model_id'])
    print('Fitted model: '+current['fitted_model'])
    print('Predictions: '+current['predictions'])

if __name__=='__main__':main()
