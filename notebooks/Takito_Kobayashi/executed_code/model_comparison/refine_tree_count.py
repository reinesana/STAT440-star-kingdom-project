"""Finalize common-round selection with actual CatBoost retraining."""
import run_experiment as experiment
from joint_final_selection import run

if __name__=='__main__':
    for model in ['xgboost','catboost','lightgbm']:
        experiment.THREADS=3 if model=='catboost' else 2
        run(model,[2023,2024,2025,2026])
