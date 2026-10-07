"""Step 1: load the shared cost data and split it into the four year folds.

Fold 1: train 2019-2022, test 2023
Fold 2: train 2019-2023, test 2024
Fold 3: train 2019-2024, test 2025
Fold 4: train 2019-2025, test 2026
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 440
DATA = Path(__file__).resolve().parents[2] / 'data' / 'train_predict_cost.csv'
TEST_YEARS = (2023, 2024, 2025, 2026)


def load_data(path=DATA):
    return pd.read_csv(path, low_memory=False)


def split_folds(df, test_years=TEST_YEARS):
    """Yield (test_year, train, test) using the leak year, never row order or pipe ID."""
    for year in test_years:
        train = df[(df['event_year'] >= 2019) & (df['event_year'] < year)]
        test = df[df['event_year'] == year]
        yield year, train, test


def xy(frame, features):
    """Return X, y (dollars) and y_log (log1p dollars) for one fold half."""
    y = frame['cost'].to_numpy(float)
    return frame[features], y, np.log1p(y)


if __name__ == '__main__':
    data = load_data()
    for year, train, test in split_folds(data):
        print(f'test {year}: train {train.event_year.min()}-{train.event_year.max()} '
              f'({len(train):,} rows), test {len(test):,} rows')
