"""
data_processing.py
Reusable functions for loading, cleaning, and feature-engineering
the Walmart sales dataset (Milestone 1 & 2).
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Walmart CSV file."""
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True)
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Engineer time-based features used across models."""
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill time-ordered missing values; drop remaining nulls."""
    df = df.sort_values("Date").copy()
    df = df.ffill()
    return df.dropna()


def time_based_split(df: pd.DataFrame, date_col: str, test_size_weeks: int = 12):
    """Split a time-indexed dataframe into train/test by date (NOT random)."""
    cutoff = df[date_col].max() - pd.Timedelta(weeks=test_size_weeks)
    train = df[df[date_col] <= cutoff]
    test = df[df[date_col] > cutoff]
    return train, test
