import pandas as pd
import numpy as np


class CoreQualityService:

    def completeness(self, df: pd.DataFrame):
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = round(((total_cells - missing_cells) / total_cells) * 100, 2)

        status = "CRITICAL" if (missing_cells / total_cells) > 0.2 else \
                 "WARNING" if (missing_cells / total_cells) > 0.1 else "GOOD"

        return {"score": completeness_score, "missing_cells": int(missing_cells), "status": status}

    def consistency(self, df: pd.DataFrame, datetime_columns: list):
        result = {}
        for col in datetime_columns:
            valid_dates = pd.to_datetime(df[col], errors='coerce')
            valid_ratio = valid_dates.notna().mean() * 100
            result[col] = {
                "validity_%": round(valid_ratio, 2),
                "status": "Inconsistent" if valid_ratio < 90 else "OK"
            }
        return result

    def accuracy(self, df: pd.DataFrame, numeric_columns: list):
        result = {}
        for col in numeric_columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            outliers_ratio = (np.abs(z_scores) > 3).mean() * 100
            result[col] = {
                "outlier_%": round(outliers_ratio, 2),
                "status": "Issue" if outliers_ratio > 10 else "OK"
            }
        return result

    def validity(self, df: pd.DataFrame, value_ranges: dict):
        result = {}
        for col, (min_val, max_val) in value_ranges.items():
            if col in df.columns:
                invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                result[col] = {
                    "invalid_count": int(invalid_count),
                    "status": "Invalid" if invalid_count > 0 else "Valid"
                }
        return result

    def timeliness(self, df: pd.DataFrame, datetime_columns: list):
        result = {}
        for col in datetime_columns:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            if df[col].dtype == 'datetime64[ns]':
                gaps = df[col].sort_values().diff().dt.total_seconds().dropna()
                if not gaps.empty:
                    median_gap = gaps.median()
                    large_gaps = gaps[gaps > (median_gap * 3)]
                    result[col] = {
                        "large_gaps": int(len(large_gaps)),
                        "status": "Issue" if len(large_gaps) > 0 else "OK"
                    }
        return result

    def uniqueness(self, df: pd.DataFrame):
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        score = 100 - ((duplicate_rows / total_rows) * 100)
        return {
            "duplicate_count": int(duplicate_rows),
            "uniqueness_score": round(score, 2),
            "status": "Issue" if duplicate_rows > 0 else "OK"
        }
