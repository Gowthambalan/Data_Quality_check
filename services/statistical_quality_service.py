# services/statistical_quality_service.py

import pandas as pd
import numpy as np
from scipy import stats

class StatisticalQualityService:

    def distribution_normality(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            if len(df[col].dropna()) > 3:
                stat, p_val = stats.shapiro(df[col].dropna()[:5000])
                result[col] = {
                    "p_value": float(round(p_val, 5)),
                    "status": "Non-normal" if p_val < 0.01 else "Normal"
                }
        return result

    def outlier_score(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            percent = (len(outliers) / len(df)) * 100
            result[col] = {
                "outlier_%": float(round(percent, 2)),
                "status": "Issue" if percent > 5 else "OK"
            }
        return result

    def variance_stability(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            chunks = np.array_split(df[col].dropna(), 5)
            variances = [chunk.var() for chunk in chunks if len(chunk) > 1]
            if len(variances) > 1:
                cv = np.std(variances) / np.mean(variances) if np.mean(variances) != 0 else np.nan
                result[col] = {
                    "cv_variance": float(round(cv, 3)),
                    "status": "Unstable" if cv > 0.5 else "Stable"
                }
        return result


    def skewness_quality(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 2 or data.nunique() == 1:
                result[col] = {
                    "skew": None,
                    "status": "No Data / Constant Values"
                }
            else:
                skew = stats.skew(data)
                result[col] = {
                    "skew": float(round(skew, 3)),
                    "status": "High Skew" if abs(skew) > 2 else "OK"
                }
        return result


    def kurtosis_quality(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 2 or data.nunique() == 1:
                result[col] = {
                    "kurtosis": None,
                    "status": "No Data / Constant Values"
                }
            else:
                kurt = stats.kurtosis(data)
                result[col] = {
                    "kurtosis": float(round(kurt, 3)),
                    "status": "High Kurtosis" if abs(kurt) > 7 else "OK"
                }
        return result



    def coefficient_variation(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            cv = std / mean if mean != 0 else np.nan

            if pd.isna(cv):  # If CV cannot be calculated
                status = "N/A"
                cv_value = None   # or keep np.nan
            else:
                status = "High Variability" if cv > 2 else "OK"
                cv_value = float(round(cv, 3))

            result[col] = {
                "cv": cv_value,
                "status": status
            }
        return result


    def statistical_anomalies(self, df, numeric_cols):
        result = {}
        for col in numeric_cols:
            median = df[col].median()
            spikes = df[df[col] > median * 5]
            drops = df[df[col] < median * 0.2]
            result[col] = {
                "spikes": int(len(spikes)),
                "drops": int(len(drops)),
                "status": "Anomalies Found" if len(spikes) + len(drops) > 0 else "OK"
            }
        return result


    def range_conformity(self, df, value_ranges):
        result = {}
        for col, (min_val, max_val) in value_ranges.items():
            if col in df.columns:
                total = len(df[col].dropna())
                if total == 0:
                    result[col] = {"conformity_%": None, "status": "No Data"}
                    continue
                in_range = df[col].between(min_val, max_val, inclusive="both").sum()
                conformity = (in_range / total) * 100
                result[col] = {
                    "conformity_%": float(round(conformity, 2)),
                    "status": "Issue" if conformity < 90 else "OK"
                }
        return result



    

