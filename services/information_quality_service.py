import pandas as pd
import numpy as np
import math

class InformationQualityService:
    def __init__(self):
        pass

    # 35. Entropy Score — measures information randomness
    def entropy_score(self, df):
        results = {}
        for col in df.columns:
            try:
                series = df[col].dropna().astype(str)
                if len(series) == 0:
                    results[col] = {"entropy_score": 0, "status": "Low entropy"}
                    continue

                probs = series.value_counts(normalize=True)
                entropy = -np.sum(probs * np.log2(probs))
                max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1
                normalized_entropy = round(entropy / max_entropy, 3)
                results[col] = {
                    "entropy_score": normalized_entropy,
                    "status": "Low entropy" if normalized_entropy < 0.1 else "OK"
                }
            except Exception:
                results[col] = {"entropy_score": 0, "status": "Issue"}
        return results

    # 36. Information Density — ratio of non-null cells
    def information_density(self, df):
        results = {}
        for col in df.columns:
            try:
                total = len(df[col])
                non_null = df[col].notna().sum()
                density = round((non_null / total) * 100, 2) if total > 0 else 0
                results[col] = {
                    "information_density_%": density,
                    "status": "Issue" if density < 80 else "OK"
                }
            except Exception:
                results[col] = {"information_density_%": 0, "status": "Issue"}
        return results

    # 37. Sparsity Score — complement of density
    def sparsity_score(self, df):
        results = {}
        for col in df.columns:
            try:
                total = len(df[col])
                non_null = df[col].notna().sum()
                density = (non_null / total) * 100 if total > 0 else 0
                sparsity = round(100 - density, 2)
                results[col] = {"sparsity_%": sparsity}
            except Exception:
                results[col] = {"sparsity_%": 100}
        return results

    # 38. Redundancy Score — detect duplicate information (simplified)
    def redundancy_score(self, df):
        results = {}
        for col in df.columns:
            try:
                dup_ratio = df[col].duplicated().mean() * 100
                redundancy = round(100 - dup_ratio, 2)
                results[col] = {
                    "redundancy_%": redundancy,
                    "status": "Issue" if redundancy < 90 else "OK"
                }
            except Exception:
                results[col] = {"redundancy_%": 0, "status": "Issue"}
        return results

    # 39. Compression Ratio — rough measure of compressibility
    def compression_ratio(self, df):
        results = {}
        for col in df.columns:
            try:
                series = df[col].dropna().astype(str)
                unique_ratio = len(series.unique()) / len(series) if len(series) > 0 else 0
                compression = round((1 - unique_ratio) * 100, 2)
                results[col] = {
                    "compression_ratio_%": compression,
                    "status": "Issue" if compression < 85 else "OK"
                }
            except Exception:
                results[col] = {"compression_ratio_%": 0, "status": "Issue"}
        return results
