import pandas as pd
import numpy as np
import math

class PrecisionQualityService:

    def _convert_to_python_types(self, obj):
        """Recursively convert numpy types to native Python types and handle NaN/Inf"""
        if isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            # Handle NaN and Inf values
            if pd.isna(obj) or math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            else:
                return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray)):
            return self._convert_to_python_types(obj.tolist())
        elif pd.isna(obj):  # Handle pandas NA, NaT, etc.
            return None
        else:
            return obj

    def decimal_precision(self, df, numeric_columns: list):
        result = {}
        for col in numeric_columns:
            try:
                # Handle NaN values in the column
                clean_col = df[col].dropna()
                if clean_col.empty:
                    result[col] = {
                        "unique_decimal_counts": [],
                        "status": "No data"
                    }
                    continue
                    
                decimals = (
                    clean_col
                    .apply(lambda x: str(x) if pd.notnull(x) else None)
                    .dropna()
                    .apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
                )
                unique_decimal_counts = sorted(decimals.unique().tolist())
                result[col] = {
                    "unique_decimal_counts": unique_decimal_counts,
                    "status": "Issue" if len(unique_decimal_counts) > 3 else "OK"
                }
            except Exception as e:
                result[col] = {"error": str(e), "status": "Error"}
        return self._convert_to_python_types(result)

    def rounding_consistency(self, df, numeric_columns: list):
        result = {}
        for col in numeric_columns:
            try:
                # Handle NaN values
                clean_col = df[col].dropna()
                if clean_col.empty:
                    result[col] = {
                        "match_%": 0.0,
                        "status": "No data"
                    }
                    continue
                    
                rounded = clean_col.round()
                match_ratio = (np.isclose(clean_col, rounded) | np.isclose(clean_col * 2, np.round(clean_col * 2))).mean() * 100
                result[col] = {
                    "match_%": round(float(match_ratio), 2),
                    "status": "Issue" if match_ratio < 88 else "OK"
                }
            except Exception as e:
                result[col] = {"error": str(e), "status": "Error"}
        return self._convert_to_python_types(result)

    def significant_figures(self, df, numeric_columns: list):
        def count_sig_figs(x):
            try:
                # Handle NaN and infinite values
                if pd.isna(x) or math.isinf(x):
                    return 0
                s = f"{x:.10f}".rstrip('0').replace('.', '').lstrip('0')
                return len(s)
            except Exception:
                return 0
                
        result = {}
        for col in numeric_columns:
            try:
                # Handle NaN values
                clean_col = df[col].dropna()
                if clean_col.empty:
                    result[col] = {
                        "valid_%": 0.0,
                        "status": "No data"
                    }
                    continue
                    
                sig_figs = clean_col.apply(count_sig_figs)
                valid_ratio = (sig_figs >= 2).mean() * 100
                result[col] = {
                    "valid_%": round(float(valid_ratio), 2),
                    "status": "Issue" if valid_ratio < 85 else "OK"
                }
            except Exception as e:
                result[col] = {"error": str(e), "status": "Error"}
        return self._convert_to_python_types(result)

    def measurement_precision(self, df, numeric_columns: list):
        result = {}
        for col in numeric_columns:
            try:
                # Handle NaN values
                clean_col = df[col].dropna()
                if clean_col.empty:
                    result[col] = {
                        "most_common_decimals": None,
                        "precision_%": 0.0,
                        "status": "No data"
                    }
                    continue
                    
                decimals = (
                    clean_col
                    .apply(lambda x: str(x) if pd.notnull(x) else None)
                    .dropna()
                    .apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
                )
                if decimals.empty:
                    result[col] = {
                        "most_common_decimals": None,
                        "precision_%": 0.0,
                        "status": "No data"
                    }
                    continue
                    
                most_common = decimals.mode()[0] if not decimals.empty else None
                ratio = (decimals == most_common).mean() * 100 if most_common is not None else 0
                result[col] = {
                    "most_common_decimals": int(most_common) if most_common is not None else None,
                    "precision_%": round(float(ratio), 2),
                    "status": "Issue" if ratio < 90 else "OK"
                }
            except Exception as e:
                result[col] = {"error": str(e), "status": "Error"}
        return self._convert_to_python_types(result)

    def calculation_accuracy(self, df, base_cols: list = None):
        if not base_cols or len(base_cols) != 3:
            return {"error": "Provide exactly 3 columns: [a, b, c] where c should equal a+b"}

        col1, col2, col3 = base_cols
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            return {"error": f"Missing columns: {missing}"}

        try:
            # Remove rows with NaN values in any of the three columns
            clean_df = df[base_cols].dropna()
            if clean_df.empty:
                return {
                    "accuracy_%": 0.0,
                    "status": "No data"
                }
                
            valid = np.isclose(clean_df[col1] + clean_df[col2], clean_df[col3], atol=0.01).mean() * 100
            return self._convert_to_python_types({
                "accuracy_%": round(float(valid), 2),
                "status": "Issue" if valid < 92 else "OK"
            })
        except Exception as e:
            return {"error": str(e), "status": "Error"}