import pandas as pd
import numpy as np
import re


class StructuralQualityService:

    def __init__(self, config):
        self.config = config
        self.expected_schema = config.get("expected_schema", [])
        self.identifier_columns = config.get("identifier_columns", [])
        self.datetime_columns = config.get("datetime_columns", [])

    #  Schema Conformity → Expected column detection
    def schema_conformity(self, df: pd.DataFrame):
        missing_cols = [col for col in self.expected_schema if col not in df.columns]
        conformity_score = 1 - (len(missing_cols) / len(self.expected_schema)) if self.expected_schema else 1
        status = "Issue" if missing_cols else "OK"
        return {"missing_columns": missing_cols, "conformity_score": round(conformity_score, 2), "status": status}

    #  Data Type Consistency → Type validation
    def data_type_consistency(self, df: pd.DataFrame):
        inconsistent_columns = []
        for col in df.columns:
            inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
            unique_types = df[col].map(type).nunique()
            if unique_types > 1:
                inconsistent_columns.append(col)

        consistency_score = 1 - (len(inconsistent_columns) / len(df.columns))
        status = "Issue" if consistency_score < 0.9 else "OK"
        return {"inconsistent_columns": inconsistent_columns, "consistency_score": round(consistency_score, 2), "status": status}

    #  Naming Convention → Regex pattern check
    def naming_convention(self, df: pd.DataFrame):
        pattern = re.compile(r"^[a-zA-Z0-9_]+$")
        invalid_names = [col for col in df.columns if not pattern.match(col) or len(col) > 50]
        score = 1 - (len(invalid_names) / len(df.columns))
        status = "Issue" if score < 0.7 else "OK"
        return {"invalid_columns": invalid_names, "naming_score": round(score, 2), "status": status}

    #  Structural Integrity → Compare cardinalities of identifier columns
    def structural_integrity(self, df: pd.DataFrame):
        if not self.identifier_columns:
            return {"status": "No identifier columns configured"}
        cardinalities = {col: df[col].nunique() for col in self.identifier_columns if col in df.columns}
        consistent = len(set(cardinalities.values())) == 1
        status = "OK" if consistent else "Inconsistent"
        return {"cardinalities": cardinalities, "status": status}

    #  Cardinality Quality → Unique ratio
    def cardinality_quality(self, df: pd.DataFrame):
        ratios = {}
        for col in self.identifier_columns:
            if col in df.columns:
                ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                ratios[col] = round(ratio, 2)
        low_cardinality = [col for col, val in ratios.items() if val < 0.5]
        status = "OK" if not low_cardinality else "Low cardinality issue"
        return {"unique_ratios": ratios, "low_cardinality_columns": low_cardinality, "status": status}

    #  Schema Drift → Detect unexpected or missing columns
    def schema_drift(self, df: pd.DataFrame):
        current_cols = set(df.columns)
        expected_cols = set(self.expected_schema)
        new_cols = list(current_cols - expected_cols)
        missing_cols = list(expected_cols - current_cols)
        status = "OK" if not new_cols and not missing_cols else "Drift Detected"
        return {"new_columns": new_cols, "missing_columns": missing_cols, "status": status}

    #  Metadata Completeness → Check min columns and datetime
    def metadata_completeness(self, df: pd.DataFrame):
        datetime_present = any(col in df.columns for col in self.datetime_columns)
        sufficient_columns = len(df.columns) >= 5
        status = "OK" if datetime_present and sufficient_columns else "Issue"
        return {"datetime_present": datetime_present, "column_count": len(df.columns), "status": status}
