import pandas as pd
import numpy as np


class SemanticQualityService:
    def __init__(self, config):
        self.config = config
        self.identifier_columns = config.get("identifier_columns", [])
        self.domain_rules = config.get("domain_rules", {})
        self.cross_field_rules = config.get("cross_field_rules", [])
        self.metadata_columns = config.get("metadata_columns", [])
        self.frequency_columns = config.get("frequency_columns", [])
        self.powerfactor_columns = config.get("powerfactor_columns", [])

    #  Business Rule Compliance → Rule validation
    def business_rule_compliance(self, df: pd.DataFrame):
        freq_cols = [col for col in self.frequency_columns if col in df.columns]
        pf_cols = [col for col in self.powerfactor_columns if col in df.columns]

        rule_violations = 0

        # Frequency column validation
        for col in freq_cols:
            std_val = df[col].std()
            if pd.notna(std_val) and std_val > 0.2:
                rule_violations += 1

        #  Power Factor validation
        for col in pf_cols:
            invalid_pf = df[(df[col] < -1) | (df[col] > 1)]
            if not invalid_pf.empty:
                rule_violations += len(invalid_pf)

        total_rules = len(freq_cols) + len(pf_cols)
        compliance_score = 1 - (rule_violations / (total_rules + 1e-6))
        status = "Issue" if compliance_score < 0.95 else "OK"

        return {
            "rule_violations": rule_violations,
            "compliance_score": round(compliance_score, 2),
            "status": status
        }

    #  Referential Integrity
    def referential_integrity(self, df: pd.DataFrame):
        inconsistent = []
        for col in self.identifier_columns:
            if col in df.columns:
                null_ratio = df[col].isna().mean()
                if null_ratio > 0.05:
                    inconsistent.append(col)
        score = 1 - (len(inconsistent) / len(self.identifier_columns)) if self.identifier_columns else 1
        status = "Issue" if score < 0.95 else "OK"
        return {"inconsistent_columns": inconsistent, "integrity_score": round(score, 2), "status": status}


    def cross_field_validation(self, df: pd.DataFrame):
        issues = []

        for pair in self.cross_field_rules:
            if len(pair) == 2 and all(col in df.columns for col in pair):
                # Compare: first column should NOT be greater than second
                inconsistent_rows = (df[pair[0]] > df[pair[1]]).sum()
                if inconsistent_rows > 0:
                    issues.append(f"{pair[0]}>{pair[1]}")

        score = 1 - (len(issues) / max(len(self.cross_field_rules), 1))
        status = "Issue" if score < 0.9 else "OK"

        return {
            "violations": issues,
            "cross_field_score": round(score, 2),
            "status": status
        }


    def domain_value_validity(self, df: pd.DataFrame):
        invalid_columns = []

        for col, rule in self.domain_rules.items():
            if col in df.columns:
                # Numeric range-based rule → [min, max]
                if isinstance(rule, list) and len(rule) == 2:
                    min_val, max_val = rule
                    if ((df[col] < min_val) | (df[col] > max_val)).any():
                        invalid_columns.append(col)

                # Category-based rule (allowed values)
                else:
                    if (~df[col].isin(rule)).any():
                        invalid_columns.append(col)

        # Score calculation
        score = 1 - (len(invalid_columns) / max(len(self.domain_rules), 1))
        status = "Issue" if score < 0.88 else "OK"

        return {
            "invalid_columns": invalid_columns,   # ✅ Only column names shown
            "domain_validity_score": round(score, 2),
            "status": status
        }


    #  Semantic Consistency
    def semantic_consistency(self, df: pd.DataFrame):
        voltage_cols = [col for col in df.columns if "volt" in col.lower()]
        current_cols = [col for col in df.columns if "current" in col.lower()]
        inconsistent_rows = 0

        for vcol in voltage_cols:
            for ccol in current_cols:
                inconsistent_rows += (df[vcol] * df[ccol] < 0).sum()

        total_combinations = len(voltage_cols) * len(current_cols)
        score = 1 - (inconsistent_rows / (len(df) * max(total_combinations, 1))) if len(df) > 0 else 1
        status = "Issue" if score < 0.92 else "OK"
        return {"semantic_inconsistencies": inconsistent_rows, "semantic_score": round(score, 2), "status": status}

    #  Data Lineage Quality
    def data_lineage_quality(self, df: pd.DataFrame):
        lineage_tracked = [col for col in self.metadata_columns if col in df.columns]
        lineage_score = len(lineage_tracked) / max(len(self.metadata_columns), 1)
        status = "Issue" if lineage_score < 0.85 else "OK"
        return {"tracked_columns": lineage_tracked, "lineage_score": round(lineage_score, 2), "status": status}
