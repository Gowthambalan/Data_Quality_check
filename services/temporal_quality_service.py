import pandas as pd
import numpy as np
from datetime import datetime

class TemporalQualityService:
    def __init__(self):
        pass

    #  Helper: Detect valid datetime columns
    def _get_datetime_columns(self, df, datetime_columns):
        detected = []
        for col in datetime_columns:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().mean() > 0.5:
                    detected.append(col)
        return detected

    #  1. Timestamp Accuracy
    def timestamp_accuracy(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            total = len(parsed)
            valid = parsed.notna().sum()
            accuracy = round((valid / total) * 100, 2) if total > 0 else 0
            results[col] = {
                "accuracy_%": accuracy,
                "status": "Issue" if accuracy < 90 else "OK"
            }
        return results

    #  2. Temporal Continuity (Gap detection)
    def temporal_continuity(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce').dropna().sort_values()
            if len(parsed) < 2:
                results[col] = {"continuity_%": 0, "status": "Issue"}
                continue

            diffs = parsed.diff().dt.total_seconds().dropna()
            median_gap = diffs.median() if len(diffs) > 0 else 0
            gap_ratio = (diffs > median_gap * 2).mean() if median_gap > 0 else 0
            continuity = round((1 - gap_ratio) * 100, 2)

            results[col] = {
                "continuity_%": continuity,
                "status": "Issue" if continuity < 88 else "OK"
            }
        return results

    #  3. Time Zone Consistency
    def time_zone_consistency(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce')

            # Get timezone info for each row (if available)
            tz_series = parsed.apply(lambda x: x.tzinfo if pd.notna(x) else None)

            # Count unique non-null timezones
            unique_tz = tz_series.dropna().unique()

            if len(unique_tz) == 0:
                # No timezone info at all → assume consistent (but naive)
                consistency = 100.0
                status = "OK"
            elif len(unique_tz) == 1:
                # One timezone → consistent
                consistency = 100.0
                status = "OK"
            else:
                # Multiple timezones found → inconsistency
                consistency = 70.0
                status = "Issue"

            results[col] = {
                "timezone_consistency_%": consistency,
                "status": status
            }
        return results


    # 4. Temporal Granularity (Regularity of intervals)
    def temporal_granularity(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce').dropna().sort_values()
            diffs = parsed.diff().dt.total_seconds().dropna()
            if len(diffs) == 0:
                granularity = 0
            else:
                std_ratio = diffs.std() / diffs.mean() if diffs.mean() > 0 else np.inf
                granularity = round(max(0, (1 - std_ratio) * 100), 2)

            results[col] = {
                "granularity_%": granularity,
                "status": "Issue" if granularity < 85 else "OK"
            }
        return results

    # 5. Freshness Score
    def freshness_score(self, df, datetime_columns):
        results = {}
        now = datetime.now()
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            latest = parsed.max()

            if pd.isna(latest):
                score = 0
            else:
                diff_days = (now - latest).days
                score = max(0, 100 - diff_days)

            results[col] = {
                "freshness_%": round(score, 2),
                "status": "Issue" if score < 85 else "OK"
            }
        return results

    # ✅ 6. Temporal Pattern Detection (Better than fixed 75%)
    def temporal_pattern(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce').dropna()
            if parsed.empty:
                results[col] = {"pattern_score_%": 0, "status": "Issue"}
                continue

            # Hour-based cyclic pattern assumption
            hour_counts = parsed.dt.hour.value_counts(normalize=True)
            pattern_score = round((1 - hour_counts.std()) * 100, 2)

            results[col] = {
                "pattern_score_%": pattern_score,
                "status": "Issue" if pattern_score < 75 else "OK"
            }
        return results

    # ✅ 7. Seasonality Detection (Daily/Weekly/Monthly repetitions)
    def seasonality_detection(self, df, datetime_columns):
        results = {}
        for col in datetime_columns:
            parsed = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(parsed) < 10:
                results[col] = {"seasonality_%": 0, "status": "Issue"}
                continue

            # Test DAILY seasonality (count per hour per day)
            daily_pattern = parsed.groupby(parsed.dt.hour).size()
            if daily_pattern.mean() > 0:
                seasonality_strength = round((1 - daily_pattern.std() / daily_pattern.mean()) * 100, 2)
            else:
                seasonality_strength = 0

            results[col] = {
                "seasonality_%": seasonality_strength,
                "status": "Issue" if seasonality_strength < 70 else "OK"
            }
        return results
