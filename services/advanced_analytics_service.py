import pandas as pd
import numpy as np

class AdvancedAnalyticsService:
    """Advanced Data Analytics Metrics for numerical columns in a DataFrame."""

    def correlation_quality(self, df, numeric_columns):
        """Check correlation strength between numeric columns."""
        try:
            df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            corr = df.corr().abs()
            n = len(numeric_columns)
            if n < 2:
                return None

            # Count strong correlations (unique pairs only, exclude diagonal)
            strong_pairs = np.sum(np.triu(corr > 0.7, k=1))
            total_pairs = n * (n - 1) / 2

            score = (strong_pairs / total_pairs) * 100 if total_pairs else 0
            return round(score, 2)
        except Exception as e:
            print(f"[correlation_quality] Error: {e}")
            return None

    def trend_consistency(self, df, numeric_columns):
        """Detect upward or downward trend consistency."""
        try:
            trends = {}
            for col in numeric_columns:
                series = pd.to_numeric(df[col], errors='coerce')
                if series.isnull().all():
                    continue
                x = np.arange(len(series))
                y = series.fillna(method='ffill')
                coef = np.polyfit(x, y, 1)[0]  # slope
                trends[col] = coef

            if not trends:
                return None

            consistent = sum(1 for val in trends.values() if abs(val) > 0.01)
            return round((consistent / len(trends)) * 100, 2)
        except Exception as e:
            print(f"[trend_consistency] Error: {e}")
            return None

    def volatility_score(self, df, numeric_columns):
        """Calculate volatility using coefficient of variation (std/mean)."""
        try:
            df_num = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            means = df_num.mean().replace(0, np.nan)
            vols = (df_num.std() / means).replace([np.inf, -np.inf], np.nan).dropna()
            if vols.empty:
                return None
            avg_vol = vols.mean()
            return round((1 - avg_vol) * 100, 2)
        except Exception as e:
            print(f"[volatility_score] Error: {e}")
            return None

    def rate_of_change(self, df, numeric_columns):
        """Average percentage rate of change between rows."""
        roc = {}
        for col in numeric_columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce')
                pct = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                roc[col] = round(pct.mean() * 100, 2) if len(pct) else None
            except Exception as e:
                print(f"[rate_of_change] Error in column {col}: {e}")
                roc[col] = None
        return roc

    def anomaly_score(self, df, numeric_columns):
        """Simple anomaly detection using Z-score method."""
        try:
            df_num = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            mean = df_num.mean()
            std = df_num.std().replace(0, np.nan)
            z_scores = np.abs((df_num - mean) / std)
            anomalies = (z_scores > 3).sum().sum()
            valid_values = df_num.count().sum()
            if valid_values == 0:
                return None
            score = (1 - (anomalies / valid_values)) * 100
            return round(score, 2)
        except Exception as e:
            print(f"[anomaly_score] Error: {e}")
            return None

    def predictability_score(self, df, numeric_columns):
        """Check predictability using autocorrelation (lag=1)."""
        try:
            scores = []
            for col in numeric_columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) < 2:
                    continue
                score = series.autocorr(lag=1)
                if not np.isnan(score):
                    scores.append(abs(score))
            return round(np.mean(scores) * 100, 2) if scores else None
        except Exception as e:
            print(f"[predictability_score] Error: {e}")
            return None

