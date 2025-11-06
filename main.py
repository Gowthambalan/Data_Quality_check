from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import yaml
import numpy as np

from services.core_quality_service import CoreQualityService
from services.statistical_quality_service import StatisticalQualityService
from services.structural_quality_service import StructuralQualityService
from services.semantic_quality_service import SemanticQualityService
from services.temporal_quality_service import TemporalQualityService
from services.information_quality_service import InformationQualityService
from services.precision_quality_service import PrecisionQualityService
from services.advanced_analytics_service import AdvancedAnalyticsService

app = FastAPI(title="Data Quality Validation API")

# =========================
# Load YAML Config
# =========================
with open("config/column_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# =========================
# Initialize Services
# =========================
core_service = CoreQualityService()
stat_service = StatisticalQualityService()
struct_service = StructuralQualityService(config)
semantic_service = SemanticQualityService(config)
temporal_service = TemporalQualityService()
info_service = InformationQualityService()
precision_service = PrecisionQualityService()
analytics_service = AdvancedAnalyticsService()

# =========================
# Utility: Read Uploaded File
# =========================
def read_file(file: UploadFile):
    if file.filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file.file)
    elif file.filename.endswith(".csv"):
        return pd.read_csv(file.file)
    else:
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls allowed")


# =========================
# Utility: Clean NaN values
# =========================
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# =========================
#  Core Quality API
# =========================
@app.post("/data-quality")
async def check_core_quality(file: UploadFile = File(...)):
    df = read_file(file)

    numeric_columns = df.select_dtypes(include=["int", "float"]).columns.tolist()
    datetime_columns = [col for col in config["datetime_columns"] if col in df.columns]

    return {
        "completeness": core_service.completeness(df),
        "consistency": core_service.consistency(df, datetime_columns),
        "accuracy": core_service.accuracy(df, numeric_columns),
        "validity": core_service.validity(df, config["value_ranges"]),
        "timeliness": core_service.timeliness(df, datetime_columns),
        "uniqueness": core_service.uniqueness(df)
    }


# =========================
#  Statistical Quality API
# =========================
@app.post("/statistical-quality")
async def check_statistical_quality(file: UploadFile = File(...)):
    df = read_file(file)
    numeric_columns = df.select_dtypes(include=["int", "float"]).columns.tolist()

    result = {
        "distribution_normality": stat_service.distribution_normality(df, numeric_columns),
        "outlier_score": stat_service.outlier_score(df, numeric_columns),
        "variance_stability": stat_service.variance_stability(df, numeric_columns),
        "skewness_quality": stat_service.skewness_quality(df, numeric_columns),
        "kurtosis_quality": stat_service.kurtosis_quality(df, numeric_columns),
        "range_conformity": stat_service.range_conformity(df, config["value_ranges"]),
        "coefficient_variation": stat_service.coefficient_variation(df, numeric_columns),
        "statistical_anomalies": stat_service.statistical_anomalies(df, numeric_columns)
    }

    return clean_nan(result)


# =========================
#  Structural Quality API
# =========================
@app.post("/structural-quality")
async def check_structural_quality(file: UploadFile = File(...)):
    df = read_file(file)

    result = {
        "schema_conformity": struct_service.schema_conformity(df),
        "data_type_consistency": struct_service.data_type_consistency(df),
        "naming_convention": struct_service.naming_convention(df),
        "structural_integrity": struct_service.structural_integrity(df),
        "cardinality_quality": struct_service.cardinality_quality(df),
        "schema_drift": struct_service.schema_drift(df),
        "metadata_completeness": struct_service.metadata_completeness(df)
    }

    return clean_nan(result)


@app.post("/semantic-quality")
async def check_semantic_quality(file: UploadFile = File(...)):
    df = read_file(file)

    result = {
        "business_rule_compliance": semantic_service.business_rule_compliance(df),
        "referential_integrity": semantic_service.referential_integrity(df),
        "cross_field_validation": semantic_service.cross_field_validation(df),
        "domain_value_validity": semantic_service.domain_value_validity(df),
        "semantic_consistency": semantic_service.semantic_consistency(df),
        "data_lineage_quality": semantic_service.data_lineage_quality(df)
    }

    return clean_nan(result)


@app.post("/temporal-quality")
async def check_temporal_quality(file: UploadFile = File(...)):
    df = read_file(file)
    datetime_columns = [col for col in config["datetime_columns"] if col in df.columns]

    temporal_service = TemporalQualityService()
    result = {
        "timestamp_accuracy": temporal_service.timestamp_accuracy(df, datetime_columns),
        "temporal_continuity": temporal_service.temporal_continuity(df, datetime_columns),
        "time_zone_consistency": temporal_service.time_zone_consistency(df, datetime_columns),
        "temporal_granularity": temporal_service.temporal_granularity(df, datetime_columns),
        "freshness_score": temporal_service.freshness_score(df, datetime_columns),
        "temporal_pattern": temporal_service.temporal_pattern(df, datetime_columns),
        "seasonality_detection": temporal_service.seasonality_detection(df, datetime_columns)
    }
    return result


@app.post("/information-quality")
async def check_information_quality(file: UploadFile = File(...)):
    df = read_file(file)
    info_service = InformationQualityService()

    result = {
        "entropy_score": info_service.entropy_score(df),
        "information_density": info_service.information_density(df),
        "sparsity_score": info_service.sparsity_score(df),
        "redundancy_score": info_service.redundancy_score(df),
        "compression_ratio": info_service.compression_ratio(df)
    }
    return result

@app.post("/precision-quality")
async def check_precision_quality(file: UploadFile = File(...)):
    df = read_file(file)
    numeric_columns = df.select_dtypes(include=["int", "float"]).columns.tolist()

    return {
        "decimal_precision": precision_service.decimal_precision(df, numeric_columns),
        "rounding_consistency": precision_service.rounding_consistency(df, numeric_columns),
        "significant_figures": precision_service.significant_figures(df, numeric_columns),
        "measurement_precision": precision_service.measurement_precision(df, numeric_columns),
        "calculation_accuracy": precision_service.calculation_accuracy(df, ["col1", "col2", "col3"])
    }


@app.post("/advanced-analytics")
async def advanced_analytics_quality(file: UploadFile = File(...)):
    df = read_file(file)
    numeric_columns = df.select_dtypes(include=["int", "float"]).columns.tolist()

    result = {
        "correlation_quality": analytics_service.correlation_quality(df, numeric_columns),
        "trend_consistency": analytics_service.trend_consistency(df, numeric_columns),
        "volatility_score": analytics_service.volatility_score(df, numeric_columns),
        "rate_of_change": analytics_service.rate_of_change(df, numeric_columns),
        "anomaly_score": analytics_service.anomaly_score(df, numeric_columns),
        "predictability_score": analytics_service.predictability_score(df, numeric_columns)
    }
    return clean_nan(result)