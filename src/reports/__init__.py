
"""Reports Package"""

from .base_report import BaseReport
from .flw_visit_count import FLWAnalysisReport
from .microplan_review import MicroplanReviewReport
from .flw_data_quality_report import FLWDataQualityReport
from .muac_sparkline_generator import MUACSparklineReport
from .visit_time_period_report import VisitTimePeriodReport
from .baseline_stats_report import BaselineStatsReport
from .flw_similarity_ranking_report import UnifiedFLWAnomalyReport
from .visit_clustering_report import VisitClusteringReport 
from .grid3_multi_res_visit_report import Grid3MultiResVisitReport
from .MLFeatureAggregationReport import MLFeatureAggregationReport
from .MLFraudDetectionReport import MLFraudDetectionReport
from .WardBoundaryExtractor import WardBoundaryExtractor
from .Grid3WardAnalysis import Grid3WardAnalysis


AVAILABLE_REPORTS = {
    "FLW Visit Analysis": FLWAnalysisReport, 
    "FLW Data Quality Assessment": FLWDataQualityReport,
    "Microplan Review": MicroplanReviewReport,
    "MUAC Sparkline Grid": MUACSparklineReport,
    "Visit Clustering": VisitClusteringReport,
    "GRID3 Multi (100m+200m+300m)": Grid3MultiResVisitReport,
    "Visit Timeline": VisitTimePeriodReport, 
    "Baseline Stats Report": BaselineStatsReport,
    "Anomaly Ranking Report": UnifiedFLWAnomalyReport,
    "ML Feature Generation": MLFeatureAggregationReport,
    "ML Fraud Detection": MLFraudDetectionReport,
    "Ward Boundary Extractor": WardBoundaryExtractor,
    "Grid3 Ward Analysis": Grid3WardAnalysis
}

__all__ = [
    "BaseReport",
    "FLWAnalysisReport",
    "FLWDataQualityReport",
    "MicroplanReviewReport",
    "MUACSparklineReport",
    "VisitClusteringReport",
    "Grid3MultiResVisitReport",
    "VisitTimePeriodReport",
    "BaselineStatsReport",
    "UnifiedFLWAnomalyReport",
    "MLFeatureAggregationReport",
    "MLFraudDetectionReport",
    "WardBoundaryExtractor",
    "Grid3WardAnalysis",
    "AVAILABLE_REPORTS",
]

