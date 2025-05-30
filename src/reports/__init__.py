
"""
Reports Package

Contains all report generators for the Coverage Analysis tool.
"""

from .base_report import BaseReport
from .flw_visit_count import FLWAnalysisReport
from .microplan_review import MicroplanReviewReport

# Dictionary of all available reports - add new reports here
AVAILABLE_REPORTS = {
    "FLW Analysis": FLWAnalysisReport,
    "Microplan Review": MicroplanReviewReport,
}

# List of all available reports for easy importing
__all__ = [
    'BaseReport',
    'FLWAnalysisReport',
    'MicroplanReviewReport',
    'AVAILABLE_REPORTS'
]
