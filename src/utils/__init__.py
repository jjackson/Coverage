"""
Utils package for Coverage Analysis project.

This package provides utilities for:
- Superset data export with pagination
- Data loading from various sources (Excel, CSV, CommCare API, Superset)

Usage examples:

    # Superset data export
    from src.utils import SupersetExporter
    exporter = SupersetExporter()
    exporter.export_query_data(query_id=92)

    # Direct function imports
    from src.utils import (
        export_superset_data,
        load_coverage_data
    )

    # Module imports
    from src.utils import superset_export, data_loader
"""

# Import main classes and functions for easy access
from .data_loader import (
    get_available_files,
    select_files_interactive,
    load_excel_data,
    load_csv_data,
    convert_to_geo_dataframe,
    create_points_geo_dataframe,
    get_du_dataframe_from_commcare_api,
    load_delivery_units_from_commcare_api,
    test_commcare_api_coverage_loader,
    load_service_delivery_df_by_opportunity_from_csv,
    get_coverage_data_from_du_api_and_service_dataframe,
    get_coverage_data_from_excel,
    export_superset_query_with_pagination,
    load_service_delivery_df_by_opportunity_from_superset
)

# Define what gets imported with "from src.utils import *"
__all__ = [
    # Data loader functions
    'get_available_files',
    'select_files_interactive',
    'load_excel_data',
    'load_csv_data',
    'convert_to_geo_dataframe',
    'create_points_geo_dataframe',
    'get_du_dataframe_from_commcare_api',
    'load_delivery_units_from_commcare_api',
    'test_commcare_api_coverage_loader',
    'load_service_delivery_df_by_opportunity_from_csv',
    'get_coverage_data_from_du_api_and_service_dataframe',
    'get_coverage_data_from_excel',
    'export_superset_query_with_pagination',
    'load_service_delivery_df_by_opportunity_from_superset',
    
    # Submodules
    'superset_export',
    'data_loader'
]

# Also make modules available
from . import data_loader 