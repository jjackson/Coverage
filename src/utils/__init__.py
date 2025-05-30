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
from .superset_export import (
    get_superset_session,
    get_saved_query_details,
    execute_paginated_query,
    export_to_csv,
    main as export_superset_data
)

from .data_loader import (
    get_available_files,
    select_files_interactive,
    load_excel_data,
    load_csv_data,
    convert_to_geo_dataframe,
    create_points_geo_dataframe,
    load_coverage_data,
    get_du_dataframe_from_commcare_api,
    load_delivery_units_from_commcare_api,
    test_commcare_api_coverage_loader,
    load_service_delivery_df_by_opportunity,
    get_coverage_data_from_du_api_and_service_dataframe,
    get_coverage_data_from_excel_and_csv,
    export_superset_query_with_pagination
)

# Define what gets imported with "from src.utils import *"
__all__ = [
    # Superset export functions
    'get_superset_session',
    'get_saved_query_details',
    'execute_paginated_query',
    'export_to_csv',
    'export_superset_data',
    
    # Data loader functions
    'get_available_files',
    'select_files_interactive',
    'load_excel_data',
    'load_csv_data',
    'convert_to_geo_dataframe',
    'create_points_geo_dataframe',
    'load_coverage_data',
    'get_du_dataframe_from_commcare_api',
    'load_delivery_units_from_commcare_api',
    'test_commcare_api_coverage_loader',
    'load_service_delivery_df_by_opportunity',
    'get_coverage_data_from_du_api_and_service_dataframe',
    'get_coverage_data_from_excel_and_csv',
    'export_superset_query_with_pagination',
    
    # Submodules
    'superset_export',
    'data_loader'
]

# Create convenience class for Superset operations
class SupersetExporter:
    """
    Convenience class for Superset data export operations.
    
    Example:
        exporter = SupersetExporter()
        exporter.export_query_data(query_id=92, output_dir="exports/")
    """
    
    def __init__(self):
        """Initialize the SupersetExporter."""
        pass
    
    def export_query_data(self, query_id: str = None, output_dir: str = None, chunk_size: int = 10000):
        """
        Export data from a Superset saved query.
        
        Args:
            query_id: Superset query ID (if None, uses SUPERSET_QUERY_ID from .env)
            output_dir: Directory to save the exported file (if None, uses current directory)
            chunk_size: Number of rows to fetch per chunk
            
        Returns:
            Path to the exported CSV file or None if failed
        """
        import os
        import time
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Use provided query_id or get from environment
        if query_id is None:
            query_id = os.getenv('SUPERSET_QUERY_ID')
            if not query_id:
                print("❌ No query_id provided and SUPERSET_QUERY_ID not found in .env file")
                return None
        
        try:
            # Set up session
            session, headers, superset_url = get_superset_session()
            
            # Get query details
            query_details = get_saved_query_details(session, headers, superset_url, query_id)
            
            # Execute paginated query
            data, columns = execute_paginated_query(
                session, headers, superset_url, 
                query_details['database_id'], 
                query_details['sql'],
                chunk_size=chunk_size
            )
            
            if data:
                # Create filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"superset_export_{query_id}_{timestamp}.csv"
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, filename)
                
                # Export to CSV
                success = export_to_csv(data, columns, filename)
                
                if success:
                    return filename
            
            return None
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None
        
        finally:
            if 'session' in locals():
                session.close()

# Also make modules available
from . import superset_export, data_loader 