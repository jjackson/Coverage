import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import pandas as pd

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .utils import data_loader
    from .sqlqueries.sql_queries import SQL_QUERIES
    from .create_delivery_map import create_leaflet_map
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import data_loader
    from src.sqlqueries.sql_queries import SQL_QUERIES
    from src.create_delivery_map import create_leaflet_map

def create_service_only_map(output_filename: Optional[str] = None) -> str:
    """
    Create a map using only service delivery data from Superset (no delivery units from CommCare).
    
    This function uses the 'solina_100k_uservisit' SQL query to load service delivery points
    and creates an interactive map showing only the service delivery locations and FLW activities.
    
    Args:
        output_filename: Optional custom filename for the HTML map output
        
    Returns:
        Path to the generated HTML map file
        
    Raises:
        ValueError: If required environment variables are missing
        RuntimeError: If data loading or map creation fails
    """
    print("Creating service-only map using Superset data...")
    
    # Load environment variables
    load_dotenv()
    
    # Get Superset configuration from environment variables
    superset_url = os.environ.get('SUPERSET_URL')
    superset_username = os.environ.get('SUPERSET_USERNAME')
    superset_password = os.environ.get('SUPERSET_PASSWORD')
    
    # Validate environment variables
    missing_vars = []
    if not superset_url:
        missing_vars.append('SUPERSET_URL')
    if not superset_username:
        missing_vars.append('SUPERSET_USERNAME')
    if not superset_password:
        missing_vars.append('SUPERSET_PASSWORD')
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("Loading service delivery data from Superset...")

    # Export data from Superset using the services_only_sql query
    export_filename = f"service_only_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = data_loader.export_superset_query_with_pagination(
        superset_url=superset_url,
        sql_query=SQL_QUERIES["services_only_sql"],
        username=superset_username,
        password=superset_password,
        output_filename=export_filename,
        verbose=True
    )
        
    # Load the exported CSV data
    service_df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(service_df)} service delivery points from Superset")

    # Create coverage data object with only service delivery data
    print("Creating coverage data object...")

    coverage_data = data_loader.get_coverage_data_service_only(service_df)
    
    # Set some metadata for the coverage data
    coverage_data.project_space = "service_only_analysis"
    coverage_data.opportunity_name = "Solina 100k Service Analysis"
    
    print(f"Created coverage data with:")
    print(f"  - Service points: {len(coverage_data.service_points) if coverage_data.service_points else 0}")
    print(f"  - FLWs: {len(coverage_data.flws) if coverage_data.flws else 0}")
    print(f"  - Service areas: {len(coverage_data.service_areas) if coverage_data.service_areas else 0}")

    # Generate the map using the existing create_leaflet_map function
    print("Generating interactive map...")
    try:
        map_filename = create_leaflet_map(coverage_data=coverage_data)
        
        # Rename the output file if a custom filename was provided
        if output_filename and map_filename != output_filename:
            import shutil
            custom_path = f"{output_filename}.html" if not output_filename.endswith('.html') else output_filename
            shutil.move(map_filename, custom_path)
            map_filename = custom_path
        
        print(f"✅ Service-only map created successfully: {map_filename}")
        return map_filename
        
    except Exception as e:
        raise RuntimeError(f"Failed to create map: {str(e)}")

def main():
    """
    Main function to create a service-only map.
    
    This function demonstrates the usage of create_service_only_map and can be used
    to test the functionality.
    """
    
    print("Starting service-only map generation...")
    print("=" * 50)
    
    # Generate automatic filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory with timestamp
    output_dir = f"service_map_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Set output filename with directory path
    output_filename = os.path.join(output_dir, f"service_only_map_{timestamp}")
    
    # Create the service-only map with automatic naming
    map_file = create_service_only_map(output_filename=output_filename)
    
    print("=" * 50)
    print("Success: Map generation completed.")

    # Optionally open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(map_file)}")
    
if __name__ == "__main__":
    exit(main()) 