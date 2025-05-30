import os
import glob
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import List, Dict, Any, Optional
import numpy as np
import requests
import base64
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import time
import json
from dotenv import load_dotenv
from src.models import CoverageData

def get_available_files(directory: str = '.', file_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Find available files of specific types in the specified directory.
    
    Args:
        directory: Directory to search in
        file_types: List of file extensions to find (without dot, e.g. ['xlsx', 'csv'])
        
    Returns:
        Dictionary with file type as key and list of matching files as value
    """
    if file_types is None:
        file_types = ['xlsx', 'xls', 'csv']
    
    result = {}
    
    for file_type in file_types:
        pattern = os.path.join(directory, f'*.{file_type}')
        files = glob.glob(pattern)
        
        # Filter out temporary Excel files
        if file_type in ['xlsx', 'xls']:
            files = [f for f in files if not os.path.basename(f).startswith('~$')]
        
        result[file_type] = files
    
    return result


def select_files_interactive(available_files: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Allow user to interactively select files based on type.
    
    Args:
        available_files: Dictionary of file types and available files
        
    Returns:
        Dictionary with file type as key and selected file path as value
    """
    selected_files = {}
    
    for file_type, files in available_files.items():
        if not files:
            print(f"No {file_type} files found.")
            continue
        
        if len(files) == 1:
            selected_files[file_type] = files[0]
            print(f"Automatically selected the only {file_type} file: {files[0]}")
            continue
        
        print(f"\nAvailable {file_type.upper()} files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        choice = 0
        while choice < 1 or choice > len(files):
            try:
                choice = int(input(f"\nEnter the number for the {file_type} file (1-{len(files)}): "))
            except ValueError:
                print("Please enter a valid number.")
        
        selected_files[file_type] = files[choice - 1]
    
    return selected_files


def load_excel_data(filepath: str, sheet_name: str = "Cases") -> pd.DataFrame:
    """
    Load data from an Excel file and perform basic preprocessing.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name of the sheet to load
        
    Returns:
        Preprocessed DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    # Convert columns to appropriate types (numeric) and handle missing values
    numeric_columns = ['buildings', 'delivery_count', 'delivery_target', 'surface_area']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if col != 'surface_area':  # Keep surface_area as float
                df[col] = df[col].astype(int)
    
    return df


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the CSV data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    return df


def convert_to_geo_dataframe(df: pd.DataFrame, wkt_column: str = 'WKT', crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with WKT geometry to a GeoDataFrame.
    
    Args:
        df: Input DataFrame
        wkt_column: Name of the column containing WKT geometry strings
        crs: Coordinate reference system
        
    Returns:
        GeoDataFrame with geometry
    """
    if wkt_column not in df.columns:
        raise ValueError(f"WKT column '{wkt_column}' not found in DataFrame")
    
    # Convert WKT to geometry
    df['geometry'] = df[wkt_column].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    return gdf


def create_points_geo_dataframe(df: pd.DataFrame, 
                               lon_column: str = 'longitude', 
                               lat_column: str = 'lattitude',  # Note: using 'lattitude' as in original code
                               crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame with point coordinates.
    
    Args:
        df: Input DataFrame
        lon_column: Name of the longitude column
        lat_column: Name of the latitude column
        crs: Coordinate reference system
        
    Returns:
        GeoDataFrame with point geometry
    """
    for col in [lon_column, lat_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Create geometry from coordinates
    geometry = gpd.points_from_xy(df[lon_column], df[lat_column])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    
    return gdf


def load_coverage_data(excel_file: str, service_delivery_csv: Optional[str] = None) -> CoverageData:
    """
    Load coverage data from Excel and optionally CSV files.
    
    This is a wrapper around CoverageData.from_excel_and_csv that provides additional error handling
    and validation.
    
    Args:
        excel_file: Path to Excel file with delivery units data
        service_delivery_csv: Optional path to CSV file with service delivery points
        
    Returns:
        Loaded CoverageData object
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    if service_delivery_csv and not os.path.exists(service_delivery_csv):
        raise FileNotFoundError(f"Service delivery CSV file not found: {service_delivery_csv}")
    
    try:
        # Use the CoverageData.from_excel_and_csv method to load the data
        coverage_data = CoverageData.from_excel_and_csv(excel_file, service_delivery_csv)
        
        # Perform basic validation
        if not coverage_data.delivery_units:
            print("Warning: No delivery units were loaded from the Excel file.")
        
        if service_delivery_csv and not coverage_data.service_points:
            print("Warning: No service delivery points were loaded from the CSV file.")
        
        return coverage_data
    
    except Exception as e:
        raise RuntimeError(f"Error loading coverage data: {str(e)}") 


def get_du_dataframe_from_commcare_api(domain: str, 
                       user: str,
                       api_key: str, 
                       base_url: str = "https://www.commcarehq.org") -> pd.DataFrame:
    """
    Load delivery unit data from CommCare's Case API v2.
    
    Args:
        domain: CommCare project space/domain
        user: Username for authentication
        api_key: API key for authentication
        base_url: Base URL for the CommCare instance (default: 'https://www.commcarehq.org')
        
    Returns:
        DataFrame with case data formatted similar to Excel import
    """
    case_type = "deliver-unit"

    # Build API endpoint
    endpoint = f"{base_url}/a/{domain}/api/case/v2/"
    
    # Set up authentication - use base64 encoding
    auth_string = f"{user}:{api_key}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        'Authorization': f'ApiKey {user}:{api_key}', 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # Parameters for the API request
    params = {
        'case_type': case_type,
        'limit': 4000  # Fetch 100 cases at a time
    }
    
    # DEBUG: Print request information
    print("\n=== DEBUG: REQUEST INFO ===")
    print(f"URL: {endpoint}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    print("=========================\n")
    
    all_cases = []
    next_url = endpoint
    
    # Paginate through all results, something is currently not working in this loop
    # TODO: Fix this
    page_count = 0
    while next_url:
        page_count += 1
        print(f"\n--- Request #{page_count}: {next_url.split('?')[0]} ---")
        
        # # For debugging, separate the initial request and paginated requests
        # if next_url == endpoint:
        #     response = requests.get(
        #         next_url,
        #         params=params,
        #         headers=headers
        #     )
        # else:
        #     response = requests.get(
        #         next_url,
        #         params=params, # AI tried to remove this, but I added it back in... not sure if needed or not.
        #         headers=headers
        #     )
        
        num_retry = 5
        response = None
        
        for attempt in range(num_retry):
            response = requests.get(
                    next_url,
                    params=params,
                    headers=headers
            )
            
            # Debug response status
            print(f"Status: {response.status_code} (attempt {attempt + 1}/{num_retry})")
            
            if response.status_code == 200:
                break  # Success, exit the retry loop
            else:
                if attempt == num_retry - 1:  # Last attempt failed
                    raise ValueError(f"API request failed with status {response.status_code} after {num_retry} attempts")
                else:
                    time.sleep(2)
                    print(f"Retrying API call")
        
        try:
            data = response.json()
            # Debug JSON
            # print(f"Data: {data}")
            # Just show case count and next URL info
            case_count = len(data.get('cases', []))
            print(f"Retrieved {case_count} cases")
            
            # Show first case type if available
            if case_count > 0:
                first_case = data.get('cases', [])[0]
                print(f"Case type: {first_case.get('case_type')}")
            
            all_cases.extend(data.get('cases', []))
            
            # Get the next page URL, if any
            next_url = data.get('next')
            if next_url:
                print(f"Next cursor: {next_url.split('?')[1] if '?' in next_url else 'NONE'}")
            else:
                print("No more pages")
                
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print("Response preview (first 100 chars):")
            print(response.text[:100])
            raise
    
    # If no cases were found
    if not all_cases:
        return pd.DataFrame()
    
    # Process case data into a format similar to Excel import
    processed_data = []
    for case in all_cases:
        # Combine all case properties with top-level case fields
        case_data = {
            **case.get('properties', {}),
            'case_id': case.get('case_id'),
            'case_name': case.get('case_name'),
            'external_id': case.get('external_id'),
            'owner_id': case.get('owner_id'),
            'date_opened': case.get('date_opened'),
            'last_modified': case.get('last_modified'),
            'closed': case.get('closed', False)
        }
        
        processed_data.append(case_data)
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    # Convert columns to appropriate types (numeric) and handle missing values
    numeric_columns = ['buildings', 'delivery_count', 'delivery_target', 'surface_area']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if col != 'surface_area':  # Keep surface_area as float
                df[col] = df[col].astype(int)
    
    # Convert WKT to geometry if 'WKT' column exists
    if 'WKT' in df.columns:
        try:
            df = convert_to_geo_dataframe(df)
        except Exception as e:
            print(f"Warning: Could not convert WKT to geometry: {e}")
    
    return df

def load_delivery_units_from_commcare_api(domain: str, 
                            user: str,
                            api_key: str, 
                            case_type: str = "deliver-unit",
                            base_url: str = "https://www.commcarehq.org") -> CoverageData:
    """
    Load delivery units directly from CommCare API into a CoverageData object.
    
    This function combines get_commcare_du_dataframe_from_api with CoverageData.load_delivery_units_from_df
    to avoid code duplication.
    
    Args:
        domain: CommCare project space/domain
        user: Username for authentication
        api_key: API key for authentication
        case_type: Case type to fetch (default: 'deliver-unit')
        base_url: Base URL for the CommCare instance
        
    Returns:
        Loaded CoverageData object
    """
    # First, get the DataFrame from CommCare
    df = get_du_dataframe_from_commcare_api(domain, user, api_key, case_type, base_url)
    
    if df.empty:
        print("Warning: No delivery units were loaded from CommCare.")
        return CoverageData()
    
    # Then use the CoverageData.load_delivery_units_from_df method to process the dataframe
    try:
        coverage_data = CoverageData.load_delivery_units_from_df(df)
        
        # Perform basic validation
        if not coverage_data.delivery_units:
            print("Warning: No delivery units were processed from CommCare data.")
        
        return coverage_data
    
    except Exception as e:
        raise RuntimeError(f"Error loading coverage data from CommCare: {str(e)}")




# Add this function at the end of the file, before the if __name__ == "__main__" block
def test_commcare_api_coverage_loader(domain: str, user: str, api_key: str):
    """
    Test function to demonstrate loading data from CommCare API into a CoverageData object.
    
    Args:
        domain: CommCare project space/domain name
        user: Username for authentication
        api_key: API key for authentication
    """
    print("Testing CommCare API coverage data loader...")
    
    try:
        # Load delivery unit cases into a CoverageData object
        print(f"Fetching delivery-unit cases from {domain}...")
        coverage_data = load_delivery_units_from_commcare_api(domain=domain, user=user, api_key=api_key)
        
        # Display basic info about the loaded data
        if coverage_data.delivery_units:
            print(f"Successfully loaded {len(coverage_data.delivery_units)} delivery units")
            print(f"Total service areas: {coverage_data.total_service_areas}")
            print(f"Total buildings: {coverage_data.total_buildings}")
            print(f"Total FLWs: {coverage_data.total_flws}")
            print(f"Completion percentage: {coverage_data.completion_percentage:.2f}%")
        else:
            print("No delivery units were loaded")
        
        return coverage_data
    
    except Exception as e:
        print(f"Error testing CommCare API coverage loader: {str(e)}")
        return None

@DeprecationWarning # TODO: Had several issues and stopped work on this function
def export_to_excel_using_commcare_export(
    domain: str,
    username: str,
    api_key: str,
    query_file_path: str,
    output_file_path: str = None,
    commcare_hq_url: str = "https://www.commcarehq.org",
    verbose: bool = False
) -> str:
    """
    Export data from CommCare HQ to Excel using the commcare-export command line tool.
    
    Args:
        domain: CommCare project space/domain name
        username: CommCare HQ username
        api_key: CommCare HQ API key
        query_file_path: Path to the query file (Excel or JSON)
        output_file_path: Optional path where the output Excel file should be saved.
                         If not provided, it will generate a file in the data directory 
                         with a name based on the domain and current timestamp.
        commcare_hq_url: Base URL for CommCare HQ (default: https://www.commcarehq.org)
        verbose: Whether to output verbose logs
        
    Returns:
        Path to the created Excel file
        
    Raises:
        FileNotFoundError: If the query file doesn't exist
        RuntimeError: If the commcare-export command fails
    """
    # Verify query file exists
    if not os.path.exists(query_file_path):
        raise FileNotFoundError(f"Query file not found: {query_file_path}")
    
    # Generate default output path if not provided
    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.getcwd(), "data")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        output_file_path = os.path.join(data_dir, f"{domain}-export-{timestamp}.xlsx")
    
    # Make sure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build command arguments
    cmd = [
        "commcare-export",
        "--output-format", "xlsx",
        "--output", output_file_path,
        "--query", query_file_path,
        "--project", domain,
        "--username", username,
        "--auth-mode", "apikey",
        "--password", api_key,
        "--commcare-hq", commcare_hq_url
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    # Execute command
    # Format command string with proper quoting for display
    def quote_if_needed(arg):
        if ' ' in arg or '\t' in arg or '"' in arg or "'" in arg:
            # Use double quotes and escape any existing double quotes
            return f'"{arg.replace('"', '\\"')}"'
        return arg
        
    cmd_str = ' '.join(quote_if_needed(arg) for arg in cmd)
    print(f"Executing command: {cmd_str}")
    
    # The actual subprocess.run call uses a list which handles spaces correctly
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    
    if verbose:
        print("Command output:")
        print(result.stdout)
    
    # Verify the file was created
    if not os.path.exists(output_file_path):
        raise RuntimeError(
            f"Command executed successfully but output file not found: {output_file_path}\n"
            f"Command output: {result.stdout}\n"
            f"Command error: {result.stderr}"
        )
    
    print(f"Successfully exported data to {output_file_path}")
    return output_file_path

def load_service_delivery_df_by_opportunity(csv_file: str) -> Dict[str, pd.DataFrame]:
    """
    Load service delivery data from CSV and group by unique opportunity_name values.
    
    Args:
        csv_file: Path to the CSV file containing service delivery data
        
    Returns:
        Dictionary with opportunity_name as key and DataFrame of service points as value
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file doesn't contain an 'opportunity_name' column
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    try:
        # Load the CSV data
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}. Please check the file path and try again.")

    # Check if opportunity_name column exists
    if 'opportunity_name' not in df.columns:
        raise ValueError("CSV file must contain an 'opportunity_name' column")
    
    # Group by opportunity_name and create dictionary of DataFrames
    opportunity_groups = {}
    
    # Get unique opportunity names (excluding NaN values)
    unique_opportunities = df['opportunity_name'].dropna().unique()
    
    for opportunity in unique_opportunities:
        # Filter data for this opportunity
        opportunity_df = df[df['opportunity_name'] == opportunity].copy()
        
        # Reset index for clean DataFrame
        opportunity_df.reset_index(drop=True, inplace=True)
        
        opportunity_groups[opportunity] = opportunity_df
    
    # Check for rows with missing opportunity_name and throw error if any exist
    missing_opportunity_count = df['opportunity_name'].isna().sum()
    if missing_opportunity_count > 0:
        raise ValueError(f"Found {missing_opportunity_count} rows with missing 'opportunity_name' values. All rows must have a valid opportunity_name.")
    
    return opportunity_groups

def get_coverage_data_from_du_api_and_service_dataframe(domain: str, user: str, api_key: str, service_df: pd.DataFrame) -> 'CoverageData':
    """
    Load coverage data from API and service Delivery from DataFrame
    
    Args:
        domain: CommCare project space/domain name
        user: Username for authentication
        api_key: API key for authentication
        service_df: DataFrame containing service delivery GPS coordinates
    """
    
    data = CoverageData()
    
    # retrieve from API and Load
    delivery_units_df = get_du_dataframe_from_commcare_api(domain, user, api_key)
    data = CoverageData.load_delivery_units_from_df(delivery_units_df)
    
    # Load service delivery data
    data.load_service_delivery_from_datafame(service_df)
    
    return data

def get_coverage_data_from_excel_and_csv(excel_file: str, service_delivery_csv: Optional[str] = None) -> 'CoverageData':
    """
    Load coverage data from Excel and CSV files
    
    Args:
        excel_file: Path to the DU Export Excel file
        service_delivery_csv: Optional path to service delivery GPS coordinates CSV
    """
    data = CoverageData()
    
    # Read the Excel file
    print(f"Loading Excel file: {excel_file}")
    delivery_units_df = pd.read_excel(excel_file, sheet_name="Cases")
    
    # Use the new from_commcare method to process the dataframe
    data = CoverageData.load_delivery_units_from_df(delivery_units_df)
    
    # Load service delivery data if provided
    if service_delivery_csv:
        service_df = data.load_service_delivery_dataframe_from_csv(service_delivery_csv)
        data.load_service_delivery_from_datafame(service_df)
    
    return data

# TODO: This function is not working, it is not retrieving the data from the query.
# def retrieve_service_delivery_csv_from_superset(
#     superset_url: str,
#     username: str,
#     password: str,
#     query_id: str,
#     output_filename: Optional[str] = None,
#     timeout: int = 300
# ) -> str:
#     """
#     Retrieve service delivery point data from Apache Superset as CSV and save to data directory.
    
#     Args:
#         superset_url: Base URL of the Superset instance (e.g., 'https://superset.example.com')
#         username: Superset username for authentication
#         password: Superset password for authentication
#         query_id: Query ID or Chart ID to export data from
#         output_filename: Optional custom filename for the CSV (without extension)
#         timeout: Request timeout in seconds (default: 300)
        
#     Returns:
#         Path to the saved CSV file
        
#     Raises:
#         ValueError: If required parameters are missing or invalid
#         RuntimeError: If authentication or data retrieval fails
#         FileNotFoundError: If the Superset instance is not accessible
#     """
#     # Validate input parameters
#     if not superset_url or not username or not password or not query_id:
#         raise ValueError("superset_url, username, password, and query_id are required")
    
#     # Clean up the base URL
#     superset_url = superset_url.rstrip('/')
    
#     # Create data directory if it doesn't exist
#     project_root = Path(__file__).parent.parent.parent  # Go up from src/utils/data_loader.py to project root
#     data_dir = project_root / "data"
#     if not data_dir.exists():
#         data_dir.mkdir(parents=True, exist_ok=True)
#         print(f"Created data directory: {data_dir}")
    
#     # Generate output filename if not provided
#     if output_filename is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_filename = f"superset_service_delivery_{query_id}_{timestamp}"
    
#     output_path = data_dir / f"{output_filename}.csv"
    
#     try:
#         print(f"Authenticating with Superset at {superset_url}...")
        
#         # 1. Authenticate and get access token
#         auth_url = f'{superset_url}/api/v1/security/login'
#         auth_payload = {
#             'username': username,
#             'password': password,
#             'provider': 'db',
#             'refresh': True
#         }

#         session = requests.Session()
#         auth_response = session.post(auth_url, json=auth_payload, timeout=timeout)
#         auth_response.raise_for_status()
        
#         # Check if authentication was successful
#         auth_data = auth_response.json()
#         if 'access_token' not in auth_data:
#             raise RuntimeError("Authentication failed - no access token received")
        
#         access_token = auth_data['access_token']
#         headers = {'Authorization': f'Bearer {access_token}'}
        
#         print("Successfully authenticated with Superset")

#         # 2. Get CSRF token which is required for SQL Lab operations
#         print("Getting CSRF token...")
#         csrf_url = f'{superset_url}/api/v1/security/csrf_token/'
#         csrf_response = session.get(csrf_url, headers=headers, timeout=timeout)
        
#         if csrf_response.status_code == 200:
#             csrf_data = csrf_response.json()
#             csrf_token = csrf_data.get('result')
#             if csrf_token:
#                 headers['X-CSRFToken'] = csrf_token
#                 # Also add Referer header which is required by Superset
#                 headers['Referer'] = superset_url
#                 print("Successfully obtained CSRF token")
#             else:
#                 print("Warning: No CSRF token in response, continuing without it")
#         else:
#             print(f"Warning: Could not get CSRF token (status {csrf_response.status_code}), continuing without it")

#         # 3. Try multiple approaches to get the data
#         csv_response = None
#         successful_endpoint = None
        
#         try:
#             # Approach 1: Try to get saved query information first
#             print(f"Retrieving saved query information for ID: {query_id}")
#             saved_query_url = f'{superset_url}/api/v1/saved_query/{query_id}'
            
#             saved_query_response = session.get(saved_query_url, headers=headers, timeout=timeout)
            
#             if saved_query_response.status_code == 200:
#                 saved_query_data = saved_query_response.json()
#                 print(f"Successfully retrieved saved query: {saved_query_data.get('result', {}).get('label', 'Unknown')}")
                
#                 # Extract SQL from saved query
#                 sql_query = saved_query_data.get('result', {}).get('sql', '')
#                 database_id = saved_query_data.get('result', {}).get('database', {}).get('id', '')
                
#                 if sql_query and database_id:
#                     # Approach 1a: Execute the SQL directly via SQL Lab API
#                     print("Executing SQL query directly...")
#                     execute_url = f'{superset_url}/api/v1/sqllab/execute/'
                    
#                     execute_payload = {
#                         'database_id': database_id,
#                         'sql': sql_query,
#                         'runAsync': False,
#                         # 'schema': saved_query_data.get('result', {}).get('schema', ''),
#                         'sql_editor_id': f'saved_query_{query_id}',
#                         # 'tab_name': f'Query {query_id}',
#                         # 'tmp_table_name': '',
#                         #'select_as_cta': False,
#                         #'ctas_method': 'TABLE',
#                         'queryLimit': 1000000,  # Large limit to get all data
#                         #'expand_data': True
#                     }
                    
#                     execute_response = session.post(
#                         execute_url, 
#                         json=execute_payload,
#                         headers=headers,
#                         timeout=timeout
#                     )
                    
#                     if execute_response.status_code == 200:
#                         execution_result = execute_response.json()
#                         print(f"Query execution status: {execution_result.get('status', 'unknown')}")
                        
#                         # Check if we have data in the response
#                         if 'data' in execution_result:
#                             data_rows = execution_result['data']
#                             print(f"Retrieved {len(data_rows)} rows via synchronous execution")
                            
#                             if len(data_rows) < 50000:  # If we got fewer rows, try async for more
#                                 print("Trying async execution for larger dataset...")
                                
#                                 # Try async execution for larger datasets
#                                 async_payload = {
#                                     'database_id': database_id,
#                                     'sql': sql_query,
#                                     'runAsync': True,
#                                     'queryLimit': 1000000,
#                                 }
                                
#                                 async_response = session.post(
#                                     execute_url, 
#                                     json=async_payload,
#                                     headers=headers,
#                                     timeout=timeout
#                                 )
                                
#                                 if async_response.status_code == 202:  # Async queries return 202
#                                     async_result = async_response.json()
#                                     query_id_result = async_result.get('query', {}).get('id')
#                                     print(f"Async query started with ID: {query_id_result}")
                                    
#                                     if query_id_result:
#                                         # Poll for results
#                                         print("Polling for async results...")
#                                         import time
                                        
#                                         for attempt in range(30):  # Try for up to 60 seconds
#                                             time.sleep(2)
#                                             status_url = f'{superset_url}/api/v1/query/{query_id_result}'
#                                             status_response = session.get(status_url, headers=headers, timeout=timeout)
                                            
#                                             if status_response.status_code == 200:
#                                                 status_data = status_response.json()
#                                                 result_data = status_data.get('result', {})
#                                                 query_status = result_data.get('status', 'unknown')
                                                
#                                                 print(f"Attempt {attempt + 1}: Status = {query_status}")
                                                
#                                                 if query_status == 'success':
#                                                     # Get the results
#                                                     if 'data' in result_data:
#                                                         async_data = result_data['data']
#                                                         print(f"Async query completed! Retrieved {len(async_data)} rows")
#                                                         # Use async data if it has more rows
#                                                         if len(async_data) > len(data_rows):
#                                                             data_rows = async_data
#                                                             execution_result = result_data
#                                                         break
#                                                 elif query_status in ['failed', 'error']:
#                                                     print(f"Async query failed: {result_data.get('errorMessage', 'Unknown error')}")
#                                                     break
#                                             else:
#                                                 print(f"Could not check async status: {status_response.status_code}")
#                                         else:
#                                             print("Async query timeout - using synchronous results")
                                
#                                 # Convert the result data to DataFrame
#                                 columns = [col['name'] for col in execution_result.get('columns', [])]
                                
#                                 if columns and data_rows:
#                                     print(f"Converting {len(data_rows)} rows to CSV...")
#                                     import pandas as pd
#                                     df = pd.DataFrame(data_rows, columns=columns)
#                                     csv_data = df.to_csv(index=False)
                                    
#                                     # Create a mock response object
#                                     class MockResponse:
#                                         def __init__(self, content):
#                                             self.content = content.encode('utf-8')
#                                             self.status_code = 200
#                                         def raise_for_status(self):
#                                             pass
                                    
#                                     csv_response = MockResponse(csv_data)
#                                     successful_endpoint = "direct_sql_execution"
#                                 else:
#                                     print("Warning: Query executed but returned no data")
#                         else:
#                             print("Warning: Query executed but no data field in response")
#                     else:
#                         print(f"SQL execution failed with status: {execute_response.status_code}")
#                         print(f"Error response: {execute_response.text}")
                
#         except Exception as e:
#             print(f"Approach 1 failed: {str(e)}")
        
#         # If all approaches failed, raise an error
#         if not csv_response:
#             raise RuntimeError(f"Failed to retrieve data using any available method for ID {query_id}")
        
#         print(f"Successfully retrieved data using method: {successful_endpoint}")
#         csv_response.raise_for_status()

#         # 3. Save CSV to file
#         with open(output_path, 'wb') as f:
#             f.write(csv_response.content)

#         print(f"Successfully saved service delivery data to: {output_path}")
        
#         # Verify file was created and has content
#         if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#             # Load and display basic info about the data
#             try:
#                 df_verify = pd.read_csv(output_path)
#                 print(f"CSV file contains {len(df_verify)} rows and {len(df_verify.columns)} columns")
#                 if len(df_verify.columns) > 0:
#                     print(f"Columns: {', '.join(df_verify.columns.tolist())}")
#             except Exception as e:
#                 print(f"Warning: Could not verify CSV content: {e}")
            
#             return output_path
#         else:
#             raise RuntimeError(f"Failed to create or write to output file: {output_path}")
    
#     except requests.exceptions.RequestException as e:
#         raise RuntimeError(f"Network error while connecting to Superset: {str(e)}")
#     except Exception as e:
#         raise RuntimeError(f"Error retrieving data from Superset: {str(e)}")
#     finally:
#         session.close()

def export_superset_query_with_pagination(
    superset_url: str,
    username: str,
    password: str,
    query_id: str,
    output_filename: Optional[str] = None,
    chunk_size: int = 10000,
    timeout: int = 120
) -> str:
    """
    Export all data from a Superset saved query using pagination to bypass the 10,000 row limit.
    
    This method uses the same approach as the superset_export.py script to fetch data in chunks
    and combine them into a single CSV file.
    
    Args:
        superset_url: Base URL of the Superset instance (e.g., 'https://superset.example.com')
        username: Superset username for authentication
        password: Superset password for authentication
        query_id: Saved query ID to export data from
        output_filename: Optional custom filename for the CSV (without extension)
        chunk_size: Number of rows to fetch per chunk (default: 10000)
        timeout: Request timeout in seconds (default: 120)
        
    Returns:
        Path to the saved CSV file
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If authentication or data retrieval fails
    """
    # Validate input parameters
    if not superset_url or not username or not password or not query_id:
        raise ValueError("superset_url, username, password, and query_id are required")
    
    # Clean up the base URL
    superset_url = superset_url.rstrip('/')
    if not superset_url.startswith(('http://', 'https://')):
        superset_url = f'https://{superset_url}'
    
    # Create data directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent  # Go up from src/utils/data_loader.py to project root
    data_dir = project_root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory: {data_dir}")
    
    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"superset_export_{query_id}_{timestamp}"
    
    output_path = data_dir / f"{output_filename}.csv"
    
    try:
        print("=== Superset Paginated Data Export ===")
        print(f"Query ID: {query_id}")
        print()
        
        # Set up session and authenticate
        print("🔐 Authenticating with Superset...")
        session = requests.Session()
        
        # Authenticate
        auth_url = f'{superset_url}/api/v1/security/login'
        auth_payload = {
            'username': username,
            'password': password,
            'provider': 'db',
            'refresh': True
        }
        
        auth_response = session.post(auth_url, json=auth_payload, timeout=timeout)
        if auth_response.status_code != 200:
            raise RuntimeError(f"Authentication failed: {auth_response.text}")
        
        auth_data = auth_response.json()
        if 'access_token' not in auth_data:
            raise RuntimeError(f"No access token in response: {auth_data}")
        
        access_token = auth_data['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get CSRF token
        csrf_url = f'{superset_url}/api/v1/security/csrf_token/'
        csrf_response = session.get(csrf_url, headers=headers, timeout=timeout)
        if csrf_response.status_code == 200:
            csrf_data = csrf_response.json()
            csrf_token = csrf_data.get('result')
            if csrf_token:
                headers['X-CSRFToken'] = csrf_token
                headers['Referer'] = superset_url
        
        # Get saved query details
        print("📋 Getting saved query details...")
        saved_query_url = f'{superset_url}/api/v1/saved_query/{query_id}'
        response = session.get(saved_query_url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get saved query: {response.text}")
        
        query_data = response.json()
        result = query_data.get('result', {})
        
        query_details = {
            'sql': result.get('sql', ''),
            'database_id': result.get('database', {}).get('id', ''),
            'label': result.get('label', 'Unknown'),
            'schema': result.get('schema', '')
        }
        
        print(f"   Query: {query_details['label']}")
        print(f"   Database ID: {query_details['database_id']}")
        print(f"   Schema: {query_details['schema']}")
        print()
        
        # Execute paginated query
        execute_url = f'{superset_url}/api/v1/sqllab/execute/'
        all_data = []
        all_columns = None
        offset = 0
        total_rows = 0
        chunk_num = 1
        base_sql = query_details['sql']
        
        print(f"🔄 Starting paginated data fetch (chunk size: {chunk_size:,})")
        print(f"📝 Base SQL preview: {base_sql[:100]}...")
        print()
        
        while True:
            # Add OFFSET and LIMIT to the SQL
            paginated_sql = f"""
{base_sql.rstrip(';')}
OFFSET {offset}
LIMIT {chunk_size}
"""
            
            payload = {
                'database_id': int(query_details['database_id']),
                'sql': paginated_sql,
                'runAsync': False,
                'queryLimit': chunk_size + 1000,  # Add buffer
            }
            
            print(f"📦 Fetching chunk {chunk_num} (rows {offset + 1:,} to {offset + chunk_size:,})...")
            
            try:
                response = session.post(execute_url, json=payload, headers=headers, timeout=timeout)
                
                if response.status_code != 200:
                    print(f"❌ Chunk {chunk_num} failed: {response.text}")
                    break
                
                result = response.json()
                
                if result.get('status') != 'success':
                    print(f"❌ Chunk {chunk_num} query failed: {result}")
                    break
                
                # Get data and columns
                chunk_data = result.get('data', [])
                columns = result.get('columns', [])
                
                if not chunk_data:
                    print(f"✅ No more data found. Stopping at chunk {chunk_num - 1}")
                    break
                
                # Store columns from first chunk
                if all_columns is None:
                    all_columns = columns
                    print(f"📊 Columns found: {len(columns)} columns")
                    column_names = [col.get('name', '') for col in columns]
                    print(f"   Column names: {', '.join(column_names[:5])}{'...' if len(column_names) > 5 else ''}")
                
                # Add chunk data to overall results
                all_data.extend(chunk_data)
                chunk_rows = len(chunk_data)
                total_rows += chunk_rows
                
                print(f"✅ Chunk {chunk_num}: {chunk_rows:,} rows fetched (Total: {total_rows:,})")
                
                # If we got fewer rows than chunk_size, we've reached the end
                if chunk_rows < chunk_size:
                    print(f"🎉 Reached end of data (chunk had {chunk_rows:,} < {chunk_size:,} rows)")
                    break
                
                # Prepare for next chunk
                offset += chunk_size
                chunk_num += 1
                
                # Small delay to be nice to the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error fetching chunk {chunk_num}: {e}")
                break
        
        print(f"\n🎯 Pagination complete!")
        print(f"   Total rows fetched: {total_rows:,}")
        print(f"   Total chunks: {chunk_num - 1}")
        
        # Export data to CSV
        if not all_data or not all_columns:
            raise RuntimeError("No data was retrieved from the query")
        
        # Create DataFrame
        column_names = [col.get('name', f'col_{i}') for i, col in enumerate(all_columns)]
        df = pd.DataFrame(all_data, columns=column_names)
        
        # Export to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"💾 Data exported to: {output_path}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Show preview
        print(f"\n📋 Data preview:")
        print(df.head().to_string())
        
        print(f"\n🎉 Export completed successfully!")
        print(f"   Final row count: {len(all_data):,}")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Error exporting data from Superset: {str(e)}")
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    # Only run the test if this file is executed directly
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Superset configuration from environment variables
    # Replace these variable names with your actual .env variable names
    superset_url = os.getenv('SUPERSET_URL')  # e.g., 'https://superset.example.com'
    superset_username = os.getenv('SUPERSET_USERNAME')  # Your Superset username
    superset_password = os.getenv('SUPERSET_PASSWORD')  # Your Superset password
    superset_query_id = os.getenv('SUPERSET_QUERY_ID')  # The query/chart ID to download
    
    # Add https:// scheme if missing
    if superset_url and not superset_url.startswith(('http://', 'https://')):
        superset_url = f'https://{superset_url}'
        print(f"Added https:// scheme to URL: {superset_url}")
    
    # Validate that all required environment variables are set
    missing_vars = []
    if not superset_url:
        missing_vars.append('SUPERSET_URL')
    if not superset_username:
        missing_vars.append('SUPERSET_USERNAME')
    if not superset_password:
        missing_vars.append('SUPERSET_PASSWORD')
    if not superset_query_id:
        missing_vars.append('SUPERSET_QUERY_ID')
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set the following in your .env file:")
        for var in missing_vars:
            print(f"- {var}")
        exit(1)
    
    try:
        print("Testing Superset CSV retrieval...")
        csv_path = export_superset_query_with_pagination(   
            superset_url=superset_url,
            username=superset_username,
            password=superset_password,
            query_id=superset_query_id
        )
        print(f"Success! CSV saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error retrieving data from Superset: {str(e)}")
        exit(1)
    