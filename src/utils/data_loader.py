import os
import glob
import pandas as pd
import geopandas as gpd
import constants as constants
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
from functools import lru_cache
from src.sqlqueries import sql_queries

def get_available_files(directory: str = 'data', file_type: str = None) -> List[str]:
    """
    Find available files of a specific type in the specified directory.
    
    Args:
        directory: Directory to search in (defaults to 'data')
        file_type: File extension to find (without dot, e.g. 'xlsx', 'csv')
        
    Returns:
        List of matching files
    """
    if file_type is None:
        raise ValueError("file_type must be specified (e.g., 'xlsx', 'csv')")
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Search for files with the specified extension
    pattern = os.path.join(directory, f'*.{file_type}')
    files = glob.glob(pattern)
    
    # Filter out temporary Excel files
    if file_type in ['xlsx', 'xls']:
        files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    return files


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


# def load_coverage_data(excel_file: str, service_delivery_csv: Optional[str] = None) -> CoverageData:
#     """
#     Load coverage data from Excel and optionally CSV files.
    
#     This is a wrapper around CoverageData.from_excel_and_csv that provides additional error handling
#     and validation.
    
#     Args:
#         excel_file: Path to Excel file with delivery units data
#         service_delivery_csv: Optional path to CSV file with service delivery points
        
#     Returns:
#         Loaded CoverageData object
#     """
#     if not os.path.exists(excel_file):
#         raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
#     if service_delivery_csv and not os.path.exists(service_delivery_csv):
#         raise FileNotFoundError(f"Service delivery CSV file not found: {service_delivery_csv}")
    
#     try:
#         # Use the CoverageData.from_excel_and_csv method to load the data
#         coverage_data = CoverageData.from_excel_and_csv(excel_file, service_delivery_csv)
        
#         # Perform basic validation
#         if not coverage_data.delivery_units:
#             print("Warning: No delivery units were loaded from the Excel file.")
        
#         if service_delivery_csv and not coverage_data.service_points:
#             print("Warning: No service delivery points were loaded from the CSV file.")
        
#         return coverage_data
    
#     except Exception as e:
#         raise RuntimeError(f"Error loading coverage data: {str(e)}") 


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
    
    # Save DataFrame to Excel file in data directory
    excel_path = ensure_data_directory_and_get_filename(
        file_prefix="du_export",
        file_id=domain,
        file_suffix="xlsx"
    )
    
    df.to_excel(excel_path, index=False)
    print(f"Delivery units data saved to: {excel_path}")
    
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

def load_service_delivery_df_by_opportunity_from_csv(csv_file: str) -> Dict[str, pd.DataFrame]:
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
    
    return group_service_delivery_df_by_opportunity(df)
    # # Group by opportunity_name and create dictionary of DataFrames
    # opportunity_groups = {}
    
    # # Get unique opportunity names (excluding NaN values)
    # unique_opportunities = df['opportunity_name'].dropna().unique()
    
    # for opportunity in unique_opportunities:
    #     # Filter data for this opportunity
    #     opportunity_df = df[df['opportunity_name'] == opportunity].copy()
        
    #     # Reset index for clean DataFrame
    #     opportunity_df.reset_index(drop=True, inplace=True)
        
    #     opportunity_groups[opportunity] = opportunity_df
    
    # # Check for rows with missing opportunity_name and throw error if any exist
    # missing_opportunity_count = df['opportunity_name'].isna().sum()
    # if missing_opportunity_count > 0:
    #     raise ValueError(f"Found {missing_opportunity_count} rows with missing 'opportunity_name' values. All rows must have a valid opportunity_name.")
    
    # return opportunity_groups

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

@lru_cache(maxsize=None) #stop reloading data during the same session
def get_coverage_data_from_excel_and_csv(excel_file: str, service_delivery_csv: Optional[str] = None) -> 'CoverageData':

    """
    Load coverage data from Excel
    
    Args:
        excel_file: Path to the DU Export Excel file
    """
    data = CoverageData()
    
    # Read the Excel file
    print(f"Loading Excel file: {excel_file}")
    delivery_units_df = pd.read_excel(excel_file, sheet_name=0)
    
    # Use the new from_commcare method to process the dataframe
    data = CoverageData.load_delivery_units_from_df(delivery_units_df)

    return data

def ensure_data_directory_and_get_filename(output_filename: Optional[str] = None, 
                                     file_prefix: str = "export", 
                                     file_id: str = None,
                                     file_suffix: str = None) -> Path:
    """
    Ensure the data directory exists and generate an output filename if not provided.
    
    Args:
        output_filename: Optional custom filename for the output file (without extension)
        file_prefix: Prefix for auto-generated filename (default: "export")
        file_id: Optional ID to include in auto-generated filename
        file_suffix: Optional file extension (e.g., "csv", "xlsx"). If provided, will be added to the returned path.
        
    Returns:
        Path object for the output file (with or without extension based on file_suffix)
        
    Example:
        # Auto-generate filename with extension
        path = ensure_data_directory_and_get_filename(file_prefix="superset_export", file_id="123", file_suffix="csv")
        # Returns: /path/to/project/data/superset_export_123_20231215_143022.csv
        
        # Use custom filename with extension
        path = ensure_data_directory_and_get_filename("my_custom_file", file_suffix="xlsx")
        # Returns: /path/to/project/data/my_custom_file.xlsx
        
        # Without extension (original behavior)
        path = ensure_data_directory_and_get_filename("my_custom_file")
        # Returns: /path/to/project/data/my_custom_file
    """
    # Create data directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent  # Go up from src/utils/data_loader.py to project root
    data_dir = project_root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory: {data_dir}")
    
    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file_id:
            output_filename = f"{file_prefix}_{file_id}_{timestamp}"
        else:
            output_filename = f"{file_prefix}_{timestamp}"
    
    output_path = data_dir / output_filename
    
    # Add file suffix if provided
    if file_suffix:
        if not file_suffix.startswith('.'):
            file_suffix = f'.{file_suffix}'
        output_path = output_path.with_suffix(file_suffix)
    
    return output_path

def export_superset_query_with_pagination(
    superset_url: str,
    username: str,
    password: str,
    output_filename: Optional[str] = None,
    chunk_size: int = constants.SQL_CHUNK_SIZE,
    timeout: int = constants.API_TIMEOUT_LIMIT,
    verbose: bool = False
) -> str:
    """
    Export all data from a Superset saved query using pagination to bypass the 10,000 row limit.
    
    This method uses the same approach as the superset_export.py script to fetch data in chunks
    and combine them into a single CSV file.
    
    Args:
        superset_url: Base URL of the Superset instance (e.g., 'https://superset.example.com')
        username: Superset username for authentication
        password: Superset password for authentication
        output_filename: Optional custom filename for the CSV (without extension)
        chunk_size: Number of rows to fetch per chunk (default: 10000)
        timeout: Request timeout in seconds (default: 120)
        verbose: Whether to show detailed progress output (default: False)
        
    Returns:
        Path to the saved CSV file
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If authentication or data retrieval fails
    """
    # Validate input parameters
    if not superset_url or not username or not password :
        raise ValueError("superset_url, username, password are required")
    
    # Clean up the base URL
    superset_url = superset_url.rstrip('/')
    if not superset_url.startswith(('http://', 'https://')):
        superset_url = f'https://{superset_url}'
    
    # Ensure data directory exists and get output path
    output_path = ensure_data_directory_and_get_filename(
        output_filename=output_filename,
        file_prefix="superset_export",
        file_suffix="csv"
    )
    
    try:
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
                headers['x-csrftoken'] = csrf_token
                headers['Referer'] = superset_url+"/sqllab"
                headers['Content-Type'] = 'application/json'
        
        # # Get saved query details
        # saved_query_url = f'{superset_url}/api/v1/saved_query/{query_id}'
        # response = session.get(saved_query_url, headers=headers, timeout=timeout)
        #
        # if response.status_code != 200:
        #     raise RuntimeError(f"Failed to get saved query: {response.text}")
        #
        # query_data = response.json()
        # result = query_data.get('result', {})
        #
        # query_details = {
        #     'sql': result.get('sql', ''),
        #     'database_id': result.get('database', {}).get('id', ''),
        #     'label': result.get('label', 'Unknown'),
        #     'schema': result.get('schema', '')
        # }
        #
        # if verbose:
        #     print(f"   Query: {query_details['label']}")
        #     print(f"   Database ID: {query_details['database_id']}")
        #     print(f"   Schema: {query_details['schema']}")
        #     print()
        #
        # Execute paginated query
        execute_url = f'{superset_url}/api/v1/sqllab/execute/'
        all_data = []
        all_columns = None
        offset = 0
        total_rows = 0
        chunk_num = 1
        base_sql = sql_queries.SQL_QUERIES["opportunity_uservisit"]
        
        while True:
            # Add OFFSET and LIMIT to the SQL
            paginated_sql = f"""
            {base_sql.rstrip(';')}
            OFFSET {offset}
            LIMIT {chunk_size}
            """
            
            payload = {
                "ctas_method": "TABLE",
                "database_id": constants.DATABASE_ID,
                "expand_data": constants.FALSE,
                "json": constants.TRUE,
                "queryLimit": chunk_size,
                "runAsync": constants.FALSE,
                "schema": constants.SCHEMA_PUBLIC,
                "select_as_cta": constants.FALSE,
                "sql": paginated_sql,
                "templateParams": "",
                "tmp_table_name": ""
            }
            
            # print(execute_url)
            # print(headers)
            # print(payload)
            response = session.post(execute_url, json=payload, headers=headers, timeout=timeout)
            result = response.json()

            if response.status_code != 200:
                break
            if result.get('status') != 'success':
                break
            
            # Get data and columns
            chunk_data = result.get('data', [])
            columns = result.get('columns', [])
            if not chunk_data:
                break
            # Store columns from first chunk
            if all_columns is None:
                all_columns = columns
                
            # Add chunk data to overall results
            all_data.extend(chunk_data)
            chunk_rows = len(chunk_data)
            total_rows += chunk_rows
            
            # If we got fewer rows than chunk_size, we've reached the end
            if chunk_rows < chunk_size:
                break
            
            # Prepare for next chunk
            offset += chunk_size
            chunk_num += 1
            
            # Small delay to be nice to the server
            time.sleep(0.5)
         
        if verbose:
            print(f"\n🎯 Pagination complete!")
            print(f"   Total rows fetched: {total_rows:,}")
            print(f"   Total chunks: {chunk_num - 1}")
        else:
            print(f"Exported {total_rows:,} rows from Superset query")
        
        # Export data to CSV
        if not all_data or not all_columns:
            raise RuntimeError("No data was retrieved from the query")
        
        # Create DataFrame
        column_names = [col.get('name', f'col_{i}') for i, col in enumerate(all_columns)]
        df = pd.DataFrame(all_data, columns=column_names)
        
        # Export to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Data exported to: {output_path}")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"Error exporting data from Superset: {str(e)}")
    finally:
        if 'session' in locals():
            session.close()

def load_service_delivery_df_by_opportunity_from_superset(superset_url, superset_username, superset_password) -> Dict[str, pd.DataFrame]:
    """
    Load service delivery data from Superset and group by unique opportunity_name values.
    
    Uses the Superset export functionality to get all service delivery data and then
    groups it by opportunity_name, similar to load_service_delivery_df_by_opportunity
    but fetching from Superset instead of a local CSV file.
    
    Returns:
        Dictionary with opportunity_name as key and DataFrame of service points as value
        
    Raises:
        ValueError: If required environment variables are missing or if data has issues
        RuntimeError: If Superset export fails
    """
    try:
        # Use the new export function to get the data
        csv_path = export_superset_query_with_pagination(
            superset_url=superset_url,
            username=superset_username,
            password=superset_password
        )
        
        # Load the exported CSV data
        df = pd.read_csv(csv_path)
        
        # Check if opportunity_name column exists
        if 'opportunity_name' not in df.columns:
            raise ValueError("CSV data from Superset must contain an 'opportunity_name' column")
        
        return group_service_delivery_df_by_opportunity(df)
        
    except Exception as e:
        raise RuntimeError(f"Error loading service delivery data from Superset: {str(e)}")

def group_service_delivery_df_by_opportunity(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group service delivery DataFrame by unique opportunity_name values.
    
    Args:
        df: DataFrame containing service delivery data with 'opportunity_name' column
        
    Returns:
        Dictionary with opportunity_name as key and DataFrame of service points as value
        
    Raises:
        ValueError: If the DataFrame doesn't contain an 'opportunity_name' column
    """
    # Check if opportunity_name column exists
    if 'opportunity_name' not in df.columns:
        raise ValueError("DataFrame must contain an 'opportunity_name' column")
    
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

if __name__ == "__main__":
    # Only run the test if this file is executed directly
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Superset configuration from environment variables
    # Replace these variable names with your actual .env variable names
    superset_url = os.getenv('SUPERSET_URL')  # e.g., 'https://superset.example.com'
    superset_username = os.getenv('SUPERSET_USERNAME')  # Your Superset username
    superset_password = os.getenv('SUPERSET_PASSWORD')  # Your Superset password
    
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
            password=superset_password
        )
        print(f"Success! CSV saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error retrieving data from Superset: {str(e)}")
        exit(1)
    