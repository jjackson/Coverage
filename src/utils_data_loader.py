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
                       case_type: str = "deliver-unit", 
                       base_url: str = "https://www.commcarehq.org") -> pd.DataFrame:
    """
    Load delivery unit data from CommCare's Case API v2.
    
    Args:
        domain: CommCare project space/domain
        user: Username for authentication
        api_key: API key for authentication
        case_type: Case type to fetch (default: 'delivery-unit')
        base_url: Base URL for the CommCare instance (default: 'https://www.commcarehq.org')
        
    Returns:
        DataFrame with case data formatted similar to Excel import
    """

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
        'limit': 2000  # Fetch 100 cases at a time
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
        
        num_retry = 3
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
        # Extract standard case fields
        case_data = {
            'case_id': case.get('case_id'),
            'case_name': case.get('case_name'),
            'external_id': case.get('external_id'),
            'owner_id': case.get('owner_id'),
            'date_opened': case.get('date_opened'),
            'last_modified': case.get('last_modified'),
            'closed': case.get('closed', False)
        }
        
        # Extract all custom properties
        properties = case.get('properties', {})
        for prop_name, prop_value in properties.items():
            case_data[prop_name] = prop_value
        
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

def load_service_delivery_by_opportunity(csv_file: str) -> Dict[str, pd.DataFrame]:
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

if __name__ == "__main__":
    # Only run the test if this file is executed directly
    parser = argparse.ArgumentParser(description='Test CommCare API data loader')
    parser.add_argument('domain', help='CommCare project space/domain name')
    parser.add_argument('user', help='Username for authentication')
    parser.add_argument('api_key', help='API key for authentication')
    
    args = parser.parse_args()
    
    test_commcare_api_coverage_loader(args.domain, args.user, args.api_key)