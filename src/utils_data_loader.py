import os
import glob
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import List, Dict, Any, Optional
import numpy as np

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