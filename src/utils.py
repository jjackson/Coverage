import os
import json
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

from src.models import CoverageData


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-serializable objects to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects - convert to dict
        return {key: convert_to_serializable(value) 
                for key, value in obj.__dict__.items()
                if not key.startswith('_')}
    else:
        return obj


def geo_dataframe_to_geojson(gdf: gpd.GeoDataFrame, filename: Optional[str] = None) -> Dict:
    """
    Convert a GeoDataFrame to GeoJSON and optionally save to file.
    
    Args:
        gdf: Input GeoDataFrame
        filename: Optional path to save the GeoJSON
        
    Returns:
        GeoJSON as a dictionary
    """
    # Convert to GeoJSON 
    geojson_str = gdf.to_json()
    geojson_data = json.loads(geojson_str)
    
    # Convert all properties to serializable Python types
    for feature in geojson_data['features']:
        for key, value in list(feature['properties'].items()):
            feature['properties'][key] = convert_to_serializable(value)
    
    # Save to file if filename is provided
    if filename:
        with open(filename, 'w') as f:
            json.dump(geojson_data, f)
    
    return geojson_data


def coverage_data_to_geojson(coverage_data: CoverageData, 
                            include_delivery_units: bool = True,
                            include_service_points: bool = True) -> Dict[str, Dict]:
    """
    Convert CoverageData to GeoJSON format.
    
    Args:
        coverage_data: CoverageData instance
        include_delivery_units: Whether to include delivery units
        include_service_points: Whether to include service points
        
    Returns:
        Dictionary with GeoJSON data for delivery units and service points
    """
    result = {}
    
    if include_delivery_units:
        # Create a list of all delivery unit features
        features = []
        for du_id, du in coverage_data.delivery_units.items():
            # Get the geometry as a dict directly
            geo_series = gpd.GeoSeries([du.geometry])
            # Handle the conversion more carefully
            try:
                # If __geo_interface__ returns a string, parse it
                if isinstance(geo_series.__geo_interface__, str):
                    geometry = json.loads(geo_series.__geo_interface__)
                # If it's already a dict, use it directly
                else:
                    geometry = geo_series.__geo_interface__
                    
                # For single geometry in a collection, extract the first one
                if isinstance(geometry, list) and len(geometry) > 0:
                    geometry = geometry[0]
            except Exception as e:
                # Fallback to a basic polygon if conversion fails
                print(f"Warning: Error converting geometry for DU {du_id}: {e}")
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
                }
            
            # Convert DeliveryUnit to a GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {
                    "id": du.id,
                    "name": du.du_name,
                    "service_area_id": du.service_area_id,
                    "flw": du.flw_id,
                    "status": du.status,
                    "buildings": du.buildings,
                    "surface_area": du.surface_area,
                    "delivery_count": du.delivery_count,
                    "delivery_target": du.delivery_target,
                    "du_checkout_remark": du.du_checkout_remark,
                    "checked_out_date": du.checked_out_date,
                },
                "geometry": geometry
            }
            features.append(feature)
        
        # Create GeoJSON structure
        result["delivery_units"] = {
            "type": "FeatureCollection",
            "features": features
        }
    
    if include_service_points and coverage_data.service_points:
        # Create a list of all service point features
        features = []
        for point in coverage_data.service_points:
            # Get point geometry safely
            try:
                geo_series = gpd.GeoSeries([point.geometry])
                # Handle the conversion more carefully
                if isinstance(geo_series.__geo_interface__, str):
                    geometry = json.loads(geo_series.__geo_interface__)
                else:
                    geometry = geo_series.__geo_interface__
                    
                # For single geometry in a collection, extract the first one
                if isinstance(geometry, list) and len(geometry) > 0:
                    geometry = geometry[0]
            except Exception as e:
                # Fallback to a basic point if conversion fails
                print(f"Warning: Error converting geometry for point {point.id}: {e}")
                geometry = {
                    "type": "Point",
                    "coordinates": [point.longitude, point.latitude]
                }
            
            # Convert ServiceDeliveryPoint to a GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {
                    "id": point.id,
                    "flw_id": point.flw_id,
                    "flw_name": point.flw_name,
                    "visit_date": point.visit_date,
                    "accuracy_in_m": point.accuracy_in_m
                },
                "geometry": geometry
            }
            features.append(feature)
        
        # Create GeoJSON structure
        result["service_points"] = {
            "type": "FeatureCollection",
            "features": features
        }
    
    return result


def create_output_directory(prefix: str = "coverage_output") -> str:
    """
    Create a timestamped output directory.
    
    Args:
        prefix: Prefix for the directory name
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def export_to_html(coverage_data: CoverageData, output_dir: str) -> Tuple[str, str]:
    """
    Export coverage data to HTML files (map and statistics).
    
    Args:
        coverage_data: CoverageData instance
        output_dir: Directory to save the output files
        
    Returns:
        Tuple of paths to the generated map and statistics HTML files
    """
    # Import the specific functions at runtime to avoid circular imports
    from create_delivery_map import create_leaflet_map
    from create_statistics import create_statistics_report
    
    # Create temporary files for the data
    temp_excel = os.path.join(output_dir, "temp_data.xlsx")
    temp_csv = os.path.join(output_dir, "temp_points.csv")
    
    # TODO: Implement conversion of CoverageData back to Excel and CSV formats
    # For now, we'll need to use the original files
    
    # Create the map and statistics
    map_file = create_leaflet_map(temp_excel, temp_csv)
    stats_file = create_statistics_report(temp_excel, temp_csv)
    
    return map_file, stats_file 