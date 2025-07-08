#code above filtered to show only specific FLW-ids
import pandas as pd
import folium
from folium import Icon, Marker
from geopy.distance import geodesic
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import os

def create_flw_pathway_map(service_df: pd.DataFrame, 
                          flw_ids: Optional[List[str]] = None,
                          output_dir: str = "flw_pathway_maps") -> Dict[str, Any]:
    """
    Create FLW pathway maps from service delivery data.
    
    Args:
        service_df: DataFrame from CoverageData.create_service_points_dataframe()
        flw_ids: Optional list of specific FLW IDs to filter by
        output_dir: Directory to save the map files
        
    Returns:
        Dictionary containing map data and statistics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data if specific FLW IDs are provided
    if flw_ids:
        filtered_data = service_df[service_df['flw_id'].isin(flw_ids)].copy()
        print(f"Filtered to {len(filtered_data)} visits for {len(flw_ids)} FLWs")
    else:
        filtered_data = service_df.copy()
        print(f"Processing all {len(filtered_data)} visits")
    
    if filtered_data.empty:
        print("No data to process")
        return {}
    
    # Ensure required columns exist
    required_columns = ['flw_id', 'flw_name', 'lattitude', 'longitude', 'visit_date']
    missing_columns = [col for col in required_columns if col not in filtered_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean and prepare data
    filtered_data = _prepare_pathway_data(filtered_data)
    
    # Create pathway segments
    pathway_data = _create_pathway_segments(filtered_data)
    
    # Generate maps for each FLW
    map_results = {}
    for flw_id in pathway_data['flw_ids']:
        flw_data = pathway_data['segments'][flw_id]
        map_file = _create_individual_flw_map(flw_data, flw_id, output_dir)
        map_results[flw_id] = {
            'map_file': map_file,
            'segments': len(flw_data),
            'total_distance': sum(seg['distance'] for seg in flw_data),
            'unusual_segments': sum(1 for seg in flw_data if seg['is_unusual'])
        }
    
    # Create summary statistics
    summary_stats = _create_summary_statistics(pathway_data, map_results)
    
    # Save summary data
    summary_file = os.path.join(output_dir, "pathway_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    return {
        'summary_stats': summary_stats,
        'map_results': map_results,
        'pathway_data': pathway_data,
        'summary_file': summary_file
    }

def _prepare_pathway_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean the pathway data."""
    
    # Convert visit_date to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    
    # Remove rows with invalid coordinates or dates
    df = df.dropna(subset=['lattitude', 'longitude', 'visit_date'])
    
    # Sort by FLW, date, and time
    df = df.sort_values(['flw_id', 'visit_date'])
    
    # Add date column for grouping
    df['date'] = df['visit_date'].dt.date
    
    return df

def _create_pathway_segments(df: pd.DataFrame) -> Dict[str, Any]:
    """Create pathway segments between consecutive visits."""
    
    segments = {}
    flw_ids = []
    
    for flw_id in df['flw_id'].unique():
        flw_data = df[df['flw_id'] == flw_id].copy()
        flw_segments = []
        
        # Group by date to create daily pathways
        for date in flw_data['date'].unique():
            daily_data = flw_data[flw_data['date'] == date].copy()
            
            if len(daily_data) < 2:
                continue  # Need at least 2 points to create a segment
            
            # Sort by visit time
            daily_data = daily_data.sort_values('visit_date')
            
            # Create segments between consecutive visits
            for i in range(len(daily_data) - 1):
                current = daily_data.iloc[i]
                next_visit = daily_data.iloc[i + 1]
                
                # Calculate distance
                try:
                    distance = geodesic(
                        (current['lattitude'], current['longitude']),
                        (next_visit['lattitude'], next_visit['longitude'])
                    ).kilometers
                except:
                    distance = 0
                
                # Determine if segment is unusual (distance > 5km or time gap > 2 hours)
                time_diff = (next_visit['visit_date'] - current['visit_date']).total_seconds() / 3600
                is_unusual = distance > 5 or time_diff > 2
                
                segment = {
                    'flw_id': flw_id,
                    'flw_name': current['flw_name'],
                    'date': date,
                    'latitude': current['lattitude'],
                    'longitude': current['longitude'],
                    'lat_next': next_visit['lattitude'],
                    'lon_next': next_visit['longitude'],
                    'visit_date': current['visit_date'],
                    'next_visit_date': next_visit['visit_date'],
                    'distance': distance,
                    'time_diff_hours': time_diff,
                    'is_unusual': is_unusual,
                    'du_name': current['du_name'],
                    'next_du_name': next_visit['du_name']
                }
                
                flw_segments.append(segment)
        
        if flw_segments:
            segments[flw_id] = flw_segments
            flw_ids.append(flw_id)
    
    return {
        'segments': segments,
        'flw_ids': flw_ids,
        'total_segments': sum(len(segs) for segs in segments.values())
    }

def _create_individual_flw_map(flw_segments: List[Dict], flw_id: str, output_dir: str) -> str:
    """Create an individual map for a specific FLW."""
    
    if not flw_segments:
        return ""
    
    # Center the map around the average coordinates
    avg_lat = sum(seg['latitude'] for seg in flw_segments) / len(flw_segments)
    avg_lon = sum(seg['longitude'] for seg in flw_segments) / len(flw_segments)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    
    # Add segments
    for segment in flw_segments:
        if pd.notnull(segment['lat_next']) and pd.notnull(segment['lon_next']):
            coords = [(segment['latitude'], segment['longitude']), 
                     (segment['lat_next'], segment['lon_next'])]
            
            color = 'red' if segment['is_unusual'] else 'blue'
            
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                tooltip=f"{segment['flw_name']} | {segment['date']} | {segment['distance']:.2f} km"
            ).add_to(m)
    
    # Add start and end markers for each day
    daily_groups = {}
    for segment in flw_segments:
        date = segment['date']
        if date not in daily_groups:
            daily_groups[date] = []
        daily_groups[date].append(segment)
    
    for date, day_segments in daily_groups.items():
        if day_segments:
            # Start marker
            start_seg = day_segments[0]
            Marker(
                location=(start_seg['latitude'], start_seg['longitude']),
                popup=f"START: {start_seg['flw_name']} on {date}",
                icon=Icon(color='green', icon='play')
            ).add_to(m)
            
            # End marker
            end_seg = day_segments[-1]
            Marker(
                location=(end_seg['lat_next'], end_seg['lon_next']),
                popup=f"END: {end_seg['flw_name']} on {date}",
                icon=Icon(color='red', icon='stop')
            ).add_to(m)
    
    # Save map
    map_file = os.path.join(output_dir, f"flw_pathway_{flw_id}.html")
    m.save(map_file)
    
    return map_file

def _create_summary_statistics(pathway_data: Dict, map_results: Dict) -> Dict[str, Any]:
    """Create summary statistics for the pathway analysis."""
    
    all_segments = []
    for flw_segments in pathway_data['segments'].values():
        all_segments.extend(flw_segments)
    
    if not all_segments:
        return {}
    
    # Calculate overall statistics
    total_distance = sum(seg['distance'] for seg in all_segments)
    unusual_segments = sum(1 for seg in all_segments if seg['is_unusual'])
    avg_distance = total_distance / len(all_segments) if all_segments else 0
    
    # FLW-level statistics
    flw_stats = []
    for flw_id, flw_segments in pathway_data['segments'].items():
        flw_distance = sum(seg['distance'] for seg in flw_segments)
        flw_unusual = sum(1 for seg in flw_segments if seg['is_unusual'])
        flw_avg_distance = flw_distance / len(flw_segments) if flw_segments else 0
        
        flw_stats.append({
            'flw_id': flw_id,
            'flw_name': flw_segments[0]['flw_name'] if flw_segments else '',
            'total_segments': len(flw_segments),
            'total_distance_km': round(flw_distance, 2),
            'unusual_segments': flw_unusual,
            'avg_segment_distance_km': round(flw_avg_distance, 2),
            'map_file': map_results.get(flw_id, {}).get('map_file', '')
        })
    
    return {
        'summary': {
            'total_flws': len(pathway_data['flw_ids']),
            'total_segments': pathway_data['total_segments'],
            'total_distance_km': round(total_distance, 2),
            'unusual_segments': unusual_segments,
            'unusual_percentage': round((unusual_segments / len(all_segments)) * 100, 1),
            'avg_segment_distance_km': round(avg_distance, 2),
            'generated_at': datetime.now().isoformat()
        },
        'flw_statistics': flw_stats
    }

def create_flw_pathway_dashboard_data(coverage_data_objects: Dict, 
                                    flw_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create pathway data in a format compatible with flw_summary_dashboard.
    
    Args:
        coverage_data_objects: Dictionary of CoverageData objects
        flw_ids: Optional list of specific FLW IDs to filter by
        
    Returns:
        Dictionary with pathway data formatted for dashboard integration
    """
    
    # Combine all service data
    all_service_data = []
    for org, coverage_data in coverage_data_objects.items():
        service_df = coverage_data.create_service_points_dataframe()
        if service_df is not None and not service_df.empty:
            service_df['opportunity'] = org
            all_service_data.append(service_df)
    
    if not all_service_data:
        return {}
    
    combined_service_df = pd.concat(all_service_data, ignore_index=True)
    
    # Create pathway analysis
    pathway_results = create_flw_pathway_map(combined_service_df, flw_ids)
    
    # Format for dashboard integration
    dashboard_data = {
        'pathway_summary': pathway_results.get('summary_stats', {}),
        'flw_pathways': pathway_results.get('map_results', {}),
        'available_maps': list(pathway_results.get('map_results', {}).keys()),
        'total_flws_analyzed': len(pathway_results.get('map_results', {})),
        'filtered_flw_ids': flw_ids if flw_ids else []
    }
    
    return dashboard_data

# Example usage function
def analyze_specific_flw_pathways(coverage_data_objects: Dict, 
                                target_flw_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze pathways for specific FLW IDs.
    
    Args:
        coverage_data_objects: Dictionary of CoverageData objects
        target_flw_ids: List of FLW IDs to analyze
        
    Returns:
        Dictionary with pathway analysis results
    """
    return create_flw_pathway_dashboard_data(coverage_data_objects, target_flw_ids)

# Integration example with existing system
def integrate_with_flw_dashboard(coverage_data_objects: Dict) -> Dict[str, Any]:
    """
    Example of how to integrate FLW pathway analysis with the existing dashboard system.
    
    This function can be called from flw_summary_dashboard.py to add pathway analysis.
    
    Args:
        coverage_data_objects: Dictionary of CoverageData objects from the main system
        
    Returns:
        Dictionary with pathway data ready for dashboard display
    """
    
    # Example: Analyze pathways for all FLWs
    pathway_data = create_flw_pathway_dashboard_data(coverage_data_objects)
    
    # Example: Analyze specific FLW IDs (you can modify this list)
    # specific_flw_ids = ['1298', '1208']  # Replace with actual FLW IDs
    # pathway_data = create_flw_pathway_dashboard_data(coverage_data_objects, specific_flw_ids)
    
    return pathway_data

# Usage example:
if __name__ == "__main__":
    # This shows how to use the module independently
    print("FLW Pathway Analysis Module")
    print("To use with existing system:")
    print("1. Import the module in your dashboard")
    print("2. Call create_flw_pathway_dashboard_data() with your coverage_data_objects")
    print("3. Use the returned data to display pathway information")
    
    # Example integration code:
    """
    # In your dashboard or analysis script:
    from src.flw_pathways import create_flw_pathway_dashboard_data
    
    # Get your coverage data objects (from existing system)
    # coverage_data_objects = load_coverage_data_objects()  # Your existing function
    
    # Analyze all FLWs
    pathway_results = create_flw_pathway_dashboard_data(coverage_data_objects)
    
    # Or analyze specific FLWs
    specific_flw_ids = ['1298', '1208']  # Replace with actual IDs
    pathway_results = create_flw_pathway_dashboard_data(coverage_data_objects, specific_flw_ids)
    
    # Access results
    summary = pathway_results['pathway_summary']
    flw_pathways = pathway_results['flw_pathways']
    
    print(f"Analyzed {summary['summary']['total_flws']} FLWs")
    print(f"Total distance: {summary['summary']['total_distance_km']} km")
    print(f"Unusual segments: {summary['summary']['unusual_segments']}")
    """

