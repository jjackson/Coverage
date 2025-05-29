"""
Microplan Review Report

Generates comprehensive analysis of microplanning data including singleton analysis,
service area summaries, and FLW summaries.
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from math import cos, pi, sqrt
from .base_report import BaseReport


class MicroplanReviewReport(BaseReport):
    """Report that generates comprehensive microplan analysis"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for microplan review report"""
        
        # Large service area threshold
        ttk.Label(parent_frame, text="Large service area threshold (%):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        large_threshold_var = tk.StringVar(value="20")
        ttk.Entry(parent_frame, textvariable=large_threshold_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Use grid approximation for faster processing
        ttk.Label(parent_frame, text="Use fast grid approximation:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        grid_approx_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=grid_approx_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Store variables
        parent_frame.large_threshold_var = large_threshold_var
        parent_frame.grid_approx_var = grid_approx_var
        
    def generate(self):
        """Generate microplan review reports"""
        output_files = []
        
        # Get parameters
        large_threshold = float(self.get_parameter_value('large_threshold', '20'))
        grid_approx = self.get_parameter_value('grid_approx', True)
        
        self.log(f"Starting microplan review analysis with {len(self.df)} delivery units")
        self.log(f"Using large service area threshold: {large_threshold}%")
        self.log(f"Using grid approximation: {grid_approx}")
        
        # Standardize column names to match what we expect
        self.df = self._standardize_column_names(self.df)
        
        # Validate required columns after standardization
        required_cols = ['name', 'WKT', 'buildings', 'service_area_id', 'flw', 'service_area_order']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns after standardization: {', '.join(missing_cols)}")
        
        # Prepare microplan data
        microplan = self.df.copy()
        microplan['visited'] = False  # Initialize as per R code
        
        # Add unique DU IDs if not present
        if 'du_id' not in microplan.columns:
            microplan['du_id'] = [f"DU{i+1}" for i in range(len(microplan))]
        
        # 1. Generate singleton analysis
        self.log("Generating singleton analysis...")
        singleton_data = self._analyze_single_building_dus_fast(microplan, grid_approx)
        
        singleton_file = self.save_csv(singleton_data, "singleton_analysis")
        output_files.append(singleton_file)
        self.log(f"Created: {os.path.basename(singleton_file)}")
        
        # 2. Generate service area summary
        self.log("Generating service area summary...")
        service_summary = self._summarize_by_service_area_basic(microplan)
        service_summary = self._add_approx_travel_distance(service_summary, microplan)
        service_summary = self._add_large_flag(service_summary, large_threshold)
        service_summary = self._add_ward(service_summary, microplan)
        
        service_file = self.save_csv(service_summary, "service_areas")
        output_files.append(service_file)
        self.log(f"Created: {os.path.basename(service_file)}")
        
        # 3. Generate FLW summary
        self.log("Generating FLW summary...")
        flw_summary = self._summarize_by_flw(service_summary)
        
        flw_file = self.save_csv(flw_summary, "flws")
        output_files.append(flw_file)
        self.log(f"Created: {os.path.basename(flw_file)}")
        
        # Log summary statistics
        self._log_summary_statistics(singleton_data, service_summary, flw_summary)
        
        return output_files
    
    def _standardize_column_names(self, df):
        """Standardize column names to match expected format"""
        
        # Create a mapping from actual CSV column names to standardized names
        column_mapping = {
            '#Buildings': 'buildings',
            'Surface Area (sq. meters)': 'surface_area_sq_meters', 
            'distance between adj sides 1': 'distance_between_adj_sides_1',
            'distance between adj sides 2': 'distance_between_adj_sides_2',
            'Ward Name': 'ward_name'
        }
        
        # Apply the mapping
        df_renamed = df.rename(columns=column_mapping)
        
        # Combine OA and service_area_id to create new service_area_id
        if 'OA' in df_renamed.columns and 'service_area_id' in df_renamed.columns:
            df_renamed['service_area_id'] = 'OA' + df_renamed['OA'].astype(str) + '-' + df_renamed['service_area_id'].astype(str)
            self.log(f"Combined OA and service_area_id fields")
        elif 'OA' in df_renamed.columns:
            self.log("Warning: OA column found but no service_area_id column to combine with")
        elif 'service_area_id' in df_renamed.columns:
            self.log("Warning: service_area_id column found but no OA column to combine with")
        
        # Log what columns we have after renaming
        self.log(f"Columns after standardization: {list(df_renamed.columns)}")
        
        return df_renamed
    
    def _analyze_single_building_dus_fast(self, microplan_data, grid_approx=True):
        """Analyze single building delivery units and calculate distances to nearest multi-building DUs"""
        
        result = microplan_data.copy()
        
        # Initialize analysis columns
        result['is_single_building'] = result['buildings'] == 1
        result['distance_to_nearest_multi_building_du'] = np.nan
        result['nearest_multi_building_du_id'] = ''
        result['nearest_multi_building_du_name'] = ''
        result['nearest_multi_building_du_count'] = np.nan
        
        # Count single and multi-building DUs
        single_building_dus = result[result['is_single_building']]
        multi_building_dus = result[~result['is_single_building']]
        
        self.log(f"Found {len(single_building_dus)} single-building DUs out of {len(result)} total DUs "
                f"({100 * len(single_building_dus) / len(result):.1f}%)")
        self.log(f"Found {len(multi_building_dus)} multi-building DUs for distance calculations")
        
        if len(single_building_dus) == 0 or len(multi_building_dus) == 0:
            self.log("Insufficient data for distance calculations")
            return result
        
        # Extract centroids
        self.log("Extracting centroids from WKT polygons...")
        result['centroid_lon'], result['centroid_lat'] = self._extract_centroids_vectorized(result['WKT'])
        
        # Filter valid centroids
        valid_centroids = ~(pd.isna(result['centroid_lon']) | pd.isna(result['centroid_lat']))
        valid_count = valid_centroids.sum()
        
        self.log(f"Successfully extracted {valid_count} valid centroids out of {len(result)} DUs")
        
        if valid_count == 0:
            self.log("No valid centroids found")
            return result
        
        # Get indices for single and multi-building DUs with valid centroids
        single_indices = result.index[result['is_single_building'] & valid_centroids].tolist()
        multi_indices = result.index[~result['is_single_building'] & valid_centroids].tolist()
        
        if len(single_indices) == 0 or len(multi_indices) == 0:
            self.log("Insufficient valid centroids for distance calculations")
            return result
        
        self.log(f"Processing {len(single_indices)} single building DUs against {len(multi_indices)} multi-building DUs")
        
        # Calculate distances
        if grid_approx:
            self._calculate_distances_grid_approx(result, single_indices, multi_indices)
        else:
            self._calculate_distances_haversine(result, single_indices, multi_indices)
        
        # Clean up temporary columns
        result.drop(['centroid_lon', 'centroid_lat'], axis=1, inplace=True)
        
        # Log summary statistics
        self._log_singleton_statistics(result)
        
        return result
    
    def _extract_centroids_vectorized(self, wkt_series):
        """Extract centroids from WKT polygons (vectorized)"""
        
        def extract_centroid_fast(wkt_str):
            if pd.isna(wkt_str) or wkt_str == "":
                return np.nan, np.nan
            
            try:
                # Extract coordinates from WKT polygon
                coords_str = wkt_str.replace("POLYGON ((", "").replace("))", "")
                coord_pairs = coords_str.split(",")[:4]  # Take first 4 pairs for speed
                
                sum_x, sum_y, count = 0, 0, 0
                for pair in coord_pairs:
                    parts = pair.strip().split(" ")
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            sum_x += x
                            sum_y += y
                            count += 1
                        except ValueError:
                            continue
                
                if count > 0:
                    return sum_x / count, sum_y / count
                else:
                    return np.nan, np.nan
            except:
                return np.nan, np.nan
        
        centroids = wkt_series.apply(extract_centroid_fast)
        lons = [c[0] for c in centroids]
        lats = [c[1] for c in centroids]
        
        return lons, lats
    
    def _calculate_distances_grid_approx(self, result, single_indices, multi_indices):
        """Calculate distances using grid approximation for speed"""
        
        self.log("Using grid approximation for faster processing...")
        
        # Get coordinates
        single_lons = result.loc[single_indices, 'centroid_lon'].values
        single_lats = result.loc[single_indices, 'centroid_lat'].values
        multi_lons = result.loc[multi_indices, 'centroid_lon'].values
        multi_lats = result.loc[multi_indices, 'centroid_lat'].values
        
        # Estimate conversion factor from degrees to meters
        center_lat = np.mean(np.concatenate([single_lats, multi_lats]))
        deg_to_m_lat = 111000  # ~111 km per degree latitude
        deg_to_m_lon = 111000 * cos(center_lat * pi / 180)  # Adjust for longitude
        
        self.log(f"Using conversion factors: lat={deg_to_m_lat:.0f} m/deg, lon={deg_to_m_lon:.0f} m/deg")
        
        # Process in batches for memory efficiency
        batch_size = min(1000, len(single_indices))
        n_batches = (len(single_indices) + batch_size - 1) // batch_size
        
        for batch in range(n_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, len(single_indices))
            batch_indices = single_indices[batch_start:batch_end]
            
            if batch % max(1, n_batches // 10) == 0 or batch == n_batches - 1:
                self.log(f"Processing batch {batch + 1} of {n_batches} (DUs {batch_start + 1}-{batch_end})")
            
            # Get batch coordinates
            batch_lons = single_lons[batch_start:batch_end]
            batch_lats = single_lats[batch_start:batch_end]
            
            # Calculate distance matrix using broadcasting
            lon_diff = batch_lons[:, np.newaxis] - multi_lons[np.newaxis, :]
            lat_diff = batch_lats[:, np.newaxis] - multi_lats[np.newaxis, :]
            
            # Convert to meters and calculate Euclidean distance
            lon_diff_m = lon_diff * deg_to_m_lon
            lat_diff_m = lat_diff * deg_to_m_lat
            distance_matrix = np.sqrt(lon_diff_m**2 + lat_diff_m**2)
            
            # Find minimum distance and corresponding index for each single DU
            min_distances = np.min(distance_matrix, axis=1)
            min_indices = np.argmin(distance_matrix, axis=1)
            
            # Update results
            for i, row_idx in enumerate(batch_indices):
                nearest_multi_idx = multi_indices[min_indices[i]]
                
                result.loc[row_idx, 'distance_to_nearest_multi_building_du'] = round(min_distances[i], 2)
                result.loc[row_idx, 'nearest_multi_building_du_id'] = result.loc[nearest_multi_idx, 'du_id']
                result.loc[row_idx, 'nearest_multi_building_du_name'] = result.loc[nearest_multi_idx, 'name']
                result.loc[row_idx, 'nearest_multi_building_du_count'] = result.loc[nearest_multi_idx, 'buildings']
    
    def _calculate_distances_haversine(self, result, single_indices, multi_indices):
        """Calculate distances using precise Haversine formula"""
        
        self.log("Using precise Haversine calculation...")
        
        def haversine_distance(lon1, lat1, lon2, lat2):
            """Calculate Haversine distance between two points"""
            R = 6371000  # Earth's radius in meters
            
            lat1_rad = lat1 * pi / 180
            lat2_rad = lat2 * pi / 180
            delta_lat = (lat2 - lat1) * pi / 180
            delta_lon = (lon2 - lon1) * pi / 180
            
            a = (np.sin(delta_lat / 2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            
            return R * c
        
        progress_interval = max(1, len(single_indices) // 20)
        
        for i, single_idx in enumerate(single_indices):
            if i % progress_interval == 0 or i == 0 or i == len(single_indices) - 1:
                self.log(f"Processing single building DU {i + 1} of {len(single_indices)}")
            
            single_lon = result.loc[single_idx, 'centroid_lon']
            single_lat = result.loc[single_idx, 'centroid_lat']
            
            # Calculate distances to all multi-building DUs
            multi_lons = result.loc[multi_indices, 'centroid_lon'].values
            multi_lats = result.loc[multi_indices, 'centroid_lat'].values
            
            distances = haversine_distance(single_lon, single_lat, multi_lons, multi_lats)
            
            # Find minimum distance
            min_idx = np.argmin(distances)
            nearest_multi_idx = multi_indices[min_idx]
            
            # Update results
            result.loc[single_idx, 'distance_to_nearest_multi_building_du'] = round(distances[min_idx], 2)
            result.loc[single_idx, 'nearest_multi_building_du_id'] = result.loc[nearest_multi_idx, 'du_id']
            result.loc[single_idx, 'nearest_multi_building_du_name'] = result.loc[nearest_multi_idx, 'name']
            result.loc[single_idx, 'nearest_multi_building_du_count'] = result.loc[nearest_multi_idx, 'buildings']
    
    def _summarize_by_service_area_basic(self, microplan_data):
        """Generate basic service area summary"""
        
        # Get unique service areas
        service_areas = microplan_data['service_area_id'].dropna().unique()
        
        summary_data = []
        
        for sa_id in service_areas:
            sa_data = microplan_data[microplan_data['service_area_id'] == sa_id]
            
            # Basic counts
            delivery_count = len(sa_data)
            total_buildings = sa_data['buildings'].sum()
            
            # Get FLW (should be consistent within service area)
            flw_values = sa_data['flw'].dropna().unique()
            flw = flw_values[0] if len(flw_values) > 0 else None
            if len(flw_values) > 1:
                self.log(f"Warning: Service area {sa_id} has multiple FLW values: {flw_values}")
            
            # Get service area order
            order_values = sa_data['service_area_order'].dropna().unique()
            sa_order = order_values[0] if len(order_values) > 0 else None
            if len(order_values) > 1:
                self.log(f"Warning: Service area {sa_id} has multiple order values: {order_values}")
            
            # Calculate percentages
            dus_with_1_building = (sa_data['buildings'] == 1).sum()
            dus_with_5_or_less = (sa_data['buildings'] <= 5).sum()
            
            percent_dus_with_1_building = (dus_with_1_building / delivery_count) * 100
            percent_dus_with_5_or_less = (dus_with_5_or_less / delivery_count) * 100
            
            # Calculate bounding box area
            bbox_area = self._calculate_bbox_area(sa_data)
            
            summary_data.append({
                'service_area_id': sa_id,
                'delivery_unit_count': delivery_count,
                'total_buildings': total_buildings,
                'flw': flw,
                'service_area_order': sa_order,
                'percent_DUs_with_1_building': round(percent_dus_with_1_building, 1),
                'percent_DUs_with_5_or_less_buildings': round(percent_dus_with_5_or_less, 1),
                'bbox_area_sqkm': round(bbox_area, 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df.sort_values('service_area_id')
    
    def _calculate_bbox_area(self, sa_data):
        """Calculate bounding box area from WKT data"""
        
        try:
            # Extract all coordinates from WKT polygons
            all_coords = []
            
            for wkt_str in sa_data['WKT'].dropna():
                if wkt_str:
                    coords_str = wkt_str.replace("POLYGON ((", "").replace("))", "")
                    coord_pairs = coords_str.split(",")
                    
                    for pair in coord_pairs:
                        parts = pair.strip().split(" ")
                        if len(parts) >= 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                all_coords.append((x, y))
                            except ValueError:
                                continue
            
            if not all_coords:
                return 0
            
            # Calculate bounding box
            xs, ys = zip(*all_coords)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            
            # Convert to area in square kilometers
            center_lat = (ymin + ymax) / 2
            lon_scale = cos(center_lat * pi / 180) * 111  # km per degree longitude
            lat_scale = 111  # km per degree latitude
            
            width_km = (xmax - xmin) * lon_scale
            height_km = (ymax - ymin) * lat_scale
            
            return width_km * height_km
            
        except Exception as e:
            self.log(f"Error calculating bbox area: {e}")
            return 0
    
    def _add_approx_travel_distance(self, summary_data, microplan_data):
        """Add approximate travel distance using nearest neighbor algorithm"""
        
        result = summary_data.copy()
        result['approx_travel_distance_km'] = 0.0
        
        for i, row in result.iterrows():
            sa_id = row['service_area_id']
            sa_data = microplan_data[microplan_data['service_area_id'] == sa_id]
            
            if len(sa_data) <= 1:
                continue
            
            # Extract centroids for this service area
            centroids = []
            for wkt_str in sa_data['WKT']:
                try:
                    coords_str = wkt_str.replace("POLYGON ((", "").replace("))", "")
                    coord_pairs = coords_str.split(",")[:4]  # Take first 4 for speed
                    
                    sum_x, sum_y, count = 0, 0, 0
                    for pair in coord_pairs:
                        parts = pair.strip().split(" ")
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            sum_x += x
                            sum_y += y
                            count += 1
                    
                    if count > 0:
                        centroids.append((sum_x / count, sum_y / count))
                except:
                    continue
            
            if len(centroids) <= 1:
                continue
            
            # Calculate travel distance using nearest neighbor algorithm
            travel_distance = self._calculate_travel_distance(centroids)
            result.loc[i, 'approx_travel_distance_km'] = round(travel_distance, 2)
        
        return result
    
    def _calculate_travel_distance(self, centroids):
        """Calculate approximate travel distance using nearest neighbor"""
        
        if len(centroids) <= 1:
            return 0
        
        # Convert to numpy array for easier computation
        coords = np.array(centroids)
        
        # Estimate conversion to kilometers
        center_lat = np.mean(coords[:, 1])
        deg_to_km_lat = 111
        deg_to_km_lon = 111 * cos(center_lat * pi / 180)
        
        visited = [False] * len(coords)
        current = 0
        visited[current] = True
        total_distance = 0
        
        # Visit all points using nearest neighbor
        for _ in range(len(coords) - 1):
            min_dist = float('inf')
            next_point = -1
            
            for j in range(len(coords)):
                if not visited[j]:
                    # Calculate distance
                    dx = (coords[j][0] - coords[current][0]) * deg_to_km_lon
                    dy = (coords[j][1] - coords[current][1]) * deg_to_km_lat
                    dist = sqrt(dx**2 + dy**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        next_point = j
            
            if next_point != -1:
                total_distance += min_dist
                current = next_point
                visited[current] = True
        
        return total_distance
    
    def _add_large_flag(self, service_data, threshold):
        """Add large flag based on percentage of single-building DUs"""
        
        result = service_data.copy()
        result['large'] = result['percent_DUs_with_1_building'] >= threshold
        
        return result
    
    def _add_ward(self, service_data, microplan_data):
        """Add ward information if available"""
        
        result = service_data.copy()
        
        if 'ward_name' not in microplan_data.columns:
            self.log("No ward_name column found in microplan data")
            return result
        
        result['Ward'] = None
        
        for i, row in result.iterrows():
            sa_id = row['service_area_id']
            sa_data = microplan_data[microplan_data['service_area_id'] == sa_id]
            
            if len(sa_data) > 0:
                ward_values = sa_data['ward_name'].dropna().unique()
                if len(ward_values) > 0:
                    if len(ward_values) > 1:
                        self.log(f"Warning: Service area {sa_id} has multiple Ward values: {ward_values}")
                    result.loc[i, 'Ward'] = ward_values[0]
        
        return result
    
    def _summarize_by_flw(self, service_data):
        """Generate FLW-level summary from service area data"""
        
        flws = service_data['flw'].dropna().unique()
        
        flw_summary = []
        
        for flw in flws:
            flw_data = service_data[service_data['flw'] == flw]
            
            # Basic aggregations
            total_service_areas = len(flw_data)
            total_delivery_units = flw_data['delivery_unit_count'].sum()
            total_buildings = flw_data['total_buildings'].sum()
            total_bbox_area = flw_data['bbox_area_sqkm'].sum()
            
            # Calculate percentage of large service areas
            num_large = flw_data['large'].sum()
            percent_large_sas = (num_large / total_service_areas) * 100
            
            # Travel distance if available
            total_travel_distance = 0
            if 'approx_travel_distance_km' in flw_data.columns:
                total_travel_distance = flw_data['approx_travel_distance_km'].sum()
            
            # Ward information if available
            ward = None
            if 'Ward' in flw_data.columns:
                ward_values = flw_data['Ward'].dropna().unique()
                if len(ward_values) > 0:
                    if len(ward_values) > 1:
                        self.log(f"Warning: FLW {flw} has multiple Ward values: {ward_values}")
                        ward = ", ".join(ward_values)
                    else:
                        ward = ward_values[0]
            
            # Check if large SAs are ordered first
            large_first = self._check_large_first(flw_data)
            
            flw_row = {
                'flw': flw,
                'total_service_areas': total_service_areas,
                'total_delivery_units': total_delivery_units,
                'total_buildings': total_buildings,
                'percent_large_SAs': round(percent_large_sas, 1),
                'total_bbox_area_sqkm': round(total_bbox_area, 2),
                'large_first': large_first
            }
            
            if 'approx_travel_distance_km' in flw_data.columns:
                flw_row['total_approx_travel_km'] = round(total_travel_distance, 2)
            
            if ward is not None:
                flw_row['Ward'] = ward
            
            flw_summary.append(flw_row)
        
        return pd.DataFrame(flw_summary).sort_values('flw')
    
    def _check_large_first(self, flw_data):
        """Check if large service areas are ordered first"""
        
        if 'service_area_order' not in flw_data.columns:
            return True  # Default to True if no order info
        
        large_count = flw_data['large'].sum()
        total_count = len(flw_data)
        
        if large_count == 0 or large_count == total_count:
            return True  # All same type
        
        # Sort by order and check if large ones come first
        sorted_data = flw_data.sort_values('service_area_order')
        large_indices = sorted_data['large'].values
        
        # Find last large and first non-large
        large_positions = np.where(large_indices)[0]
        non_large_positions = np.where(~large_indices)[0]
        
        if len(large_positions) > 0 and len(non_large_positions) > 0:
            last_large = large_positions[-1]
            first_non_large = non_large_positions[0]
            return last_large < first_non_large
        
        return True
    
    def _log_singleton_statistics(self, result):
        """Log summary statistics for singleton analysis"""
        
        single_results = result[result['is_single_building'] & 
                              ~pd.isna(result['distance_to_nearest_multi_building_du'])]
        
        if len(single_results) == 0:
            return
        
        distances = single_results['distance_to_nearest_multi_building_du']
        
        self.log(f"Single building DU analysis summary:")
        self.log(f"  Analyzed: {len(single_results)} DUs")
        self.log(f"  Mean distance to nearest multi-building DU: {distances.mean():.2f} meters")
        self.log(f"  Median distance: {distances.median():.2f} meters")
        self.log(f"  Min/Max distance: {distances.min():.2f}/{distances.max():.2f} meters")
        
        # Distance categories
        very_close = (distances <= 100).sum()
        close = ((distances > 100) & (distances <= 500)).sum()
        moderate = ((distances > 500) & (distances <= 1000)).sum()
        far = (distances > 1000).sum()
        
        total = len(single_results)
        self.log(f"  Distance categories:")
        self.log(f"    =100m: {very_close} ({100*very_close/total:.1f}%)")
        self.log(f"    100-500m: {close} ({100*close/total:.1f}%)")
        self.log(f"    500m-1km: {moderate} ({100*moderate/total:.1f}%)")
        self.log(f"    >1km: {far} ({100*far/total:.1f}%)")
    
    def _log_summary_statistics(self, singleton_data, service_summary, flw_summary):
        """Log overall summary statistics"""
        
        total_dus = len(singleton_data)
        total_buildings = singleton_data['buildings'].sum()
        single_building_dus = singleton_data['is_single_building'].sum()
        
        self.log(f"Overall summary:")
        self.log(f"  Total DUs: {total_dus:,}")
        self.log(f"  Total buildings: {total_buildings:,}")
        self.log(f"  Single-building DUs: {single_building_dus:,} ({100*single_building_dus/total_dus:.1f}%)")
        self.log(f"  Service areas: {len(service_summary)}")
        self.log(f"  FLWs: {len(flw_summary)}")
        
        if 'large' in service_summary.columns:
            large_sas = service_summary['large'].sum()
            self.log(f"  Large service areas: {large_sas} ({100*large_sas/len(service_summary):.1f}%)")

