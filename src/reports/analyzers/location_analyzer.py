"""Location analysis for FLW visits and cases"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


class LocationAnalyzer:
    def __init__(self, df, flw_id_col, log_func, auto_detect_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
        self.auto_detect_column = auto_detect_func
    
    def analyze_flw_location_metrics(self):
        """Generate FLW-level location metrics for integration with basic analysis"""
        flw_list = self.df[self.flw_id_col].unique()
        flw_list = flw_list[pd.notna(flw_list)]
        
        results = []
        
        for flw_id in flw_list:
            flw_data = self.df[self.df[self.flw_id_col] == flw_id].copy()
            
            # Calculate the two new location metrics
            location_metrics = self._calculate_flw_location_metrics(flw_data)
            
            result_row = {
                self.flw_id_col: flw_id,
                'median_distance_traveled_per_multi_visit_day': location_metrics['median_distance_traveled_per_multi_visit_day'],
                'avg_bounding_box_area_multi_visit_cases': location_metrics['avg_bounding_box_area_multi_visit_cases']
            }
            
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        self.log(f"FLW location metrics calculated for {len(results_df)} FLWs")
        return results_df
    
    def analyze_case_locations(self):
        """Generate comprehensive case-level location analysis"""
        # Check for required columns
        lat_col = self._detect_lat_column()
        lng_col = self._detect_lng_column()
        
        if not lat_col or not lng_col:
            self.log("Warning: GPS coordinates not found - skipping case location analysis")
            return None
        
        # Check for case_id column
        if 'case_id' not in self.df.columns:
            self.log("Warning: case_id column not found - skipping case location analysis")
            return None
        
        self.log(f"Using GPS columns: {lat_col}, {lng_col}")
        
        # Filter to cases with GPS data
        gps_data = self.df[
            (self.df['case_id'].notna()) & 
            (self.df['case_id'] != '') &
            (self.df[lat_col].notna()) & 
            (self.df[lng_col].notna())
        ].copy()
        
        if len(gps_data) == 0:
            self.log("No cases with GPS data found")
            return None
        
        # Group by case and analyze each
        case_groups = gps_data.groupby('case_id')
        results = []
        
        for case_id, case_data in case_groups:
            case_analysis = self._analyze_single_case(case_id, case_data, lat_col, lng_col)
            if case_analysis:
                results.append(case_analysis)
        
        if not results:
            return None
        
        results_df = pd.DataFrame(results)
        self.log(f"Case location analysis complete: {len(results_df)} cases analyzed")
        return results_df
    
    def _calculate_flw_location_metrics(self, flw_data):
        """Calculate FLW-level location metrics"""
        metrics = {
            'median_distance_traveled_per_multi_visit_day': None,
            'avg_bounding_box_area_multi_visit_cases': None
        }
        
        # Check for GPS columns
        lat_col = self._detect_lat_column()
        lng_col = self._detect_lng_column()
        
        if not lat_col or not lng_col:
            return metrics
        
        # Filter to visits with GPS data
        gps_data = flw_data[
            (flw_data[lat_col].notna()) & 
            (flw_data[lng_col].notna())
        ].copy()
        
        if len(gps_data) == 0:
            return metrics
        
        # 1. Median distance traveled per multi-visit day
        if 'visit_day' in gps_data.columns and 'visit_date' in gps_data.columns:
            daily_distances = []
            
            for visit_day, day_data in gps_data.groupby('visit_day'):
                if len(day_data) >= 2:  # Multi-visit day
                    # Sort by timestamp for chronological order
                    day_data_sorted = day_data.sort_values('visit_date')
                    
                    total_distance = 0
                    for i in range(len(day_data_sorted) - 1):
                        lat1 = day_data_sorted.iloc[i][lat_col]
                        lng1 = day_data_sorted.iloc[i][lng_col]
                        lat2 = day_data_sorted.iloc[i + 1][lat_col]
                        lng2 = day_data_sorted.iloc[i + 1][lng_col]
                        
                        distance = self._haversine_distance(lat1, lng1, lat2, lng2)
                        total_distance += distance
                    
                    daily_distances.append(total_distance)
            
            if daily_distances:
                metrics['median_distance_traveled_per_multi_visit_day'] = round(np.median(daily_distances), 3)
        
        # 2. Average bounding box area for multi-visit cases
        if 'case_id' in gps_data.columns:
            case_bounding_boxes = []
            
            for case_id, case_data in gps_data.groupby('case_id'):
                if pd.notna(case_id) and case_id != '' and len(case_data) >= 2:
                    bounding_box_area = self._calculate_bounding_box_area(case_data, lat_col, lng_col)
                    if bounding_box_area is not None:
                        case_bounding_boxes.append(bounding_box_area)
            
            if case_bounding_boxes:
                metrics['avg_bounding_box_area_multi_visit_cases'] = round(np.mean(case_bounding_boxes), 6)
        
        return metrics
    
    def _analyze_single_case(self, case_id, case_data, lat_col, lng_col):
        """Analyze a single case's location data"""
        # Sort by visit_date for chronological order
        case_data_sorted = case_data.sort_values('visit_date')
        
        # Basic info
        flw_id = case_data[self.flw_id_col].iloc[0]
        opportunity_name = case_data.get('opportunity_name', pd.Series([None])).iloc[0]
        total_visits = len(case_data_sorted)
        visits_with_gps = len(case_data_sorted)  # Already filtered to GPS-only data
        
        result = {
            'case_id': case_id,
            self.flw_id_col: flw_id,
            'opportunity_name': opportunity_name,
            'total_visits': total_visits,
            'visits_with_gps': visits_with_gps
        }
        
        # Add individual visit coordinates and dates (up to 6 visits)
        for i in range(min(6, len(case_data_sorted))):
            visit = case_data_sorted.iloc[i]
            visit_num = i + 1
            result[f'visit_{visit_num}_lat'] = visit[lat_col]
            result[f'visit_{visit_num}_lng'] = visit[lng_col]
            
            # Handle timezone-aware dates for Excel compatibility
            visit_date = visit['visit_date']
            if hasattr(visit_date, 'tz_localize') and visit_date.tz is not None:
                visit_date = visit_date.tz_localize(None)  # Remove timezone
            result[f'visit_{visit_num}_date'] = visit_date
        
        # Calculate location metrics (only for cases with 2+ visits)
        if total_visits >= 2:
            # Total distance traveled (chronological order)
            total_distance = 0
            for i in range(len(case_data_sorted) - 1):
                lat1 = case_data_sorted.iloc[i][lat_col]
                lng1 = case_data_sorted.iloc[i][lng_col]
                lat2 = case_data_sorted.iloc[i + 1][lat_col]
                lng2 = case_data_sorted.iloc[i + 1][lng_col]
                total_distance += self._haversine_distance(lat1, lng1, lat2, lng2)
            
            result['total_distance_traveled_km'] = round(total_distance, 3)
            
            # Bounding box area
            bounding_box_area = self._calculate_bounding_box_area(case_data_sorted, lat_col, lng_col)
            result['bounding_box_area_km2'] = round(bounding_box_area, 6) if bounding_box_area else None
            
            # Max distance from first visit
            first_lat = case_data_sorted.iloc[0][lat_col]
            first_lng = case_data_sorted.iloc[0][lng_col]
            max_distance_from_first = 0
            
            for i in range(1, len(case_data_sorted)):
                distance = self._haversine_distance(
                    first_lat, first_lng,
                    case_data_sorted.iloc[i][lat_col],
                    case_data_sorted.iloc[i][lng_col]
                )
                max_distance_from_first = max(max_distance_from_first, distance)
            
            result['max_distance_from_first_visit_km'] = round(max_distance_from_first, 3)
            
            # Visit spread radius (from centroid)
            centroid_lat = case_data_sorted[lat_col].mean()
            centroid_lng = case_data_sorted[lng_col].mean()
            
            distances_from_centroid = []
            for _, visit in case_data_sorted.iterrows():
                distance = self._haversine_distance(
                    centroid_lat, centroid_lng,
                    visit[lat_col], visit[lng_col]
                )
                distances_from_centroid.append(distance)
            
            result['visit_spread_radius_km'] = round(np.mean(distances_from_centroid), 3)
            
            # Average distance between visits
            result['avg_distance_between_visits_km'] = round(total_distance / (total_visits - 1), 3)
        
        return result
    
    def _calculate_bounding_box_area(self, case_data, lat_col, lng_col):
        """Calculate the area of the bounding box containing all visits"""
        if len(case_data) < 2:
            return None
        
        min_lat = case_data[lat_col].min()
        max_lat = case_data[lat_col].max()
        min_lng = case_data[lng_col].min()
        max_lng = case_data[lng_col].max()
        
        # Calculate distances for the bounding box
        lat_distance = self._haversine_distance(min_lat, min_lng, max_lat, min_lng)
        lng_distance = self._haversine_distance(min_lat, min_lng, min_lat, max_lng)
        
        return lat_distance * lng_distance
    
    def _haversine_distance(self, lat1, lng1, lat2, lng2):
        """Calculate the great circle distance between two points on Earth in kilometers"""
        # Convert decimal degrees to radians
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def _detect_lat_column(self):
        """Auto-detect latitude column"""
        lat_patterns = ['latitude', 'lat', 'visit_latitude', 'visit_lat', 'gps_lat', 'gps_latitude']
        return self.auto_detect_column(lat_patterns, required=False)
    
    def _detect_lng_column(self):
        """Auto-detect longitude column"""
        lng_patterns = ['longitude', 'lng', 'lon', 'long', 'visit_longitude', 'visit_lng', 'visit_lon', 'gps_lng', 'gps_longitude']
        return self.auto_detect_column(lng_patterns, required=False)
