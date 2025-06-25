"""Basic FLW analysis - visit counts, dates, approval rates, case analysis, and location metrics"""

import pandas as pd
import numpy as np
from .location_analyzer import LocationAnalyzer


class BasicAnalyzer:
    def __init__(self, df, flw_id_col, log_func, auto_detect_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
        self.auto_detect_column = auto_detect_func
    
    def analyze(self, min_visits=1):
        """Generate basic FLW analysis including case metrics and location data"""
        flw_list = self.df[self.flw_id_col].unique()
        flw_list = flw_list[pd.notna(flw_list)]
        
        results = []
        
        # Generate location metrics for all FLWs
        location_analyzer = LocationAnalyzer(self.df, self.flw_id_col, self.log, self.auto_detect_column)
        location_metrics_df = location_analyzer.analyze_flw_location_metrics()
        
        for flw_id in flw_list:
            flw_data = self.df[self.df[self.flw_id_col] == flw_id].copy()
            
            total_visits = len(flw_data)
            if total_visits < min_visits:
                continue
            
            # Calculate approval percentage
            pct_approved = None
            if 'status' in flw_data.columns:
                approved_count = (flw_data['status'] == 'approved').sum()
                pct_approved = round(approved_count / total_visits, 3) if total_visits > 0 else 0
            
            # Date analysis
            first_visit = flw_data['visit_date'].min().date()
            last_visit = flw_data['visit_date'].max().date()
            total_days_with_visits = flw_data['visit_day'].nunique()
            
            # Calculate median visits per day
            daily_visits = flw_data.groupby('visit_day').size()
            median_visits_per_day = daily_visits.median()
            
            # Case analysis
            case_metrics = self._analyze_cases(flw_data)
            
            # Get location metrics for this FLW
            location_metrics = {}
            flw_location_data = location_metrics_df[location_metrics_df[self.flw_id_col] == flw_id]
            if len(flw_location_data) > 0:
                location_metrics = {
                    'median_distance_traveled_per_multi_visit_day': flw_location_data['median_distance_traveled_per_multi_visit_day'].iloc[0],
                    'avg_bounding_box_area_multi_visit_cases': flw_location_data['avg_bounding_box_area_multi_visit_cases'].iloc[0]
                }
            
            # Get metadata
            opportunity_name = self._get_most_common_value(flw_data, 'opportunity_name')
            flw_name = self._get_most_common_value(flw_data, 'flw_name')
            
            # Build result
            result_row = {
                self.flw_id_col: flw_id,
                'total_visits': total_visits,
                'first_visit_date': first_visit,
                'last_visit_date': last_visit,
                'total_days_with_visits': total_days_with_visits,
                'median_visits_per_day': median_visits_per_day
            }
            
            # Add case metrics
            result_row.update(case_metrics)
            
            # Add location metrics
            result_row.update(location_metrics)
            
            if flw_name:
                result_row['flw_name'] = flw_name
            if opportunity_name:
                result_row['opportunity_name'] = opportunity_name
            if pct_approved is not None:
                result_row['pct_approved'] = pct_approved
                
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('last_visit_date', ascending=False)
        
        self.log(f"Basic analysis complete: {len(results_df)} FLWs with >={min_visits} visits")
        return results_df
    
    def _analyze_cases(self, flw_data):
        """Analyze case-related metrics for a single FLW"""
        case_metrics = {
            'number_cases': None,
            'percent_active_days_with_2plus_cases': None,
            'percent_cases_with_2plus_visits': None
        }
        
        # Check if case_id column exists
        if 'case_id' not in flw_data.columns:
            return case_metrics
        
        # Filter out null case_ids
        case_data = flw_data[flw_data['case_id'].notna() & (flw_data['case_id'] != '')].copy()
        
        if len(case_data) == 0:
            return case_metrics
        
        # 1. Count unique cases
        number_cases = case_data['case_id'].nunique()
        case_metrics['number_cases'] = number_cases
        
        # 2. Percent active days with 2+ cases
        if 'visit_day' in case_data.columns:
            daily_cases = case_data.groupby('visit_day')['case_id'].nunique()
            total_active_days = len(daily_cases)
            days_with_2plus_cases = (daily_cases >= 2).sum()
            
            if total_active_days > 0:
                case_metrics['percent_active_days_with_2plus_cases'] = round(
                    days_with_2plus_cases / total_active_days, 3
                )
        
        # 3. Percent cases with 2+ visits
        visits_per_case = case_data.groupby('case_id').size()
        total_cases = len(visits_per_case)
        cases_with_2plus_visits = (visits_per_case >= 2).sum()
        
        if total_cases > 0:
            case_metrics['percent_cases_with_2plus_visits'] = round(
                cases_with_2plus_visits / total_cases, 3
            )
        
        return case_metrics
    
    def _get_most_common_value(self, df, column_name):
        """Get the most common value in a column"""
        if column_name not in df.columns:
            return None
        value_counts = df[column_name].value_counts()
        return value_counts.index[0] if len(value_counts) > 0 else None
