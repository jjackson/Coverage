
"""Basic FLW analysis - visit counts, dates, approval rates"""

import pandas as pd
import numpy as np


class BasicAnalyzer:
    def __init__(self, df, flw_id_col, log_func, auto_detect_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
        self.auto_detect_column = auto_detect_func
    
    def analyze(self, min_visits=1):
        """Generate basic FLW analysis"""
        flw_list = self.df[self.flw_id_col].unique()
        flw_list = flw_list[pd.notna(flw_list)]
        
        results = []
        
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
    
    def _get_most_common_value(self, df, column_name):
        """Get the most common value in a column"""
        if column_name not in df.columns:
            return None
        value_counts = df[column_name].value_counts()
        return value_counts.index[0] if len(value_counts) > 0 else None



