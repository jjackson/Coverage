# analyzers/work_window_analyzer.py
"""
Work Window Analysis - calculates daily work patterns and efficiency metrics
"""

import pandas as pd
import numpy as np


class WorkWindowAnalyzer:
    def __init__(self, df, flw_id_col, log_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
    
    def analyze(self):
        """Generate work window analysis for all FLWs"""
        # First, create daily summaries for each FLW
        daily_summary = self._create_daily_summaries()
        
        if len(daily_summary) == 0:
            self.log("No data available for work window analysis")
            return pd.DataFrame()
        
        # Then aggregate by FLW to get overall patterns
        flw_summary = self._aggregate_by_flw(daily_summary)
        
        self.log(f"Work window analysis complete: {len(flw_summary)} FLWs analyzed")
        return flw_summary
    
    def _create_daily_summaries(self):
        """Create daily work window summaries for each FLW-day combination"""
        daily_data = []
        
        for flw_id in self.df[self.flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = self.df[self.df[self.flw_id_col] == flw_id].copy()
            
            # Group by day to analyze daily patterns
            for visit_day in flw_data['visit_day'].unique():
                day_data = flw_data[flw_data['visit_day'] == visit_day]
                
                start_time = day_data['visit_date'].min()
                end_time = day_data['visit_date'].max()
                visits_on_day = len(day_data)
                
                # Calculate work window in minutes
                work_window_minutes = (end_time - start_time).total_seconds() / 60
                
                daily_data.append({
                    self.flw_id_col: flw_id,
                    'visit_day': visit_day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'visits_on_day': visits_on_day,
                    'work_window_minutes': work_window_minutes
                })
        
        return pd.DataFrame(daily_data)
    
    def _aggregate_by_flw(self, daily_df):
        """Aggregate daily data to create FLW-level work window metrics"""
        flw_summaries = []
        
        for flw_id in daily_df[self.flw_id_col].unique():
            flw_daily = daily_df[daily_df[self.flw_id_col] == flw_id]
            
            # Calculate average visits per day
            avg_visits_per_day = flw_daily['visits_on_day'].mean()
            
            # Calculate average start and end times
            avg_start_time = self._format_avg_time(flw_daily['start_time'])
            avg_end_time = self._format_avg_time(flw_daily['end_time'])
            
            # Calculate average work window
            avg_work_window_minutes = flw_daily['work_window_minutes'].mean()
            
            # Calculate efficiency metrics
            avg_minutes_per_visit = self._calculate_avg_minutes_per_visit(
                flw_daily['work_window_minutes'], 
                flw_daily['visits_on_day']
            )
            
            visits_per_hour = self._calculate_visits_per_hour(
                flw_daily['visits_on_day'], 
                flw_daily['work_window_minutes']
            )
            
            flw_summaries.append({
                self.flw_id_col: flw_id,
                'avg_visits_per_day': round(avg_visits_per_day, 1),
                'avg_start_time': avg_start_time,
                'avg_end_time': avg_end_time,
                'avg_work_window_minutes': round(avg_work_window_minutes, 1),
                'avg_minutes_per_visit': round(avg_minutes_per_visit, 1),
                'visits_per_hour': visits_per_hour
            })
        
        return pd.DataFrame(flw_summaries)
    
    def _format_avg_time(self, time_series):
        """Format average time as HH:MM"""
        if len(time_series) == 0:
            return None
        
        # Extract hours and minutes
        hours = time_series.dt.hour
        minutes = time_series.dt.minute
        
        # Convert to total minutes since midnight
        total_minutes = hours * 60 + minutes
        
        # Calculate average
        avg_minutes = total_minutes.mean()
        
        # Convert back to hours and minutes
        avg_hours = int(avg_minutes // 60)
        avg_mins = int(avg_minutes % 60)
        
        return f"{avg_hours:02d}:{avg_mins:02d}"
    
    def _calculate_avg_minutes_per_visit(self, window_minutes, visits):
        """Calculate average minutes per visit, handling division by zero"""
        # Safely divide, avoiding division by zero
        result = window_minutes / visits
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.mean()
    
    def _calculate_visits_per_hour(self, visits_series, window_minutes_series):
        """Calculate visits per hour, avoiding extreme values"""
        # Only consider windows of at least 10 minutes to avoid skewed rates
        valid_mask = window_minutes_series >= 10
        
        if not valid_mask.any():
            return 0
        
        # Calculate hourly rates for valid windows
        hourly_rates = 60 * visits_series[valid_mask] / window_minutes_series[valid_mask]
        
        # Get average rate and cap at reasonable maximum
        avg_rate = hourly_rates.mean()
        return round(min(avg_rate, 30), 1)  # Cap at 30 visits per hour (seems reasonable)