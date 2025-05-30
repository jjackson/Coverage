# analyzers/timeline_analyzer.py
"""
Daily Timeline Analysis - generates day-by-day performance matrices
Creates timeline views showing FLW performance across consecutive days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TimelineAnalyzer:
    def __init__(self, df, flw_id_col, log_func, auto_detect_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
        self.auto_detect_column = auto_detect_func
    
    def analyze(self, max_days=30):
        """
        Generate daily timeline analysis showing FLW performance day by day
        
        Args:
            max_days: Maximum number of days to include (0 = all days)
            
        Returns:
            Dictionary of timeline DataFrames (forms timeline, work window timeline)
        """
        self.log("Generating daily timeline analysis...")
        
        # Get FLW data with daily statistics
        flw_data = self._prepare_flw_daily_data(max_days)
        
        if not flw_data:
            self.log("No FLW data available for timeline analysis")
            return {}
        
        # Generate different timeline views
        timelines = {}
        
        # Forms timeline (visit counts per day)
        forms_timeline = self._create_timeline_dataframe(flw_data, 'forms', max_days)
        if len(forms_timeline) > 0:
            timelines['Daily Forms Timeline'] = forms_timeline
        
        # Work window timeline (minutes worked per day)
        work_window_timeline = self._create_timeline_dataframe(flw_data, 'work_window_minutes', max_days)
        if len(work_window_timeline) > 0:
            timelines['Daily Work Window Timeline'] = work_window_timeline
        
        self.log(f"Daily timeline analysis complete: {len(flw_data)} FLWs analyzed")
        return timelines
    
    def _prepare_flw_daily_data(self, max_days):
        """Prepare daily statistics for each FLW"""
        flw_data = []
        
        for flw_id in self.df[self.flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_visits = self.df[self.df[self.flw_id_col] == flw_id].copy()
            
            if len(flw_visits) == 0:
                continue
            
            # Get FLW metadata
            opportunity_name = self._get_most_common_value(flw_visits, 'opportunity_name')
            flw_name = self._get_most_common_value(flw_visits, 'flw_name')
            
            # Get date range for this FLW
            first_visit_date = flw_visits['visit_date'].min().date()
            last_visit_date = flw_visits['visit_date'].max().date()
            
            # Create daily statistics
            daily_stats = self._create_daily_stats(flw_visits, first_visit_date, last_visit_date, max_days)
            
            flw_data.append({
                'flw_id': flw_id,
                'flw_name': flw_name,
                'opportunity_name': opportunity_name,
                'first_visit_date': first_visit_date,
                'daily_stats': daily_stats
            })
        
        return flw_data
    
    def _create_daily_stats(self, flw_visits, first_date, last_date, max_days):
        """Create daily statistics for a single FLW"""
        
        # Calculate the date range to analyze
        if max_days > 0:
            end_date = min(last_date, first_date + timedelta(days=max_days - 1))
        else:
            end_date = last_date
        
        # Create statistics for each day
        current_date = first_date
        daily_stats = {}
        day_number = 1
        
        while current_date <= end_date:
            # Get visits for this specific date
            visits_on_date = flw_visits[flw_visits['visit_date'].dt.date == current_date]
            
            if len(visits_on_date) > 0:
                # Calculate forms count (number of visits/forms)
                forms_count = len(visits_on_date)
                
                # Calculate work window minutes
                start_time = visits_on_date['visit_date'].min()
                end_time = visits_on_date['visit_date'].max()
                work_window_minutes = (end_time - start_time).total_seconds() / 60
                
                daily_stats[day_number] = {
                    'date': current_date,
                    'forms': forms_count,
                    'work_window_minutes': round(work_window_minutes, 1)
                }
            else:
                # No visits on this day
                daily_stats[day_number] = {
                    'date': current_date,
                    'forms': 0,
                    'work_window_minutes': 0
                }
            
            current_date += timedelta(days=1)
            day_number += 1
        
        return daily_stats
    
    def _create_timeline_dataframe(self, flw_data, indicator, max_days):
        """
        Create a timeline dataframe for a specific indicator (forms or work_window_minutes)
        
        Args:
            flw_data: List of FLW data dictionaries
            indicator: 'forms' or 'work_window_minutes'
            max_days: Maximum number of day columns to include
        """
        
        # Determine the maximum number of days we need columns for
        max_day_number = 0
        for flw in flw_data:
            max_day_number = max(max_day_number, len(flw['daily_stats']))
        
        if max_days > 0:
            max_day_number = min(max_day_number, max_days)
        
        if max_day_number == 0:
            return pd.DataFrame()
        
        # Create the timeline data
        timeline_data = []
        
        for flw in flw_data:
            row = {
                'opportunity_name': flw['opportunity_name'],
                self.flw_id_col: flw['flw_id'],
                'flw_name': flw['flw_name'],
                'indicator': indicator
            }
            
            # Add day columns (day_1, day_2, day_3, etc.)
            for day_num in range(1, max_day_number + 1):
                if day_num in flw['daily_stats']:
                    value = flw['daily_stats'][day_num][indicator]
                else:
                    value = 0  # No data for this day
                
                row[f'day_{day_num}'] = value
            
            timeline_data.append(row)
        
        # Convert to DataFrame
        timeline_df = pd.DataFrame(timeline_data)
        
        # Sort by opportunity_name, then flw_id for consistent ordering
        if len(timeline_df) > 0:
            sort_columns = []
            if 'opportunity_name' in timeline_df.columns and timeline_df['opportunity_name'].notna().any():
                sort_columns.append('opportunity_name')
            sort_columns.append(self.flw_id_col)
            
            timeline_df = timeline_df.sort_values(sort_columns)
        
        return timeline_df
    
    def _get_most_common_value(self, df, column_name):
        """Get the most common value in a column, return None if column doesn't exist"""
        if column_name not in df.columns:
            return None
        
        value_counts = df[column_name].value_counts()
        return value_counts.index[0] if len(value_counts) > 0 else None
    
    def get_timeline_summary(self, timeline_data):
        """
        Generate summary statistics about the timeline data
        Useful for understanding patterns across all FLWs
        """
        if not timeline_data:
            return {}
        
        summary = {}
        
        for timeline_name, df in timeline_data.items():
            if len(df) == 0:
                continue
            
            # Get day columns
            day_columns = [col for col in df.columns if col.startswith('day_')]
            
            if not day_columns:
                continue
            
            # Calculate summary stats for each day
            day_stats = {}
            for day_col in day_columns:
                day_values = df[day_col].dropna()
                if len(day_values) > 0:
                    day_stats[day_col] = {
                        'total_flws_active': (day_values > 0).sum(),
                        'avg_value': round(day_values.mean(), 1),
                        'median_value': round(day_values.median(), 1),
                        'max_value': day_values.max()
                    }
            
            summary[timeline_name] = {
                'total_flws': len(df),
                'total_days': len(day_columns),
                'day_statistics': day_stats
            }
        
        return summary