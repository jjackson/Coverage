# analyzers/opportunity_analyzer.py
"""
Opportunity Analysis - generates opportunity-level summaries
Creates one row per opportunity with aggregated metrics across all FLWs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class OpportunityAnalyzer:
    def __init__(self, df, flw_id_col, log_func, auto_detect_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
        self.auto_detect_column = auto_detect_func
    
    def analyze(self):
        """Generate opportunity-level summary with one row per opportunity"""
        
        if 'opportunity_name' not in self.df.columns:
            self.log("No opportunity_name column found - cannot generate opportunity summary")
            return None
            
        self.log("Generating opportunity summary...")
        
        opportunities = []
        
        for opp_name in self.df['opportunity_name'].unique():
            if pd.isna(opp_name):
                continue
                
            opp_data = self.df[self.df['opportunity_name'] == opp_name].copy()
            
            if len(opp_data) == 0:
                continue
            
            # Calculate all metrics for this opportunity
            opp_metrics = self._calculate_opportunity_metrics(opp_data, opp_name)
            opportunities.append(opp_metrics)
        
        if not opportunities:
            self.log("No opportunities found for summary")
            return None
        
        # Convert to DataFrame and sort by opportunity name
        summary_df = pd.DataFrame(opportunities)
        summary_df = summary_df.sort_values('last_visit_date', ascending=False)
        
        self.log(f"Created opportunity summary: {len(summary_df)} opportunities")
        return summary_df
    
    def _calculate_opportunity_metrics(self, opp_data, opp_name):
        """Calculate all metrics for a single opportunity"""
        
        # Basic metrics
        num_flws = opp_data[self.flw_id_col].nunique()
        first_visit_date = opp_data['visit_date'].min().date()
        last_visit_date = opp_data['visit_date'].max().date()
        total_visits = len(opp_data)
        
        # Calculate total active days (unique days with at least one visit)
        total_active_days = opp_data['visit_day'].nunique()
        
        # Calculate total unique DUs visited
        total_unique_dus_visited = self._calculate_unique_dus_visited(opp_data)
        
        # Calculate average DUs per day
        avg_dus_per_day = total_unique_dus_visited / total_active_days if total_active_days > 0 else 0
        
        # Calculate median visits per active day (FLW-day level)
        flw_day_visits = opp_data.groupby([self.flw_id_col, 'visit_day']).size()
        median_visits_per_active_day = flw_day_visits.median()
        
        # Calculate work window metrics
        avg_work_window_minutes = self._calculate_avg_work_window_per_flw_day(opp_data)
        median_minutes_per_visit = self._calculate_median_minutes_per_visit(opp_data)
        
        # Calculate recent activity percentages
        pct_flws_active_last_3_days = self._calculate_recent_activity_pct(opp_data, last_visit_date, 3)
        pct_flws_active_last_7_days = self._calculate_recent_activity_pct(opp_data, last_visit_date, 7)
        
        # Calculate weekly medians (weeks 1-8 from first visit)
        weekly_medians = self._calculate_weekly_medians(opp_data, first_visit_date)
        
        # Calculate estimated monthly visits
        estimated_monthly_visits = self._calculate_estimated_monthly_visits(opp_data, last_visit_date)
        
        # Build result row
        result_row = {
            'opportunity_name': opp_name,
            'num_flws': num_flws,
            'first_visit_date': first_visit_date,
            'last_visit_date': last_visit_date,
            'total_visits': total_visits,
            'total_active_days': total_active_days,
            'total_unique_dus_visited': total_unique_dus_visited,
            'avg_dus_per_day': round(avg_dus_per_day, 1),
            'median_visits_per_active_day': round(median_visits_per_active_day, 1),
            'avg_work_window_minutes': round(avg_work_window_minutes, 1) if avg_work_window_minutes is not None else None,
            'median_minutes_per_visit': round(median_minutes_per_visit, 1) if median_minutes_per_visit is not None else None,
            'pct_flws_active_last_3_days': round(pct_flws_active_last_3_days, 3),
            'pct_flws_active_last_7_days': round(pct_flws_active_last_7_days, 3),
            'estimated_monthly_visits': estimated_monthly_visits
        }
        
        # Add weekly median columns
        for week_num in range(1, 9):
            col_name = f'median_visits_week_{week_num}'
            if week_num in weekly_medians:
                result_row[col_name] = round(weekly_medians[week_num], 1)
            else:
                result_row[col_name] = '.'  # Week hasn't occurred yet
        
        return result_row
    
    def _calculate_unique_dus_visited(self, opp_data):
        """Calculate total unique DUs visited for this opportunity"""
        
        # Try different possible DU column names
        du_columns = ['du_name', 'du name', 'delivery_unit', 'delivery_unit_name']
        du_col = None
        
        for col in du_columns:
            if col in opp_data.columns:
                du_col = col
                break
        
        if du_col is None:
            self.log("Warning: No DU name column found - returning 0 for unique DUs")
            return 0
        
        # Count unique non-null DU names
        unique_dus = opp_data[du_col].dropna().nunique()
        return unique_dus
    
    def _calculate_avg_work_window_per_flw_day(self, opp_data):
        """Calculate average work window minutes per FLW-day"""
        
        flw_day_windows = []
        
        for flw_id in opp_data[self.flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = opp_data[opp_data[self.flw_id_col] == flw_id]
            
            for visit_day in flw_data['visit_day'].unique():
                day_data = flw_data[flw_data['visit_day'] == visit_day]
                
                if len(day_data) > 1:  # Only calculate if multiple visits in a day
                    start_time = day_data['visit_date'].min()
                    end_time = day_data['visit_date'].max()
                    work_window_minutes = (end_time - start_time).total_seconds() / 60
                    flw_day_windows.append(work_window_minutes)
        
        if flw_day_windows:
            return np.mean(flw_day_windows)
        else:
            return None
    
    def _calculate_median_minutes_per_visit(self, opp_data):
        """Calculate median minutes per visit across all FLW-days"""
        
        minutes_per_visit_list = []
        
        for flw_id in opp_data[self.flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = opp_data[opp_data[self.flw_id_col] == flw_id]
            
            for visit_day in flw_data['visit_day'].unique():
                day_data = flw_data[flw_data['visit_day'] == visit_day]
                
                if len(day_data) > 1:  # Only calculate if multiple visits in a day
                    start_time = day_data['visit_date'].min()
                    end_time = day_data['visit_date'].max()
                    work_window_minutes = (end_time - start_time).total_seconds() / 60
                    visits_count = len(day_data)
                    
                    # Calculate minutes per visit for this FLW-day
                    minutes_per_visit = work_window_minutes / visits_count
                    minutes_per_visit_list.append(minutes_per_visit)
        
        if minutes_per_visit_list:
            return np.median(minutes_per_visit_list)
        else:
            return None
    
    def _calculate_recent_activity_pct(self, opp_data, last_visit_date, days_back):
        """Calculate percentage of FLWs active in the last N days (as decimal 0-1)"""
        
        cutoff_date = last_visit_date - timedelta(days=days_back - 1)  # -1 because we include the last day
        recent_data = opp_data[opp_data['visit_date'].dt.date >= cutoff_date]
        
        total_flws = opp_data[self.flw_id_col].nunique()
        active_flws = recent_data[self.flw_id_col].nunique()
        
        return (active_flws / total_flws) if total_flws > 0 else 0
    
    def _calculate_weekly_medians(self, opp_data, first_visit_date):
        """Calculate median visits per FLW-day for each opportunity week (1-8)"""
        
        weekly_medians = {}
        
        for week_num in range(1, 9):
            week_start = first_visit_date + timedelta(days=(week_num - 1) * 7)
            week_end = first_visit_date + timedelta(days=week_num * 7 - 1)
            
            # Filter data for this week
            week_data = opp_data[
                (opp_data['visit_date'].dt.date >= week_start) & 
                (opp_data['visit_date'].dt.date <= week_end)
            ]
            
            if len(week_data) == 0:
                continue  # Week hasn't occurred yet or no data
            
            # Calculate visits per FLW-day for this week
            flw_day_visits = week_data.groupby([self.flw_id_col, 'visit_day']).size()
            
            if len(flw_day_visits) > 0:
                weekly_medians[week_num] = flw_day_visits.median()
        
        return weekly_medians
    
    def _calculate_estimated_monthly_visits(self, opp_data, last_visit_date):
        """Calculate estimated monthly visits (4 * visits in last 7 days)"""
        
        cutoff_date = last_visit_date - timedelta(days=6)  # Last 7 days including last_visit_date
        recent_data = opp_data[opp_data['visit_date'].dt.date >= cutoff_date]
        
        visits_last_7_days = len(recent_data)
        return visits_last_7_days * 4
    
    def get_opportunity_performance_tiers(self, summary_df):
        """
        Categorize opportunities into performance tiers based on key metrics
        Useful for identifying high/low performing opportunities
        """
        if summary_df is None or len(summary_df) == 0:
            return None
        
        # Calculate performance tiers based on visits per FLW
        summary_df = summary_df.copy()
        summary_df['visits_per_flw'] = summary_df['total_visits'] / summary_df['num_flws']
        
        # Define tiers based on quartiles
        q25, q75 = summary_df['visits_per_flw'].quantile([0.25, 0.75])
        
        def categorize_performance(visits_per_flw):
            if visits_per_flw >= q75:
                return 'High'
            elif visits_per_flw <= q25:
                return 'Low'
            else:
                return 'Medium'
        
        summary_df['performance_tier'] = summary_df['visits_per_flw'].apply(categorize_performance)
        
        return summary_df
    
    def get_weekly_trend_analysis(self, summary_df):
        """
        Analyze weekly trends across opportunities
        Shows how visit patterns change over the first 8 weeks
        """
        if summary_df is None or len(summary_df) == 0:
            return None
        
        # Extract weekly median columns
        week_cols = [f'median_visits_week_{i}' for i in range(1, 9)]
        week_cols = [col for col in week_cols if col in summary_df.columns]
        
        if not week_cols:
            return None
        
        # Calculate trends for each opportunity
        trends = []
        
        for _, row in summary_df.iterrows():
            opp_name = row['opportunity_name']
            
            # Get weekly values (excluding '.' placeholders)
            weekly_values = []
            for col in week_cols:
                val = row[col]
                if val != '.' and pd.notna(val):
                    weekly_values.append(float(val))
            
            if len(weekly_values) >= 2:  # Need at least 2 weeks for trend
                # Calculate simple trend (slope)
                weeks = list(range(1, len(weekly_values) + 1))
                trend_slope = np.polyfit(weeks, weekly_values, 1)[0]
                
                trend_direction = 'Increasing' if trend_slope > 0.1 else 'Decreasing' if trend_slope < -0.1 else 'Stable'
                
                trends.append({
                    'opportunity_name': opp_name,
                    'weeks_with_data': len(weekly_values),
                    'first_week_median': weekly_values[0],
                    'last_week_median': weekly_values[-1],
                    'trend_slope': round(trend_slope, 2),
                    'trend_direction': trend_direction
                })
        
        return pd.DataFrame(trends) if trends else None