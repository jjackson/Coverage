"""
FLW Metrics Engine - Batch-based analysis for FLW performance metrics
Uses shared batching utility for consistent longitudinal analysis

Save this file as: src/reports/flw_metrics_engine.py
"""

import pandas as pd
import numpy as np
from .utils.batch_utility import BatchUtility
from math import radians, cos, sin, asin, sqrt


class FLWMetricsEngine:
    """Engine for calculating FLW performance metrics using consistent batching"""
    
    def __init__(self, batch_size=300, min_size=100):
        self.batch_utility = BatchUtility(batch_size=batch_size, min_size=min_size)
        self.population_stats = {}
    
    def calculate_batch_metrics(self, batch_df):
        """Calculate metrics for a single batch"""
        metrics = {}
        
        # Add derived fields needed for calculations
        batch_df = self._add_derived_fields(batch_df)
        
        # 1. Time per visit (from work window analysis)
        if 'work_window_minutes' in batch_df.columns and 'visits_this_day' in batch_df.columns:
            # Calculate average minutes per visit for multi-visit days
            multi_visit_data = batch_df[batch_df['visits_this_day'] > 1].copy()
            if len(multi_visit_data) > 0:
                minutes_per_visit = multi_visit_data['work_window_minutes'] / multi_visit_data['visits_this_day']
                metrics['time_per_visit'] = round(minutes_per_visit.mean(), 2)
            else:
                metrics['time_per_visit'] = None
        else:
            metrics['time_per_visit'] = None
        
        # 2. Total distance traveled (if GPS data available)
        if 'latitude' in batch_df.columns and 'longitude' in batch_df.columns:
            gps_data = batch_df[['latitude', 'longitude', 'visit_date']].dropna()
            if len(gps_data) >= 2:
                # Sort by date for chronological order
                gps_data = gps_data.sort_values('visit_date')
                
                total_distance = 0
                for i in range(len(gps_data) - 1):
                    lat1 = gps_data.iloc[i]['latitude']
                    lng1 = gps_data.iloc[i]['longitude']
                    lat2 = gps_data.iloc[i + 1]['latitude']
                    lng2 = gps_data.iloc[i + 1]['longitude']
                    
                    distance = self._haversine_distance(lat1, lng1, lat2, lng2)
                    total_distance += distance
                
                metrics['total_distance_traveled'] = round(total_distance, 3)
            else:
                metrics['total_distance_traveled'] = None
        else:
            metrics['total_distance_traveled'] = None
        
        # 3. Visits per day (average)
        unique_days = batch_df['visit_day'].nunique()
        if unique_days > 0:
            metrics['visits_per_day'] = round(len(batch_df) / unique_days, 2)
        else:
            metrics['visits_per_day'] = None
        
        # 4. Median visits per day
        if unique_days > 0:
            daily_visit_counts = batch_df.groupby('visit_day').size()
            metrics['median_visits_per_day'] = round(daily_visit_counts.median(), 1)
        else:
            metrics['median_visits_per_day'] = None
        
        # 5. Diligence Score - NEW METRIC
        metrics['diligence_score'] = self._calculate_diligence_score(batch_df)
        
        # 6. Total visits - NEW METRIC
        metrics['total_visits'] = len(batch_df)
        
        # 7. Work window in hours - NEW METRIC
        if 'work_window_minutes' in batch_df.columns and 'visits_this_day' in batch_df.columns:
            multi_visit_data = batch_df[batch_df['visits_this_day'] > 1].copy()
            if len(multi_visit_data) > 0:
                work_window_hours = multi_visit_data['work_window_minutes'].mean() / 60
                metrics['work_window_hours'] = round(work_window_hours, 2)
            else:
                metrics['work_window_hours'] = None
        else:
            metrics['work_window_hours'] = None
        
        # 8. Percent "no" under treatment for malnutrition - NEW METRIC
        metrics['percent_no_under_treatment'] = self._calculate_percent_no(batch_df, 'under_treatment_for_mal')
        
        # 9. Deviation from population "no" under treatment - NEW METRIC
        percent_no_treatment = metrics['percent_no_under_treatment']
        if percent_no_treatment is not None and 'percent_no_under_treatment' in self.population_stats:
            pop_percent = self.population_stats['percent_no_under_treatment']
            metrics['deviation_no_under_treatment'] = round(percent_no_treatment - pop_percent, 2)
        else:
            metrics['deviation_no_under_treatment'] = None
        
        # 10. Percent "no" diarrhea last month - NEW METRIC
        metrics['percent_no_diarrhea'] = self._calculate_percent_no(batch_df, 'diarrhea_last_month')
        
        # 11. Deviation from population "no" diarrhea - NEW METRIC
        percent_no_diarrhea = metrics['percent_no_diarrhea']
        if percent_no_diarrhea is not None and 'percent_no_diarrhea' in self.population_stats:
            pop_percent = self.population_stats['percent_no_diarrhea']
            metrics['deviation_no_diarrhea'] = round(percent_no_diarrhea - pop_percent, 2)
        else:
            metrics['deviation_no_diarrhea'] = None
        
        return metrics
    
    def _calculate_diligence_score(self, batch_df):
        """
        Calculate diligence score based on variation in diligence fields.
        FLW gets 1 point for each diligence field where they show multiple different values.
        
        Args:
            batch_df: DataFrame containing the batch data
            
        Returns:
            int: Total diligence score (0 to number of diligence fields)
        """
        # Find all columns that start with "diligence"
        diligence_columns = [col for col in batch_df.columns if col.lower().startswith('diligence')]
        
        if not diligence_columns:
            # No diligence fields found
            return None
        
        diligence_score = 0
        
        for col in diligence_columns:
            # Get non-null values for this diligence field
            values = batch_df[col].dropna()
            
            if len(values) > 0:
                # Count unique values
                unique_values = values.nunique()
                
                # Award 1 point if there are multiple different values
                if unique_values > 1:
                    diligence_score += 1
                # Award 0 points if all values are the same (unique_values == 1)
        
        return diligence_score
    
    def _calculate_percent_no(self, batch_df, column_name):
        """
        Calculate percentage of "no" responses for a binary health variable.
        
        Args:
            batch_df: DataFrame containing the batch data
            column_name: Name of the column to analyze
            
        Returns:
            float: Percentage of "no" responses (0-100), or None if no valid data
        """
        if column_name not in batch_df.columns:
            return None
        
        # Get valid responses (exclude nulls)
        valid_responses = batch_df[column_name].dropna()
        
        # Filter to yes/no responses only
        yes_no_responses = valid_responses[valid_responses.str.lower().isin(['yes', 'no'])]
        
        if len(yes_no_responses) == 0:
            return None
        
        # Calculate percentage of "no" responses
        no_count = (yes_no_responses.str.lower() == 'no').sum()
        total_count = len(yes_no_responses)
        percent_no = (no_count / total_count) * 100
        
        return round(percent_no, 2)
    
    def _calculate_population_stats(self, df_clean):
        """Calculate population-level statistics for deviation metrics"""
        
        self.population_stats = {}
        
        # Calculate population percent "no" for under treatment
        pop_percent_no_treatment = self._calculate_percent_no(df_clean, 'under_treatment_for_mal')
        if pop_percent_no_treatment is not None:
            self.population_stats['percent_no_under_treatment'] = pop_percent_no_treatment
            print(f"Population percent 'no' under treatment: {pop_percent_no_treatment:.2f}%")
        
        # Calculate population percent "no" for diarrhea
        pop_percent_no_diarrhea = self._calculate_percent_no(df_clean, 'diarrhea_last_month')
        if pop_percent_no_diarrhea is not None:
            self.population_stats['percent_no_diarrhea'] = pop_percent_no_diarrhea
            print(f"Population percent 'no' diarrhea: {pop_percent_no_diarrhea:.2f}%")
    
    def _add_derived_fields(self, df):
        """Add derived fields needed for metric calculations"""
        df = df.copy()
        
        # Add visit day for work window calculations
        if 'visit_day' not in df.columns:
            df['visit_day'] = df['visit_date'].dt.date
        
        # Add work window calculations if we have multiple visits per day
        if 'flw_id' in df.columns:
            daily_groups = df.groupby(['flw_id', 'visit_day'])
            
            work_window_data = []
            for (flw_id, visit_day), day_data in daily_groups:
                if len(day_data) > 1:  # Multi-visit day
                    start_time = day_data['visit_date'].min()
                    end_time = day_data['visit_date'].max()
                    work_window_minutes = (end_time - start_time).total_seconds() / 60
                    
                    for idx in day_data.index:
                        work_window_data.append({
                            'index': idx,
                            'work_window_minutes': work_window_minutes,
                            'visits_this_day': len(day_data)
                        })
            
            # Add work window data back to main dataframe
            if work_window_data:
                work_window_df = pd.DataFrame(work_window_data).set_index('index')
                df = df.join(work_window_df, how='left')
        
        return df
    
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
    
    def run_metrics_analysis(self, df_clean):
        """Run complete metrics analysis with batching on pre-prepared data"""
        
        print("=== FLW Performance Metrics Analysis ===")
        
        # Calculate population statistics first
        print("Calculating population statistics for deviation metrics...")
        self._calculate_population_stats(df_clean)
        
        # Create batches for all FLW/opportunity pairs using prepared data
        print("Creating batches for performance metrics...")
        batch_records = self.batch_utility.create_all_flw_batches(
            df_clean, 
            'flw_id', 
            'opportunity_name' if 'opportunity_name' in df_clean.columns else None,
            include_all_batch=True
        )
        
        print(f"Created {len(batch_records)} batch records for performance metrics")
        
        # Calculate metrics for each batch
        all_results = []
        
        for batch_record in batch_records:
            batch_df = batch_record['batch_data']
            batch_metrics = self.calculate_batch_metrics(batch_df)
            
            # Get FLW name if available
            flw_name = None
            if 'flw_name' in batch_df.columns and len(batch_df) > 0:
                flw_name_values = batch_df['flw_name'].dropna()
                if len(flw_name_values) > 0:
                    flw_name = flw_name_values.mode().iloc[0] if len(flw_name_values.mode()) > 0 else flw_name_values.iloc[0]
            
            # Create result record for each metric
            for metric_name, metric_value in batch_metrics.items():
                result = {
                    'flw_id': batch_record['flw_id'],
                    'batch_number': batch_record['batch_number'],
                    'batch_start_date': batch_record['batch_start_date'],
                    'batch_end_date': batch_record['batch_end_date'],
                    'total_visits_in_batch': batch_record['visits_in_batch'],
                    'analysis_type': 'performance_metrics',
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'assessment_result': None,
                    'quality_score_name': None,
                    'quality_score_value': None
                }
                
                # Add opportunity name if available
                if 'opportunity_name' in batch_record:
                    result['opportunity_name'] = batch_record['opportunity_name']
                
                # Add FLW name if available
                if flw_name:
                    result['flw_name'] = flw_name
                
                all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        print(f"Performance metrics analysis complete: {len(results_df)} metric records generated")
        
        return results_df
