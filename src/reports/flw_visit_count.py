"""
FLW Analysis Report

Generates comprehensive FLW analysis combining basic metrics and work window analysis.
Based on the R function run_group_analysis(visits, idu_db, groupBy = "flw_id").
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from .base_report import BaseReport


class FLWAnalysisReport(BaseReport):
    """Comprehensive FLW analysis report combining basic metrics and work window analysis"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for FLW Analysis report"""
        
        # Minimum visits filter
        ttk.Label(parent_frame, text="Minimum visits to include:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        min_visits_var = tk.StringVar(value="1")
        ttk.Entry(parent_frame, textvariable=min_visits_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Include work window analysis
        ttk.Label(parent_frame, text="Include work window analysis:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        include_work_windows_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=include_work_windows_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Generate separate files option
        ttk.Label(parent_frame, text="Generate separate analysis files:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        separate_files_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent_frame, variable=separate_files_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Generate daily timeline analysis
        ttk.Label(parent_frame, text="Generate daily timeline analysis:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        include_daily_timeline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=include_daily_timeline_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Maximum days to include in timeline
        ttk.Label(parent_frame, text="Max days in timeline (0=all):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        max_timeline_days_var = tk.StringVar(value="30")
        ttk.Entry(parent_frame, textvariable=max_timeline_days_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Store variables for access
        parent_frame.min_visits_var = min_visits_var
        parent_frame.include_work_windows_var = include_work_windows_var
        parent_frame.separate_files_var = separate_files_var
        parent_frame.include_daily_timeline_var = include_daily_timeline_var
        parent_frame.max_timeline_days_var = max_timeline_days_var

        # Add opportunity summary option
        ttk.Label(parent_frame, text="Generate opportunity summary:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        generate_opp_summary_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent_frame, variable=generate_opp_summary_var).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        parent_frame.generate_opp_summary_var = generate_opp_summary_var

        
    def generate(self):
        """Generate comprehensive FLW analysis reports"""
        output_files = []
        
        # Get parameters
        min_visits = int(self.get_parameter_value('min_visits', '1'))
        include_work_windows = self.get_parameter_value('include_work_windows', True)
        separate_files = self.get_parameter_value('separate_files', False)
        include_daily_timeline = self.get_parameter_value('include_daily_timeline', True)
        max_timeline_days = int(self.get_parameter_value('max_timeline_days', '30'))
        
        # Auto-detect required columns
        flw_id_patterns = ['flw_id', 'flw id', 'worker_id', 'worker id', 'field_worker_id']
        flw_id_col = self.auto_detect_column(flw_id_patterns, required=True)
        self.log(f"Using '{flw_id_col}' as FLW ID column")
        
        # Clean and prepare data
        self.log("Cleaning and preparing visit data...")
        df = self._clean_visits_data()
        
        # Generate basic analysis
        self.log("Generating basic FLW analysis...")
        basic_results = self._analyze_flw_basic(df, flw_id_col, min_visits)
        
        # Generate work window analysis if requested
        work_window_results = None
        if include_work_windows:
            self.log("Generating work window analysis...")
            work_window_results = self._analyze_work_windows(df, flw_id_col)
        
        # Save results
        if separate_files:
            # Save basic analysis
            basic_file = self.save_csv(basic_results, "flw_basic_analysis")
            output_files.append(basic_file)
            self.log(f"Created: {os.path.basename(basic_file)}")
            
            # Save work window analysis if generated
            if work_window_results is not None:
                work_file = self.save_csv(work_window_results, "flw_work_window_analysis")
                output_files.append(work_file)
                self.log(f"Created: {os.path.basename(work_file)}")
        
        # Always create combined analysis
        if work_window_results is not None:
            self.log("Combining basic and work window analysis...")
            combined_results = self._combine_analyses(basic_results, work_window_results, flw_id_col)
        else:
            combined_results = basic_results
            
        
        generate_opp_summary = self.get_parameter_value('generate_opp_summary', False)

        if generate_opp_summary:
            opp_summary_file = self._generate_opportunity_summary(df)
            if opp_summary_file:
                output_files.append(opp_summary_file)


        # Save combined analysis
        combined_file = self.save_csv(combined_results, "flw_combined_analysis")
        output_files.append(combined_file)
        self.log(f"Created: {os.path.basename(combined_file)}")
        
        # Generate daily timeline analysis if requested
        if include_daily_timeline:
            timeline_files = self._generate_daily_timeline_analysis(df, flw_id_col, max_timeline_days)
            output_files.extend(timeline_files)
        
        # Generate summary statistics
        stats_file = self._generate_flw_statistics(combined_results)
        if stats_file:
            output_files.append(stats_file)
        
        return output_files
    
    def _generate_daily_timeline_analysis(self, df, flw_id_col, max_days):
        """Generate daily timeline analysis showing FLW performance day by day"""
        
        self.log("Generating daily timeline analysis...")
        output_files = []
        
        # Get unique FLWs and their date ranges
        flw_data = []
        
        for flw_id in df[flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_visits = df[df[flw_id_col] == flw_id].copy()
            
            if len(flw_visits) == 0:
                continue
            
            # Get FLW metadata
            opportunity_name = self._get_most_common_value(flw_visits, 'opportunity_name')
            flw_name = self._get_most_common_value(flw_visits, 'flw_name')
            
            # Get date range for this FLW
            first_visit_date = flw_visits['visit_date'].min().date()
            last_visit_date = flw_visits['visit_date'].max().date()
            
            # Create daily aggregations
            daily_stats = self._create_daily_stats(flw_visits, first_visit_date, last_visit_date, max_days)
            
            flw_data.append({
                'flw_id': flw_id,
                'flw_name': flw_name,
                'opportunity_name': opportunity_name,
                'first_visit_date': first_visit_date,
                'daily_stats': daily_stats
            })
        
        if not flw_data:
            self.log("No FLW data available for timeline analysis")
            return output_files
        
        # Generate forms timeline
        forms_timeline = self._create_timeline_dataframe(flw_data, 'forms', max_days)
        if len(forms_timeline) > 0:
            forms_file = self.save_csv(forms_timeline, "flw_daily_forms_timeline")
            output_files.append(forms_file)
            self.log(f"Created: {os.path.basename(forms_file)}")
        
        # Generate work window timeline
        work_window_timeline = self._create_timeline_dataframe(flw_data, 'work_window_minutes', max_days)
        if len(work_window_timeline) > 0:
            work_window_file = self.save_csv(work_window_timeline, "flw_daily_work_window_timeline")
            output_files.append(work_window_file)
            self.log(f"Created: {os.path.basename(work_window_file)}")
        
        self.log(f"Daily timeline analysis complete: {len(flw_data)} FLWs analyzed")
        return output_files
    
    def _get_most_common_value(self, df, column_name):
        """Get the most common value in a column, return None if column doesn't exist"""
        if column_name not in df.columns:
            return None
        
        value_counts = df[column_name].value_counts()
        return value_counts.index[0] if len(value_counts) > 0 else None
    
    def _create_daily_stats(self, flw_visits, first_date, last_date, max_days):
        """Create daily statistics for a single FLW"""
        
        # Calculate the date range to analyze
        if max_days > 0:
            end_date = min(last_date, first_date + timedelta(days=max_days - 1))
        else:
            end_date = last_date
        
        # Create a complete date range
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
        """Create a timeline dataframe for a specific indicator"""
        
        # Determine the maximum number of days we need columns for
        max_day_number = 0
        for flw in flw_data:
            max_day_number = max(max_day_number, len(flw['daily_stats']))
        
        if max_days > 0:
            max_day_number = min(max_day_number, max_days)
        
        if max_day_number == 0:
            return pd.DataFrame()
        
        # Create the base dataframe structure
        timeline_data = []
        
        for flw in flw_data:
            row = {
                'opportunity_name': flw['opportunity_name'],
                'flw_id': flw['flw_id'],
                'flw_name': flw['flw_name'],
                'indicator': indicator
            }
            
            # Add day columns
            for day_num in range(1, max_day_number + 1):
                if day_num in flw['daily_stats']:
                    value = flw['daily_stats'][day_num][indicator]
                else:
                    value = 0  # No data for this day
                
                row[f'day_{day_num}'] = value
            
            timeline_data.append(row)
        
        # Convert to DataFrame and sort
        timeline_df = pd.DataFrame(timeline_data)
        
        # Sort by opportunity_name, then flw_id
        if len(timeline_df) > 0:
            sort_columns = []
            if 'opportunity_name' in timeline_df.columns:
                sort_columns.append('opportunity_name')
            sort_columns.append('flw_id')
            
            timeline_df = timeline_df.sort_values(sort_columns)
        
        return timeline_df
    
    def _clean_visits_data(self):
        """Clean and prepare visits data - equivalent to clean_visits() in R"""
        df = self.df.copy()
        initial_rows = len(df)
        
        # Handle lattitude/longitude columns (fix the misspelling)
        if 'lattitude' in df.columns:
            df['latitude'] = df['lattitude']
            df = df.drop('lattitude', axis=1)
            self.log("Renamed 'lattitude' column to 'latitude'")
        
        # Remove rows with invalid coordinates
        coord_cols = ['latitude', 'longitude'] if 'latitude' in df.columns else ['lattitude', 'longitude'] if 'lattitude' in df.columns else []
        
        if len(coord_cols) == 2 and all(col in df.columns for col in coord_cols):
            before_coord_filter = len(df)
            df = df.dropna(subset=coord_cols)
            df = df[
                (df[coord_cols[0]] >= -90) & (df[coord_cols[0]] <= 90) & 
                (df[coord_cols[1]] >= -180) & (df[coord_cols[1]] <= 180)
            ]
            removed_coords = before_coord_filter - len(df)
            if removed_coords > 0:
                self.log(f"Removed {removed_coords} rows with invalid coordinates")
        
        # Parse visit_date properly (handle potential timezone info like in R)
        if 'visit_date' in df.columns:
            # Try to parse the visit_date, handling various formats
            try:
                df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce', utc=True)
            except:
                df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
            
            # Remove rows with invalid dates
            before_date_filter = len(df)
            df = df.dropna(subset=['visit_date'])
            removed_dates = before_date_filter - len(df)
            if removed_dates > 0:
                self.log(f"Removed {removed_dates} rows with invalid dates")
            
            # Create visit_day column (date only)
            df['visit_day'] = df['visit_date'].dt.date
            
            # Create visit_date_time as copy for time analysis
            df['visit_date_time'] = df['visit_date']
            
        final_rows = len(df)
        self.log(f"Data cleaning complete: {initial_rows} ? {final_rows} rows ({initial_rows - final_rows} removed)")
        
        return df
    
    def _analyze_flw_basic(self, df, flw_id_col, min_visits):
        """Generate basic FLW analysis - equivalent to analyze_by_group_basic() in R"""
        
        # Get unique FLWs
        flw_list = df[flw_id_col].unique()
        flw_list = flw_list[pd.notna(flw_list)]  # Remove NaN values
        
        results = []
        
        for flw_id in flw_list:
            flw_data = df[df[flw_id_col] == flw_id].copy()
            
            # Basic metrics
            total_visits = len(flw_data)
            
            # Skip if below minimum visits threshold
            if total_visits < min_visits:
                continue
            
            # Calculate percent approved if status column exists
            pct_approved = None
            if 'status' in flw_data.columns:
                approved_count = (flw_data['status'] == 'approved').sum()
                pct_approved = round(approved_count / total_visits, 3) if total_visits > 0 else 0
            
            # Date analysis
            first_visit = flw_data['visit_date'].min().date() if len(flw_data) > 0 else None
            last_visit = flw_data['visit_date'].max().date() if len(flw_data) > 0 else None
            
            # Count unique days with visits
            total_days_with_visits = flw_data['visit_day'].nunique()
            
            # Calculate median visits per day (only for days when FLW worked)
            daily_visits = flw_data.groupby('visit_day').size()
            median_visits_per_day = daily_visits.median()
            
            # Get opportunity and FLW name information if available
            opportunity_name = None
            flw_name = None
            
            if 'opportunity_name' in flw_data.columns:
                # Use most common opportunity for this FLW
                opportunity_counts = flw_data['opportunity_name'].value_counts()
                opportunity_name = opportunity_counts.index[0] if len(opportunity_counts) > 0 else None
            
            if 'flw_name' in flw_data.columns:
                # Use most common FLW name (should be consistent for each flw_id)
                name_counts = flw_data['flw_name'].value_counts()
                flw_name = name_counts.index[0] if len(name_counts) > 0 else None
            
            # Build result row
            result_row = {
                flw_id_col: flw_id,
                'total_visits': total_visits,
                'first_visit_date': first_visit,
                'last_visit_date': last_visit,
                'total_days_with_visits': total_days_with_visits,
                'median_visits_per_day': median_visits_per_day
            }
            
            # Add optional columns in logical order
            if flw_name is not None:
                result_row['flw_name'] = flw_name
            if opportunity_name is not None:
                result_row['opportunity_name'] = opportunity_name
            if pct_approved is not None:
                result_row['pct_approved'] = pct_approved
                
            results.append(result_row)
        
        # Convert to DataFrame and sort by last visit date (most recent first)
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('last_visit_date', ascending=False)
        
        self.log(f"Basic analysis complete: {len(results_df)} FLWs with ={min_visits} visits")
        return results_df
    
    def _analyze_work_windows(self, df, flw_id_col):
        """Generate work window analysis - equivalent to analyze_work_windows() in R"""
        
        # Group by FLW and day to find daily work patterns
        daily_summary = []
        
        for flw_id in df[flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = df[df[flw_id_col] == flw_id].copy()
            
            # Group by day
            for visit_day in flw_data['visit_day'].unique():
                day_data = flw_data[flw_data['visit_day'] == visit_day]
                
                start_time = day_data['visit_date'].min()
                end_time = day_data['visit_date'].max()
                visits_on_day = len(day_data)
                
                # Calculate work window in minutes
                work_window_minutes = (end_time - start_time).total_seconds() / 60
                
                daily_summary.append({
                    flw_id_col: flw_id,
                    'visit_day': visit_day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'visits_on_day': visits_on_day,
                    'work_window_minutes': work_window_minutes
                })
        
        daily_df = pd.DataFrame(daily_summary)
        
        if len(daily_df) == 0:
            return pd.DataFrame()
        
        # Aggregate by FLW
        flw_summary = []
        
        for flw_id in daily_df[flw_id_col].unique():
            flw_daily = daily_df[daily_df[flw_id_col] == flw_id]
            
            # Calculate average visits per day
            avg_visits_per_day = flw_daily['visits_on_day'].mean()
            
            # Calculate average start and end times
            avg_start_time = self._format_avg_time(flw_daily['start_time'])
            avg_end_time = self._format_avg_time(flw_daily['end_time'])
            
            # Calculate average work window
            avg_work_window_minutes = flw_daily['work_window_minutes'].mean()
            
            # Calculate efficiency metrics
            avg_minutes_per_visit = self._safe_divide(
                flw_daily['work_window_minutes'], 
                flw_daily['visits_on_day']
            ).mean()
            
            visits_per_hour = self._calculate_visits_per_hour(
                flw_daily['visits_on_day'], 
                flw_daily['work_window_minutes']
            )
            
            flw_summary.append({
                flw_id_col: flw_id,
                'avg_visits_per_day': round(avg_visits_per_day, 1),
                'avg_start_time': avg_start_time,
                'avg_end_time': avg_end_time,
                'avg_work_window_minutes': round(avg_work_window_minutes, 1),
                'avg_minutes_per_visit': round(avg_minutes_per_visit, 1),
                'visits_per_hour': visits_per_hour
            })
        
        results_df = pd.DataFrame(flw_summary)
        self.log(f"Work window analysis complete: {len(results_df)} FLWs analyzed")
        return results_df
    
    def _format_avg_time(self, time_series):
        """Format average time as HH:MM - equivalent to format_avg_time() in R"""
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
    
    def _safe_divide(self, numerator, denominator):
        """Safely divide two series, avoiding division by zero"""
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)
    
    def _calculate_visits_per_hour(self, visits_series, window_minutes_series):
        """Calculate visits per hour, avoiding extreme values"""
        # Only consider windows of at least 10 minutes
        valid_mask = window_minutes_series >= 10
        
        if not valid_mask.any():
            return 0
        
        # Calculate rates for valid windows
        rates = 60 * visits_series[valid_mask] / window_minutes_series[valid_mask]
        
        # Get average rate and cap at reasonable maximum
        avg_rate = rates.mean()
        return round(min(avg_rate, 30), 1)  # Cap at 30 visits per hour
    
    def _combine_analyses(self, basic_results, work_window_results, flw_id_col):
        """Combine basic and work window analysis results"""
        
        if work_window_results is None or len(work_window_results) == 0:
            return basic_results
        
        # Merge the dataframes
        combined = basic_results.merge(
            work_window_results, 
            on=flw_id_col, 
            how='outer'  # Keep all FLWs from both analyses
        )
        
        self.log(f"Combined analysis: {len(combined)} FLWs total")
        return combined
    
    def _generate_flw_statistics(self, flw_results):
        """Generate summary statistics about FLW performance"""
        
        if len(flw_results) == 0:
            self.log("No FLW data to generate statistics")
            return None
        
        try:
            self.log("Generating FLW performance statistics...")
            
            # Basic statistics
            stats = {
                'total_flws': len(flw_results),
                'metric': 'summary'
            }
            
            # Visit count statistics (if available)
            if 'total_visits' in flw_results.columns:
                visits = flw_results['total_visits']
                stats.update({
                    'total_visits_all_flws': visits.sum(),
                    'avg_visits_per_flw': round(visits.mean(), 1),
                    'median_visits_per_flw': visits.median(),
                    'min_visits_per_flw': visits.min(),
                    'max_visits_per_flw': visits.max()
                })
                
                # Performance categories (based on quartiles)
                q25, q75 = visits.quantile([0.25, 0.75])
                low_performers = (visits <= q25).sum()
                high_performers = (visits >= q75).sum()
                
                stats.update({
                    'low_performers_count': low_performers,
                    'high_performers_count': high_performers,
                    'low_performers_pct': round(100 * low_performers / len(flw_results), 1),
                    'high_performers_pct': round(100 * high_performers / len(flw_results), 1)
                })
            
            # Work window statistics (if available)
            if 'avg_work_window_minutes' in flw_results.columns:
                work_windows = flw_results['avg_work_window_minutes'].dropna()
                if len(work_windows) > 0:
                    stats.update({
                        'avg_work_window_minutes': round(work_windows.mean(), 1),
                        'median_work_window_minutes': round(work_windows.median(), 1)
                    })
            
            if 'visits_per_hour' in flw_results.columns:
                efficiency = flw_results['visits_per_hour'].dropna()
                if len(efficiency) > 0:
                    stats.update({
                        'avg_visits_per_hour': round(efficiency.mean(), 1),
                        'median_visits_per_hour': round(efficiency.median(), 1)
                    })
            
            # Convert to DataFrame
            stats_df = pd.DataFrame([stats])
            
            # Save statistics
            stats_file = self.save_csv(stats_df, "flw_analysis_statistics")
            self.log(f"Created: {os.path.basename(stats_file)}")
            
            # Log key findings
            if 'total_visits_all_flws' in stats:
                self.log(f"Summary: {stats['total_flws']} FLWs, "
                        f"{stats['total_visits_all_flws']} total visits, "
                        f"{stats['avg_visits_per_flw']} avg visits per FLW")
            
            return stats_file
            
        except Exception as e:
            self.log(f"Could not generate statistics: {str(e)}")
            return None

    def _generate_opportunity_summary(self, df):
        """Generate opportunity-level summary with one row per opportunity"""
        
        if 'opportunity_name' not in df.columns:
            self.log("No opportunity_name column found - cannot generate opportunity summary")
            return None
            
        self.log("Generating opportunity summary...")
        
        # Auto-detect FLW ID column
        flw_id_patterns = ['flw_id', 'flw id', 'worker_id', 'worker id', 'field_worker_id']
        flw_id_col = self.auto_detect_column(flw_id_patterns, required=True)
        
        opportunities = []
        
        for opp_name in df['opportunity_name'].unique():
            if pd.isna(opp_name):
                continue
                
            opp_data = df[df['opportunity_name'] == opp_name].copy()
            
            if len(opp_data) == 0:
                continue
            
            # Basic metrics
            num_flws = opp_data[flw_id_col].nunique()
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
            flw_day_visits = opp_data.groupby([flw_id_col, 'visit_day']).size()
            median_visits_per_active_day = flw_day_visits.median()
            
            # Calculate average work window minutes per FLW-day
            avg_work_window_minutes = self._calculate_avg_work_window_per_flw_day(opp_data, flw_id_col)
            
            # Calculate median minutes per visit
            median_minutes_per_visit = self._calculate_median_minutes_per_visit(opp_data, flw_id_col)
            
            # Calculate recent activity percentages
            pct_flws_active_last_3_days = self._calculate_recent_activity_pct(opp_data, flw_id_col, last_visit_date, 3)
            pct_flws_active_last_7_days = self._calculate_recent_activity_pct(opp_data, flw_id_col, last_visit_date, 7)
            
            # Calculate weekly medians (weeks 1-8 from first visit)
            weekly_medians = self._calculate_weekly_medians(opp_data, flw_id_col, first_visit_date)
            
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
            
            opportunities.append(result_row)
        
        if not opportunities:
            self.log("No opportunities found for summary")
            return None
        
        # Convert to DataFrame and sort by opportunity name
        summary_df = pd.DataFrame(opportunities)
        summary_df = summary_df.sort_values('opportunity_name')
        
        # Save the summary
        summary_file = self.save_csv(summary_df, "opportunity_summary")
        self.log(f"Created opportunity summary: {os.path.basename(summary_file)} ({len(summary_df)} opportunities)")
        
        return summary_file
    
    def _calculate_avg_work_window_per_flw_day(self, opp_data, flw_id_col):
        """Calculate average work window minutes per FLW-day"""
        
        flw_day_windows = []
        
        for flw_id in opp_data[flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = opp_data[opp_data[flw_id_col] == flw_id]
            
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
    
    def _calculate_median_minutes_per_visit(self, opp_data, flw_id_col):
        """Calculate median minutes per visit across all FLW-days"""
        
        minutes_per_visit_list = []
        
        for flw_id in opp_data[flw_id_col].unique():
            if pd.isna(flw_id):
                continue
                
            flw_data = opp_data[opp_data[flw_id_col] == flw_id]
            
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
    
    def _calculate_recent_activity_pct(self, opp_data, flw_id_col, last_visit_date, days_back):
        """Calculate percentage of FLWs active in the last N days (as decimal 0-1)"""
        
        cutoff_date = last_visit_date - timedelta(days=days_back - 1)  # -1 because we include the last day
        recent_data = opp_data[opp_data['visit_date'].dt.date >= cutoff_date]
        
        total_flws = opp_data[flw_id_col].nunique()
        active_flws = recent_data[flw_id_col].nunique()
        
        return (active_flws / total_flws) if total_flws > 0 else 0
    
    def _calculate_unique_dus_visited(self, opp_data):
        """Calculate total unique DUs visited for this opportunity"""
        
        if 'du_name' not in opp_data.columns:
            # Try alternative column names
            du_columns = ['du_name', 'du name', 'delivery_unit', 'delivery_unit_name']
            du_col = None
            
            for col in du_columns:
                if col in opp_data.columns:
                    du_col = col
                    break
            
            if du_col is None:
                self.log("Warning: No DU name column found - returning 0 for unique DUs")
                return 0
        else:
            du_col = 'du_name'
        
        # Count unique non-null DU names
        unique_dus = opp_data[du_col].dropna().nunique()
        return unique_dus
    
    def _calculate_unique_dus_visited(self, opp_data):
        """Calculate total unique DUs visited for this opportunity"""
        
        if 'du_name' not in opp_data.columns:
            # Try alternative column names
            du_columns = ['du_name', 'du name', 'delivery_unit', 'delivery_unit_name']
            du_col = None
            
            for col in du_columns:
                if col in opp_data.columns:
                    du_col = col
                    break
            
            if du_col is None:
                self.log("Warning: No DU name column found - returning 0 for unique DUs")
                return 0
        else:
            du_col = 'du_name'
        
        # Count unique non-null DU names
        unique_dus = opp_data[du_col].dropna().nunique()
        return unique_dus
    
    def _calculate_weekly_medians(self, opp_data, flw_id_col, first_visit_date):
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
            flw_day_visits = week_data.groupby([flw_id_col, 'visit_day']).size()
            
            if len(flw_day_visits) > 0:
                weekly_medians[week_num] = flw_day_visits.median()
        
        return weekly_medians
    
    def _calculate_estimated_monthly_visits(self, opp_data, last_visit_date):
        """Calculate estimated monthly visits (4 * visits in last 7 days)"""
        
        cutoff_date = last_visit_date - timedelta(days=6)  # Last 7 days including last_visit_date
        recent_data = opp_data[opp_data['visit_date'].dt.date >= cutoff_date]
        
        visits_last_7_days = len(recent_data)
        return visits_last_7_days * 4

