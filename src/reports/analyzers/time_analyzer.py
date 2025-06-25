"""
Time Analysis - calculates visit duration and timing metrics
Handles form_start_time and form_end_time as time-only data (no midnight crossover)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta


class TimeAnalyzer:
    def __init__(self, df, flw_id_col, log_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
    
    def analyze(self):
        """Generate time-based analysis for all FLWs"""
        
        # Check if required columns exist
        if not self._has_required_columns():
            self.log("Warning: Missing form_start_time or form_end_time columns - skipping time analysis")
            return None
        
        try:
            # Prepare time data (much simpler now)
            time_df = self._prepare_time_data()
            
            if len(time_df) == 0:
                self.log("No time data found")
                return None
            
            # Log timing data quality
            total_visits = len(time_df)
            visits_with_issues = time_df['has_timing_issues'].sum()
            valid_visits = total_visits - visits_with_issues
            
            self.log(f"Time data: {total_visits} total visits, {valid_visits} with valid timing, {visits_with_issues} with issues")
            
            if valid_visits == 0:
                self.log("No visits with valid timing data")
                return None
            
            # Analyze each FLW
            flw_list = time_df[self.flw_id_col].unique()
            flw_list = flw_list[pd.notna(flw_list)]
            
            results = []
            
            for flw_id in flw_list:
                try:
                    flw_data = time_df[time_df[self.flw_id_col] == flw_id].copy()
                    flw_metrics = self._analyze_flw_timing(flw_id, flw_data)
                    results.append(flw_metrics)
                except Exception as e:
                    self.log(f"Warning: Could not analyze FLW {flw_id}: {str(e)}")
                    continue
            
            if not results:
                self.log("No FLW time analysis results generated")
                return None
            
            results_df = pd.DataFrame(results)
            self.log(f"Time analysis complete: {len(results_df)} FLWs analyzed")
            return results_df
            
        except Exception as e:
            self.log(f"Error in time analysis: {str(e)}")
            return None
    
    def _has_required_columns(self):
        """Check if required time columns exist"""
        required_cols = ['form_start_time', 'form_end_time']
        has_cols = all(col in self.df.columns for col in required_cols)
        if has_cols:
            self.log(f"Found time columns: {required_cols}")
        else:
            missing = [col for col in required_cols if col not in self.df.columns]
            self.log(f"Missing time columns: {missing}")
        return has_cols
    
    def _prepare_time_data(self):
        """Prepare and clean time data - simplified approach"""
        time_df = self.df.copy()
        
        # Identify timing issues
        time_df['has_timing_issues'] = self._identify_timing_issues(time_df)
        
        # Parse time strings and calculate duration
        time_df['visit_duration_minutes'] = self._calculate_visit_duration(time_df)
        
        return time_df
    
    def _identify_timing_issues(self, df):
        """Identify visits with timing issues - simplified for time-only data"""
        issues = pd.Series(False, index=df.index)
        
        # Convert to string and check for missing/empty values
        start_strings = df['form_start_time'].astype(str).str.strip()
        end_strings = df['form_end_time'].astype(str).str.strip()
        
        # Check for missing/empty/invalid values
        invalid_values = {'', 'nan', 'None', 'NaT', 'nat'}
        
        missing_start = (
            df['form_start_time'].isna() | 
            start_strings.isin(invalid_values)
        )
        
        missing_end = (
            df['form_end_time'].isna() | 
            end_strings.isin(invalid_values)
        )
        
        issues |= missing_start | missing_end
        
        # For non-missing times, try to parse and check if start > end
        valid_mask = ~missing_start & ~missing_end
        
        if valid_mask.any():
            try:
                # Parse time strings
                start_times = pd.to_datetime(start_strings[valid_mask], format='%H:%M:%S', errors='coerce')
                end_times = pd.to_datetime(end_strings[valid_mask], format='%H:%M:%S', errors='coerce')
                
                # Mark failed parsing as issues
                parse_failed = start_times.isna() | end_times.isna()
                issues.loc[valid_mask] |= parse_failed
                
                # Check for start > end (no midnight crossover assumed)
                successfully_parsed = valid_mask & ~issues
                if successfully_parsed.any():
                    start_subset = pd.to_datetime(start_strings[successfully_parsed], format='%H:%M:%S')
                    end_subset = pd.to_datetime(end_strings[successfully_parsed], format='%H:%M:%S')
                    invalid_order = start_subset > end_subset
                    issues.loc[successfully_parsed] |= invalid_order
                
            except Exception as e:
                self.log(f"Warning: Could not validate time format: {str(e)}")
                # Mark questionable entries as having issues
                issues.loc[valid_mask] = True
        
        return issues
    
    def _calculate_visit_duration(self, df):
        """Calculate visit duration in minutes - simple time arithmetic"""
        duration = pd.Series(np.nan, index=df.index)
        
        # Only process visits without timing issues
        valid_mask = ~df['has_timing_issues']
        
        if not valid_mask.any():
            return duration
        
        try:
            # Get clean time strings
            start_strings = df.loc[valid_mask, 'form_start_time'].astype(str).str.strip()
            end_strings = df.loc[valid_mask, 'form_end_time'].astype(str).str.strip()
            
            # Parse as datetime objects (we only care about the time part)
            start_times = pd.to_datetime(start_strings, format='%H:%M:%S', errors='coerce')
            end_times = pd.to_datetime(end_strings, format='%H:%M:%S', errors='coerce')
            
            # Calculate duration in minutes
            time_diff = end_times - start_times
            duration_minutes = time_diff.dt.total_seconds() / 60
            
            # Only keep positive durations (should be guaranteed by our validation)
            duration.loc[valid_mask] = duration_minutes
            
        except Exception as e:
            self.log(f"Warning: Could not calculate visit duration: {str(e)}")
        
        return duration
    
    def _analyze_flw_timing(self, flw_id, flw_data):
        """Analyze timing metrics for a single FLW"""
        
        total_visits = len(flw_data)
        timing_issues = flw_data['has_timing_issues'].sum()
        
        result = {
            self.flw_id_col: flw_id,
            'percent_visits_with_timing_issues': round(timing_issues / total_visits, 3) if total_visits > 0 else 0
        }
        
        # Duration metrics (only for visits without timing issues)
        valid_durations = flw_data['visit_duration_minutes'].dropna()
        
        if len(valid_durations) > 0:
            result.update({
                'avg_visit_duration_minutes': round(valid_durations.mean(), 1),
                'median_visit_duration_minutes': round(valid_durations.median(), 1),
                'min_visit_duration_minutes': round(valid_durations.min(), 1),
                'max_visit_duration_minutes': round(valid_durations.max(), 1)
            })
        else:
            result.update({
                'avg_visit_duration_minutes': None,
                'median_visit_duration_minutes': None,
                'min_visit_duration_minutes': None,
                'max_visit_duration_minutes': None
            })
        
        # Time between consecutive visits
        avg_gap = self._calculate_avg_time_between_visits(flw_data)
        result['avg_minutes_between_consecutive_visits'] = avg_gap
        
        return result
    
    def _calculate_avg_time_between_visits(self, flw_data):
        """Calculate average time between consecutive visits on the same day for this FLW"""
        
        # Filter to visits with valid timing and duration data
        valid_visits = flw_data[
            ~flw_data['has_timing_issues'] & 
            flw_data['visit_duration_minutes'].notna()
        ].copy()
        
        if len(valid_visits) < 2:
            return None
        
        try:
            # Add visit_day column for grouping
            valid_visits['visit_day'] = pd.to_datetime(valid_visits['visit_date']).dt.date
            
            # Parse times for each visit
            start_time_strings = valid_visits['form_start_time'].astype(str).str.strip()
            end_time_strings = valid_visits['form_end_time'].astype(str).str.strip()
            
            start_times = pd.to_datetime(start_time_strings, format='%H:%M:%S', errors='coerce')
            end_times = pd.to_datetime(end_time_strings, format='%H:%M:%S', errors='coerce')
            
            valid_visits = valid_visits.assign(
                start_time_parsed=start_times,
                end_time_parsed=end_times
            )
            
            # Calculate gaps within each day
            all_gaps = []
            
            for visit_day, day_data in valid_visits.groupby('visit_day'):
                if len(day_data) < 2:
                    continue  # Need at least 2 visits on same day
                
                # Sort visits by start time within this day
                day_data = day_data.sort_values('start_time_parsed').dropna(subset=['start_time_parsed', 'end_time_parsed'])
                
                if len(day_data) < 2:
                    continue
                
                # Calculate gaps between consecutive visits on this day
                for i in range(len(day_data) - 1):
                    current_end = day_data.iloc[i]['end_time_parsed']
                    next_start = day_data.iloc[i + 1]['start_time_parsed']
                    
                    # Calculate gap in minutes (same day, so just time difference)
                    gap_minutes = (next_start - current_end).total_seconds() / 60
                    
                    # Only include reasonable gaps (positive and less than 12 hours)
                    if 0 <= gap_minutes <= 720:  # 12 hours = 720 minutes
                        all_gaps.append(gap_minutes)
            
            if all_gaps:
                return round(np.mean(all_gaps), 1)
            else:
                return None
                
        except Exception as e:
            self.log(f"Warning: Could not calculate same-day time between visits for FLW {flw_data[self.flw_id_col].iloc[0]}: {str(e)}")
            return None
    
    def get_timing_quality_summary(self, time_results_df):
        """Generate a summary of timing data quality across all FLWs"""
        
        if time_results_df is None or len(time_results_df) == 0:
            return None
        
        timing_issues = time_results_df['percent_visits_with_timing_issues']
        
        summary = {
            'total_flws_analyzed': len(time_results_df),
            'flws_with_no_timing_issues': (timing_issues == 0).sum(),
            'flws_with_some_timing_issues': (timing_issues > 0).sum(),
            'avg_percent_timing_issues': round(timing_issues.mean(), 3),
            'median_percent_timing_issues': round(timing_issues.median(), 3),
            'max_percent_timing_issues': round(timing_issues.max(), 3)
        }
        
        return summary
