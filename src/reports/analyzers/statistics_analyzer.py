"""
Statistics Analysis - generates summary statistics across all FLWs
Creates overall performance metrics and distribution analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class StatisticsAnalyzer:
    def __init__(self, flw_results_df, log_func):
        self.flw_results = flw_results_df
        self.log = log_func
    
    def analyze(self):
        """Generate summary statistics about FLW performance"""
        
        if self.flw_results is None or len(self.flw_results) == 0:
            self.log("No FLW data to generate statistics")
            return None
        
        try:
            self.log("Generating FLW performance statistics...")
            
            # Generate multiple types of statistics
            stats_rows = []
            
            # Overall summary statistics
            overall_stats = self._calculate_overall_stats()
            stats_rows.append(overall_stats)
            
            # Visit distribution statistics
            visit_stats = self._calculate_visit_distribution_stats()
            if visit_stats:
                stats_rows.append(visit_stats)
            
            # Work window statistics (if available)
            work_window_stats = self._calculate_work_window_stats()
            if work_window_stats:
                stats_rows.append(work_window_stats)
            
            # Performance tier statistics
            performance_stats = self._calculate_performance_tier_stats()
            if performance_stats:
                stats_rows.append(performance_stats)
            
            # Activity recency statistics
            activity_stats = self._calculate_activity_stats()
            if activity_stats:
                stats_rows.append(activity_stats)
            
            # Convert to DataFrame
            stats_df = pd.DataFrame(stats_rows)
            
            # Log key findings
            self._log_key_findings(overall_stats)
            
            return stats_df
            
        except Exception as e:
            self.log(f"Could not generate statistics: {str(e)}")
            return None
    
    def _calculate_overall_stats(self):
        """Calculate basic overall statistics"""
        
        stats = {
            'statistic_type': 'overall_summary',
            'metric': 'summary',
            'total_flws': len(self.flw_results)
        }
        
        # Visit count statistics (if available)
        if 'total_visits' in self.flw_results.columns:
            visits = self.flw_results['total_visits']
            stats.update({
                'total_visits_all_flws': int(visits.sum()),
                'avg_visits_per_flw': round(visits.mean(), 1),
                'median_visits_per_flw': int(visits.median()),
                'min_visits_per_flw': int(visits.min()),
                'max_visits_per_flw': int(visits.max()),
                'std_visits_per_flw': round(visits.std(), 1)
            })
        
        return stats
    
    def _calculate_visit_distribution_stats(self):
        """Calculate statistics about visit distribution patterns"""
        
        if 'total_visits' not in self.flw_results.columns:
            return None
        
        visits = self.flw_results['total_visits']
        
        # Calculate quartiles and percentiles
        q25, q50, q75 = visits.quantile([0.25, 0.5, 0.75])
        p10, p90 = visits.quantile([0.1, 0.9])
        
        # Performance categories based on quartiles
        low_performers = (visits <= q25).sum()
        high_performers = (visits >= q75).sum()
        
        stats = {
            'statistic_type': 'visit_distribution',
            'metric': 'visit_patterns',
            'q25_visits': int(q25),
            'q50_visits_median': int(q50),
            'q75_visits': int(q75),
            'p10_visits': int(p10),
            'p90_visits': int(p90),
            'low_performers_count': int(low_performers),
            'high_performers_count': int(high_performers),
            'low_performers_pct': round(100 * low_performers / len(self.flw_results), 1),
            'high_performers_pct': round(100 * high_performers / len(self.flw_results), 1)
        }
        
        return stats
    
    def _calculate_work_window_stats(self):
        """Calculate work window and efficiency statistics"""
        
        stats = {
            'statistic_type': 'work_windows',
            'metric': 'efficiency'
        }
        
        # Work window minutes statistics
        if 'avg_work_window_minutes' in self.flw_results.columns:
            work_windows = self.flw_results['avg_work_window_minutes'].dropna()
            if len(work_windows) > 0:
                stats.update({
                    'avg_work_window_minutes': round(work_windows.mean(), 1),
                    'median_work_window_minutes': round(work_windows.median(), 1),
                    'min_work_window_minutes': round(work_windows.min(), 1),
                    'max_work_window_minutes': round(work_windows.max(), 1)
                })
        
        # Visits per hour statistics
        if 'visits_per_hour' in self.flw_results.columns:
            efficiency = self.flw_results['visits_per_hour'].dropna()
            if len(efficiency) > 0:
                stats.update({
                    'avg_visits_per_hour': round(efficiency.mean(), 1),
                    'median_visits_per_hour': round(efficiency.median(), 1),
                    'min_visits_per_hour': round(efficiency.min(), 1),
                    'max_visits_per_hour': round(efficiency.max(), 1)
                })
        
        # Minutes per visit statistics
        if 'avg_minutes_per_visit' in self.flw_results.columns:
            minutes_per_visit = self.flw_results['avg_minutes_per_visit'].dropna()
            if len(minutes_per_visit) > 0:
                stats.update({
                    'avg_minutes_per_visit': round(minutes_per_visit.mean(), 1),
                    'median_minutes_per_visit': round(minutes_per_visit.median(), 1)
                })
        
        # Only return if we have at least some work window data
        if len([k for k in stats.keys() if k not in ['statistic_type', 'metric']]) > 0:
            return stats
        else:
            return None
    
    def _calculate_performance_tier_stats(self):
        """Calculate statistics about performance tiers"""
        
        if 'total_visits' not in self.flw_results.columns:
            return None
        
        visits = self.flw_results['total_visits']
        
        # Define performance tiers
        q33, q67 = visits.quantile([0.33, 0.67])
        
        low_tier = visits <= q33
        medium_tier = (visits > q33) & (visits < q67)
        high_tier = visits >= q67
        
        stats = {
            'statistic_type': 'performance_tiers',
            'metric': 'tier_distribution',
            'low_tier_count': int(low_tier.sum()),
            'medium_tier_count': int(medium_tier.sum()),
            'high_tier_count': int(high_tier.sum()),
            'low_tier_pct': round(100 * low_tier.sum() / len(self.flw_results), 1),
            'medium_tier_pct': round(100 * medium_tier.sum() / len(self.flw_results), 1),
            'high_tier_pct': round(100 * high_tier.sum() / len(self.flw_results), 1),
            'low_tier_avg_visits': round(visits[low_tier].mean(), 1) if low_tier.any() else 0,
            'medium_tier_avg_visits': round(visits[medium_tier].mean(), 1) if medium_tier.any() else 0,
            'high_tier_avg_visits': round(visits[high_tier].mean(), 1) if high_tier.any() else 0
        }
        
        return stats
    
    def _calculate_activity_stats(self):
        """Calculate statistics about recent activity and date patterns"""
        
        stats = {
            'statistic_type': 'activity_patterns',
            'metric': 'temporal_analysis'
        }
        
        # Date range analysis
        if 'first_visit_date' in self.flw_results.columns and 'last_visit_date' in self.flw_results.columns:
            first_dates = pd.to_datetime(self.flw_results['first_visit_date'])
            last_dates = pd.to_datetime(self.flw_results['last_visit_date'])
            
            stats.update({
                'earliest_first_visit': first_dates.min().date(),
                'latest_first_visit': first_dates.max().date(),
                'earliest_last_visit': last_dates.min().date(),
                'latest_last_visit': last_dates.max().date()
            })
            
            # Calculate average tenure (days between first and last visit)
            tenure_days = (last_dates - first_dates).dt.days
            stats.update({
                'avg_tenure_days': round(tenure_days.mean(), 1),
                'median_tenure_days': round(tenure_days.median(), 1),
                'max_tenure_days': int(tenure_days.max())
            })
        
        # Days with visits analysis
        if 'total_days_with_visits' in self.flw_results.columns:
            active_days = self.flw_results['total_days_with_visits']
            stats.update({
                'avg_days_with_visits': round(active_days.mean(), 1),
                'median_days_with_visits': round(active_days.median(), 1),
                'max_days_with_visits': int(active_days.max())
            })
        
        # Visits per day analysis
        if 'median_visits_per_day' in self.flw_results.columns:
            visits_per_day = self.flw_results['median_visits_per_day'].dropna()
            if len(visits_per_day) > 0:
                stats.update({
                    'avg_median_visits_per_day': round(visits_per_day.mean(), 1),
                    'overall_median_visits_per_day': round(visits_per_day.median(), 1)
                })
        
        return stats
    
    def _log_key_findings(self, overall_stats):
        """Log the most important findings from the analysis"""
        
        findings = []
        
        if 'total_flws' in overall_stats:
            findings.append(f"{overall_stats['total_flws']} FLWs analyzed")
        
        if 'total_visits_all_flws' in overall_stats:
            findings.append(f"{overall_stats['total_visits_all_flws']} total visits")
        
        if 'avg_visits_per_flw' in overall_stats:
            findings.append(f"{overall_stats['avg_visits_per_flw']} avg visits per FLW")
        
        if 'median_visits_per_flw' in overall_stats:
            findings.append(f"{overall_stats['median_visits_per_flw']} median visits per FLW")
        
        if findings:
            self.log(f"Key findings: {', '.join(findings)}")
    
    def generate_performance_summary(self):
        """
        Generate a human-readable performance summary
        Useful for executive reporting
        """
        if self.flw_results is None or len(self.flw_results) == 0:
            return "No data available for performance summary"
        
        summary_parts = []
        
        # Basic overview
        total_flws = len(self.flw_results)
        summary_parts.append(f"Analysis of {total_flws} field workers")
        
        if 'total_visits' in self.flw_results.columns:
            total_visits = self.flw_results['total_visits'].sum()
            avg_visits = self.flw_results['total_visits'].mean()
            summary_parts.append(f"Total visits: {total_visits:,} (avg {avg_visits:.1f} per FLW)")
        
        # Performance distribution
        if 'total_visits' in self.flw_results.columns:
            visits = self.flw_results['total_visits']
            q25, q75 = visits.quantile([0.25, 0.75])
            high_performers = (visits >= q75).sum()
            low_performers = (visits <= q25).sum()
            
            summary_parts.append(f"Performance distribution: {high_performers} high performers ({100*high_performers/total_flws:.0f}%), {low_performers} need attention ({100*low_performers/total_flws:.0f}%)")
        
        # Efficiency metrics
        if 'visits_per_hour' in self.flw_results.columns:
            efficiency = self.flw_results['visits_per_hour'].dropna()
            if len(efficiency) > 0:
                avg_efficiency = efficiency.mean()
                summary_parts.append(f"Average efficiency: {avg_efficiency:.1f} visits per hour")
        
        return ". ".join(summary_parts) + "."
    
    def identify_outliers(self, metric_column='total_visits', method='iqr'):
        """
        Identify outlier FLWs based on specified metric
        
        Args:
            metric_column: Column to analyze for outliers
            method: 'iqr' for interquartile range or 'std' for standard deviation
        """
        if metric_column not in self.flw_results.columns:
            return None
        
        values = self.flw_results[metric_column].dropna()
        outlier_indices = []
        
        if method == 'iqr':
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'std':
            mean_val = values.mean()
            std_val = values.std()
            outlier_mask = np.abs(values - mean_val) > 2 * std_val
        
        else:
            return None
        
        outlier_flws = self.flw_results[outlier_mask]
        
        return {
            'method': method,
            'metric': metric_column,
            'outlier_count': len(outlier_flws),
            'outlier_percentage': round(100 * len(outlier_flws) / len(self.flw_results), 1),
            'outlier_data': outlier_flws
        }