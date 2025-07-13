"""
Strong Negative Analyzer - Temporal analysis of strong_negative assessment patterns
Analyzes repeat offenders and sequential patterns in data quality issues

Save this file as: src/analyzers/strong_negative_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime


class StrongNegativeAnalyzer:
    """Analyzer for temporal patterns in strong_negative assessment results"""
    
    def __init__(self, log_function=None):
        self.log = log_function if log_function else print
    
    def analyze_strong_negative_patterns(self, longitudinal_df):
        """
        Main analysis function that creates temporal analysis for each metric with strong_negative results
        
        Args:
            longitudinal_df: DataFrame with longitudinal data including assessment_result column
        
        Returns:
            dict: Dictionary with one DataFrame per metric that has strong_negative cases
        """
        
        if len(longitudinal_df) == 0:
            self.log("No longitudinal data provided for strong negative analysis")
            return {}
        
        self.log("Starting strong negative temporal analysis...")
        
        # Filter to quality assessment data with numbered batches only
        quality_data = longitudinal_df[
            (longitudinal_df['analysis_type'] == 'quality_assessment') &
            (longitudinal_df['batch_number'] != 'all') &
            (longitudinal_df['assessment_result'].notna()) &
            (longitudinal_df['assessment_result'] != 'insufficient_data')
        ].copy()
        
        if len(quality_data) == 0:
            self.log("No quality assessment data found for strong negative analysis")
            return {}
        
        self.log(f"Filtered to {len(quality_data)} quality assessment records")
        
        # Find metrics that have strong_negative cases
        metrics_with_strong_negative = quality_data[
            quality_data['assessment_result'] == 'strong_negative'
        ]['metric_name'].unique()
        
        if len(metrics_with_strong_negative) == 0:
            self.log("No metrics found with strong_negative assessment results")
            return {}
        
        self.log(f"Found {len(metrics_with_strong_negative)} metrics with strong_negative cases: {list(metrics_with_strong_negative)}")
        
        results = {}
        
        # Analyze each metric separately
        for metric_name in metrics_with_strong_negative:
            self.log(f"Analyzing strong negative patterns for: {metric_name}")
            
            metric_analysis = self._analyze_metric_patterns(quality_data, metric_name)
            if metric_analysis is not None and len(metric_analysis) > 0:
                # Create a clean tab name
                tab_name = self._create_tab_name(metric_name)
                results[tab_name] = metric_analysis
                self.log(f"Created analysis for {tab_name}: {len(metric_analysis)} rows")
            else:
                self.log(f"No analysis results for {metric_name}")
        
        self.log(f"Strong negative analysis complete. Created {len(results)} metric analyses.")
        return results
    
    def _create_tab_name(self, metric_name):
        """Create a clean tab name from metric name"""
        # Convert metric names to readable tab names
        name_mapping = {
            'female_child_ratio': 'Gender Ratio Timeline',
            'red_muac_percentage': 'Red MUAC Timeline', 
            'under_12_months_percentage': 'Young Child Timeline'
        }
        
        return name_mapping.get(metric_name, f"{metric_name} Timeline")
    
    def _analyze_metric_patterns(self, quality_data, metric_name):
        """
        Analyze temporal patterns for a specific metric
        
        Args:
            quality_data: Filtered quality assessment data
            metric_name: Specific metric to analyze
            
        Returns:
            DataFrame: Analysis results with one row per opportunity + summary row
        """
        
        # Get data for this metric only
        metric_data = quality_data[quality_data['metric_name'] == metric_name].copy()
        
        if len(metric_data) == 0:
            return None
        
        # Convert batch_number to int for proper sorting
        metric_data['batch_number'] = pd.to_numeric(metric_data['batch_number'], errors='coerce')
        metric_data = metric_data.dropna(subset=['batch_number'])
        metric_data['batch_number'] = metric_data['batch_number'].astype(int)
        
        # Sort by FLW, opportunity, and batch number for temporal analysis
        metric_data = metric_data.sort_values(['flw_id', 'opportunity_name', 'batch_number'])
        
        # Get unique opportunities
        opportunities = metric_data['opportunity_name'].unique()
        results = []
        
        # Analyze each opportunity
        for opp_name in opportunities:
            opp_data = metric_data[metric_data['opportunity_name'] == opp_name]
            opp_analysis = self._analyze_opportunity_patterns(opp_data, opp_name)
            results.append(opp_analysis)
        
        # Create "All Opportunities" summary
        all_analysis = self._analyze_opportunity_patterns(metric_data, "All Opportunities")
        results.append(all_analysis)
        
        # Convert to DataFrame and sort by total_batches descending
        results_df = pd.DataFrame(results)
        
        # Sort by total_batches, but keep "All Opportunities" at the end
        all_opps_row = results_df[results_df['opportunity_name'] == "All Opportunities"]
        other_rows = results_df[results_df['opportunity_name'] != "All Opportunities"]
        other_rows = other_rows.sort_values('total_batches', ascending=False)
        
        # Combine with "All Opportunities" at the end
        final_results = pd.concat([other_rows, all_opps_row], ignore_index=True)
        
        return final_results
    
    def _analyze_opportunity_patterns(self, opp_data, opp_name):
        """
        Analyze patterns for a specific opportunity
        
        Args:
            opp_data: Data for this opportunity only
            opp_name: Opportunity name
            
        Returns:
            dict: Analysis results for this opportunity
        """
        
        # Basic counts
        total_flws = opp_data['flw_id'].nunique()
        total_visits = opp_data['total_visits_in_batch'].sum() if 'total_visits_in_batch' in opp_data.columns else len(opp_data)
        total_batches = len(opp_data)
        
        # Classify each batch as 'strong_negative' or 'ok'
        opp_data = opp_data.copy()
        opp_data['batch_status'] = opp_data['assessment_result'].apply(
            lambda x: 'strong_negative' if x == 'strong_negative' else 'ok'
        )
        
        # Analyze each FLW's pattern
        flw_patterns = {}
        flw_batch_classifications = {}
        
        for flw_id in opp_data['flw_id'].unique():
            flw_data = opp_data[opp_data['flw_id'] == flw_id].sort_values('batch_number')
            batch_statuses = flw_data['batch_status'].tolist()
            
            # Classify each batch for this FLW
            flw_batch_classifications[flw_id] = self._classify_flw_batches(batch_statuses)
        
        # Breakdown: by batch timing and status
        batch_counts = {
            'all_ok_flw_batches': 0,
            'ok_before_strong_negative': 0,
            'first_strong_negative': 0,
            'repeat_strong_negative': 0,
            'ok_after_strong_negative': 0
        }
        
        for flw_id in opp_data['flw_id'].unique():
            flw_data = opp_data[opp_data['flw_id'] == flw_id]
            flw_classifications = flw_batch_classifications[flw_id]
            
            for classification in flw_classifications:
                if classification in batch_counts:
                    batch_counts[classification] += 1
        
        # Calculate base percentages
        pct_all_ok_flw_batches = (batch_counts['all_ok_flw_batches'] / total_batches * 100) if total_batches > 0 else 0
        pct_ok_before_strong_negative = (batch_counts['ok_before_strong_negative'] / total_batches * 100) if total_batches > 0 else 0
        pct_first_strong_negative = (batch_counts['first_strong_negative'] / total_batches * 100) if total_batches > 0 else 0
        pct_repeat_strong_negative = (batch_counts['repeat_strong_negative'] / total_batches * 100) if total_batches > 0 else 0
        pct_ok_after_strong_negative = (batch_counts['ok_after_strong_negative'] / total_batches * 100) if total_batches > 0 else 0
        
        # Calculate computed convenience columns
        post_first_total_batches = batch_counts['repeat_strong_negative'] + batch_counts['ok_after_strong_negative']
        
        if post_first_total_batches > 0:
            computed_post_first_strong_negative_total = (post_first_total_batches / total_batches * 100)
            computed_strong_negative_after_strong_negative = (batch_counts['repeat_strong_negative'] / post_first_total_batches * 100)
            computed_ok_after_strong_negative = (batch_counts['ok_after_strong_negative'] / post_first_total_batches * 100)
        else:
            # Edge case: no post-first batches - leave computed columns blank
            computed_post_first_strong_negative_total = None
            computed_strong_negative_after_strong_negative = None
            computed_ok_after_strong_negative = None
        
        return {
            'opportunity_name': opp_name,
            'total_flws': total_flws,
            'total_visits': total_visits,
            'total_batches': total_batches,
            
            # Base breakdown: by batch timing/status
            'pct_all_ok_flw_batches': round(pct_all_ok_flw_batches, 1),
            'pct_ok_before_strong_negative': round(pct_ok_before_strong_negative, 1),
            'pct_first_strong_negative': round(pct_first_strong_negative, 1),
            'pct_repeat_strong_negative': round(pct_repeat_strong_negative, 1),
            'pct_ok_after_strong_negative': round(pct_ok_after_strong_negative, 1),
            
            # Computed convenience columns
            'computed_post_first_strong_negative_total': round(computed_post_first_strong_negative_total, 1) if computed_post_first_strong_negative_total is not None else None,
            'computed_strong_negative_after_strong_negative': round(computed_strong_negative_after_strong_negative, 1) if computed_strong_negative_after_strong_negative is not None else None,
            'computed_ok_after_strong_negative': round(computed_ok_after_strong_negative, 1) if computed_ok_after_strong_negative is not None else None,
            
            # Verification (should sum to 100%)
            'breakdown_total': round(pct_all_ok_flw_batches + pct_ok_before_strong_negative + 
                                   pct_first_strong_negative + pct_repeat_strong_negative + pct_ok_after_strong_negative, 1)
        }
    
    def _classify_flw_batches(self, batch_statuses):
        """
        Classify each batch for an FLW based on temporal patterns
        
        Args:
            batch_statuses: List of 'strong_negative' or 'ok' in chronological order
            
        Returns:
            List of classifications for each batch
        """
        
        if 'strong_negative' not in batch_statuses:
            # All OK FLW - all batches are 'all_ok_flw_batches'
            return ['all_ok_flw_batches'] * len(batch_statuses)
        
        # Find first strong_negative position
        first_strong_negative_idx = batch_statuses.index('strong_negative')
        
        classifications = []
        
        for i, status in enumerate(batch_statuses):
            if i < first_strong_negative_idx:
                # OK batch before any strong negative
                classifications.append('ok_before_strong_negative')
            elif i == first_strong_negative_idx:
                # This is the first strong_negative batch
                classifications.append('first_strong_negative')
            elif status == 'strong_negative':
                # Strong negative batch after the first one
                classifications.append('repeat_strong_negative')
            else:
                # OK batch after at least one strong negative (i > first_strong_negative_idx and status == 'ok')
                classifications.append('ok_after_strong_negative')
        
        return classifications
