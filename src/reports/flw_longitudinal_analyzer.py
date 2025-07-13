"""
FLW Longitudinal Analyzer - Correlation and trend analysis for longitudinal data
Analyzes correlations between metrics at batch and FLW levels, plus quartile analysis

Save this file as: src/reports/flw_longitudinal_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class FLWLongitudinalAnalyzer:
    """Analyzer for correlations and trends in FLW longitudinal data"""
    
    def __init__(self, log_function=None):
        self.log = log_function if log_function else print
    
    def analyze_longitudinal_data(self, longitudinal_df):
        """
        Main analysis function that creates correlation matrices for batch and FLW levels
        
        Args:
            longitudinal_df: DataFrame with columns: flw_id, opportunity_name, flw_name, 
                           batch_number, batch_start_date, batch_end_date, total_visits_in_batch,
                           analysis_type, metric_name, metric_value, assessment_result, 
                           quality_score_name, quality_score_value
        
        Returns:
            dict: Dictionary with correlation matrices and quartile analyses
        """
        
        if len(longitudinal_df) == 0:
            self.log("No longitudinal data provided for correlation analysis")
            return {}
        
        self.log("Starting longitudinal correlation analysis...")
        
        results = {}
        
        # Tab 1: Batch-level correlations (numbered batches only)
        self.log("Creating batch-level correlation matrix...")
        batch_data = longitudinal_df[longitudinal_df['batch_number'] != 'all'].copy()
        if len(batch_data) > 0:
            batch_correlations = self._create_correlation_matrix(batch_data, "batch")
            if batch_correlations is not None:
                results['Batch Correlations'] = batch_correlations
                self.log(f"Batch correlation matrix created: {batch_correlations.shape[0]} metrics")
            else:
                self.log("No viable metrics found for batch-level correlations")
        else:
            self.log("No numbered batch data found for batch correlations")
        
        # Tab 2: FLW-level correlations (batch_number = 'all' only)
        self.log("Creating FLW-level correlation matrix...")
        flw_data = longitudinal_df[longitudinal_df['batch_number'] == 'all'].copy()
        if len(flw_data) > 0:
            flw_correlations = self._create_correlation_matrix(flw_data, "flw")
            if flw_correlations is not None:
                results['FLW Overall Correlations'] = flw_correlations
                self.log(f"FLW correlation matrix created: {flw_correlations.shape[0]} metrics")
            else:
                self.log("No viable metrics found for FLW-level correlations")
        else:
            self.log("No 'all' batch data found for FLW correlations")
        
        # Tabs 3+: Quartile analyses for specified metrics
        self.log("Creating quartile analyses...")
        quartile_metrics = [
            'diligence_score', 
            'median_visits_per_day',
            'female_child_ratio',
            'red_muac_percentage', 
            'under_12_months_percentage'
        ]
        
        if len(flw_data) > 0:
            quartile_results = self._create_quartile_analysis(flw_data, quartile_metrics)
            if quartile_results:
                results.update(quartile_results)
                self.log(f"Created {len(quartile_results)} quartile analysis tabs")
            else:
                self.log("No quartile analyses could be created")
        else:
            self.log("No FLW-level data available for quartile analysis")
        
        self.log(f"Longitudinal analysis complete. Created {len(results)} tabs total.")
        return results
    
    def _create_quartile_analysis(self, flw_data, quartile_metrics):
        """
        Create quartile analysis tabs for specified metrics
        
        Args:
            flw_data: FLW-level data (batch_number = 'all')
            quartile_metrics: List of metric names to create quartile analyses for
            
        Returns:
            dict: Dictionary with quartile analysis DataFrames
        """
        quartile_results = {}
        
        # First, convert FLW data to wide format to get all metrics per FLW
        wide_data = self._convert_flw_data_to_wide(flw_data)
        
        if wide_data is None or len(wide_data) == 0:
            self.log("No wide format data available for quartile analysis")
            return {}
        
        for quartile_metric in quartile_metrics:
            self.log(f"Creating quartile analysis for {quartile_metric}...")
            
            # Check if the metric exists and has sufficient data
            if quartile_metric not in wide_data.columns:
                self.log(f"  Metric {quartile_metric} not found in data")
                continue
                
            metric_data = wide_data[quartile_metric].dropna()
            if len(metric_data) < 4:  # Need at least 4 observations for quartiles
                self.log(f"  Insufficient data for {quartile_metric}: {len(metric_data)} observations")
                continue
            
            # Debug: Show the actual data distribution
            self.log(f"  {quartile_metric} data summary:")
            self.log(f"    Count: {len(metric_data)}")
            self.log(f"    Min: {metric_data.min()}, Max: {metric_data.max()}")
            self.log(f"    Unique values: {sorted(metric_data.unique())}")
            self.log(f"    Value counts: {dict(metric_data.value_counts().sort_index())}")
                
            # Create quartiles
            try:
                # First try standard qcut
                quartile_labels = pd.qcut(metric_data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], precision=2, duplicates='drop')
                quartile_ranges = pd.qcut(metric_data, q=4, precision=2, duplicates='drop')
                
                # Map quartile ranges for display
                range_mapping = {}
                for label, range_val in zip(quartile_labels.cat.categories, quartile_ranges.cat.categories):
                    range_mapping[label] = f"{range_val.left:.2f}-{range_val.right:.2f}"
                
            except ValueError as e:
                # If qcut fails due to duplicate edges, try different approaches
                self.log(f"  Standard quartiles failed for {quartile_metric}, trying alternatives: {str(e)}")
                
                try:
                    # Try with duplicates='drop' and see if we get fewer than 4 quartiles
                    quartile_ranges_temp = pd.qcut(metric_data, q=4, precision=2, duplicates='drop')
                    unique_quartiles = len(quartile_ranges_temp.cat.categories)
                    
                    if unique_quartiles < 4:
                        # Use custom percentile-based approach
                        self.log(f"  Only {unique_quartiles} unique quartiles possible, using percentile approach")
                        
                        # Calculate percentiles manually
                        percentiles = [0, 25, 50, 75, 100]
                        quartile_bounds = np.percentile(metric_data, percentiles)
                        
                        # Remove duplicate bounds and adjust labels accordingly
                        unique_bounds = np.unique(quartile_bounds)
                        num_bins = len(unique_bounds) - 1
                        
                        if num_bins < 2:
                            self.log(f"  Insufficient variation in {quartile_metric} for quartile analysis")
                            continue
                        
                        # Create appropriate labels for the number of bins we actually have
                        if num_bins == 2:
                            bin_labels = ['Q1-Q2', 'Q3-Q4']
                        elif num_bins == 3:
                            bin_labels = ['Q1', 'Q2-Q3', 'Q4']
                        else:
                            bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
                        
                        # Create labels based on actual bounds
                        quartile_labels = pd.cut(metric_data, bins=unique_bounds, labels=bin_labels, include_lowest=True)
                        
                        # Create range mapping
                        range_mapping = {}
                        for i, label in enumerate(bin_labels):
                            range_mapping[label] = f"{unique_bounds[i]:.2f}-{unique_bounds[i+1]:.2f}"
                    else:
                        # Standard approach worked with duplicates='drop'
                        quartile_labels = pd.qcut(metric_data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], precision=2, duplicates='drop')
                        
                        range_mapping = {}
                        for label, range_val in zip(quartile_labels.cat.categories, quartile_ranges_temp.cat.categories):
                            range_mapping[label] = f"{range_val.left:.2f}-{range_val.right:.2f}"
                
                except Exception as e2:
                    self.log(f"  All quartile methods failed for {quartile_metric}: {str(e2)}")
                    continue
            
            # Create quartile assignments for all FLWs with this metric
            wide_data_with_quartiles = wide_data.copy()
            wide_data_with_quartiles['quartile'] = quartile_labels
            
            # Calculate median for each metric within each quartile
            quartile_summary = []
            
            # Get all numeric columns (metrics)
            metric_columns = [col for col in wide_data.columns 
                            if col not in ['flw_id', 'opportunity_name', 'flw_name'] 
                            and wide_data[col].dtype in ['int64', 'float64']]
            
            # First row: Quartile ranges
            range_row = {'metric_name': 'Quartile Ranges'}
            
            # Add ranges for actual quartiles that exist
            actual_quartiles = quartile_labels.cat.categories.tolist()
            for quartile in actual_quartiles:
                range_row[quartile] = range_mapping.get(quartile, 'N/A')
            
            # Fill in missing standard quartile columns
            for standard_q in ['Q1', 'Q2', 'Q3', 'Q4']:
                if standard_q not in range_row:
                    range_row[standard_q] = 'N/A'
                    
            quartile_summary.append(range_row)
            
            # Second row: Number of FLWs per quartile
            count_row = {'metric_name': 'Number of FLWs'}
            
            # Count FLWs in each actual quartile
            for quartile in actual_quartiles:
                quartile_flws = wide_data_with_quartiles[
                    wide_data_with_quartiles['quartile'] == quartile
                ]
                count_row[quartile] = len(quartile_flws)
            
            # Fill in missing standard quartile columns
            for standard_q in ['Q1', 'Q2', 'Q3', 'Q4']:
                if standard_q not in count_row:
                    count_row[standard_q] = 0
                    
            quartile_summary.append(count_row)
            
            # Subsequent rows: Median values for each metric
            for metric_col in metric_columns:
                metric_row = {'metric_name': metric_col}
                
                # Get the actual quartile labels that exist
                actual_quartiles = quartile_labels.cat.categories.tolist()
                
                for quartile in actual_quartiles:
                    quartile_flws = wide_data_with_quartiles[
                        wide_data_with_quartiles['quartile'] == quartile
                    ]
                    
                    if len(quartile_flws) > 0:
                        metric_values = quartile_flws[metric_col].dropna()
                        if len(metric_values) > 0:
                            median_value = metric_values.median()
                            metric_row[quartile] = round(median_value, 3)
                        else:
                            metric_row[quartile] = None
                    else:
                        metric_row[quartile] = None
                
                # Fill in any missing standard quartile columns with None
                for standard_q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if standard_q not in metric_row:
                        metric_row[standard_q] = None
                
                quartile_summary.append(metric_row)
            
            # Convert to DataFrame
            quartile_df = pd.DataFrame(quartile_summary)
            
            # Add to results with descriptive tab name
            tab_name = f"Quartiles - {quartile_metric.replace('_', ' ').title()}"
            quartile_results[tab_name] = quartile_df
            
            self.log(f"  Created quartile analysis: {len(quartile_df)} metrics across 4 quartiles")
        
        return quartile_results
    
    def _convert_flw_data_to_wide(self, flw_data):
        """
        Convert FLW-level longitudinal data to wide format for quartile analysis
        
        Args:
            flw_data: FLW-level data (batch_number = 'all')
            
        Returns:
            DataFrame: Wide format with one row per FLW and columns for each metric
        """
        
        # Use similar logic to correlation matrix creation but simpler
        wide_rows = []
        
        # Group by flw_id, opportunity_name
        grouped = flw_data.groupby(['flw_id', 'opportunity_name'])
        
        for group_key, group_data in grouped:
            row = {
                'flw_id': group_key[0],
                'opportunity_name': group_key[1]
            }
            
            # Add flw_name if available
            flw_names = group_data['flw_name'].dropna()
            if len(flw_names) > 0:
                row['flw_name'] = flw_names.iloc[0]
            
            # Add each metric value
            for _, metric_row in group_data.iterrows():
                metric_name = metric_row['metric_name']
                if pd.notna(metric_name):
                    # Use metric_value if available, otherwise quality_score_value
                    if pd.notna(metric_row['metric_value']):
                        row[metric_name] = metric_row['metric_value']
                    elif pd.notna(metric_row['quality_score_value']):
                        # Use quality score name if available, otherwise metric_name
                        score_name = metric_row['quality_score_name']
                        column_name = score_name if pd.notna(score_name) else f"{metric_name}_quality"
                        row[column_name] = metric_row['quality_score_value']
            
            wide_rows.append(row)
        
        if not wide_rows:
            return None
            
        wide_df = pd.DataFrame(wide_rows)
        self.log(f"Converted FLW data to wide format: {len(wide_df)} FLWs, {len(wide_df.columns)} total columns")
        
        return wide_df
    
    def _create_correlation_matrix(self, data, level_name):
        """
        Create correlation matrix for the given data
        
        Args:
            data: Filtered DataFrame for this analysis level
            level_name: 'batch' or 'flw' for logging
            
        Returns:
            DataFrame: Correlation matrix with metrics as both rows and columns
        """
        
        # Get all unique metric names
        unique_metrics = data['metric_name'].dropna().unique()
        self.log(f"Found {len(unique_metrics)} unique metrics for {level_name} level: {list(unique_metrics)}")
        
        if len(unique_metrics) == 0:
            return None
        
        # Determine which column to use for each metric and build wide table
        viable_metrics = {}
        metric_data_info = []
        
        for metric_name in unique_metrics:
            metric_rows = data[data['metric_name'] == metric_name]
            
            # Check quality_score_value variation
            quality_values = metric_rows['quality_score_value'].dropna()
            quality_unique_count = len(quality_values.unique()) if len(quality_values) > 0 else 0
            
            # Check metric_value variation  
            metric_values = metric_rows['metric_value'].dropna()
            metric_unique_count = len(metric_values.unique()) if len(metric_values) > 0 else 0
            
            # Decide which column to use and what to call it
            if quality_unique_count > 1:
                # Use quality score and get the quality score name for labeling
                quality_score_names = metric_rows['quality_score_name'].dropna().unique()
                if len(quality_score_names) > 0:
                    column_name = quality_score_names[0]  # Use quality score name as column name
                else:
                    column_name = f"{metric_name}_quality"  # Fallback name
                
                viable_metrics[column_name] = {
                    'source_column': 'quality_score_value',
                    'metric_name': metric_name
                }
                data_info = f"quality_score ({quality_unique_count} unique values) -> {column_name}"
            elif metric_unique_count > 1:
                viable_metrics[metric_name] = {
                    'source_column': 'metric_value', 
                    'metric_name': metric_name
                }
                data_info = f"metric_value ({metric_unique_count} unique values)"
            else:
                # Drop - no variation
                data_info = f"dropped (no variation: quality={quality_unique_count}, metric={metric_unique_count})"
            
            metric_data_info.append({
                'metric_name': metric_name,
                'column_used': viable_metrics.get(metric_name, {}).get('source_column', 'none') if metric_name in viable_metrics else viable_metrics.get(list(viable_metrics.keys())[-1] if viable_metrics else '', {}).get('source_column', 'none'),
                'total_rows': len(metric_rows),
                'quality_unique': quality_unique_count,
                'metric_unique': metric_unique_count,
                'decision': data_info
            })
        
        self.log(f"Metric selection for {level_name} level:")
        for info in metric_data_info:
            self.log(f"  {info['metric_name']}: {info['decision']}")
        
        if len(viable_metrics) == 0:
            self.log(f"No metrics with variation found for {level_name} level")
            return None
        
        if len(viable_metrics) == 1:
            self.log(f"Only one viable metric found for {level_name} level - cannot create correlation matrix")
            return None
        
        # Create wide table for correlation analysis
        wide_data = self._pivot_to_wide_format(data, viable_metrics, level_name)
        
        if wide_data is None or len(wide_data) == 0:
            self.log(f"Failed to create wide format data for {level_name} level")
            return None
        
        # Calculate correlation matrix
        metric_columns = [col for col in wide_data.columns if col in viable_metrics.keys()]
        
        if len(metric_columns) < 2:
            self.log(f"Insufficient metric columns for correlation: {metric_columns}")
            return None
        
        correlation_data = wide_data[metric_columns]
        
        # Remove rows where all values are NaN
        correlation_data = correlation_data.dropna(how='all')
        
        if len(correlation_data) < 2:
            self.log(f"Insufficient data rows for correlation analysis: {len(correlation_data)}")
            return None
        
        # Calculate correlations
        correlation_matrix = correlation_data.corr()
        
        # Add sample size information
        sample_sizes = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                valid_pairs = correlation_data[[col1, col2]].dropna()
                sample_sizes.loc[col1, col2] = len(valid_pairs)
        
        # Create formatted output with correlation values and sample sizes
        formatted_matrix = self._format_correlation_matrix(correlation_matrix, sample_sizes)
        
        self.log(f"Created {level_name} correlation matrix: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        
        return formatted_matrix
    
    def _pivot_to_wide_format(self, data, viable_metrics, level_name):
        """
        Convert long format data to wide format for correlation analysis
        
        Args:
            data: Long format DataFrame
            viable_metrics: Dict mapping metric_name to column_name to use
            level_name: 'batch' or 'flw' for logging
            
        Returns:
            DataFrame: Wide format with one row per observation unit
        """
        
        wide_rows = []
        
        if level_name == "batch":
            # Group by flw_id, opportunity_name, batch_number
            groupby_cols = ['flw_id', 'opportunity_name', 'batch_number']
        else:
            # Group by flw_id, opportunity_name (batch_number should all be 'all')
            groupby_cols = ['flw_id', 'opportunity_name']
        
        grouped = data.groupby(groupby_cols)
        
        for group_key, group_data in grouped:
            # Create base row with grouping information
            if level_name == "batch":
                row = {
                    'flw_id': group_key[0],
                    'opportunity_name': group_key[1], 
                    'batch_number': group_key[2]
                }
            else:
                row = {
                    'flw_id': group_key[0],
                    'opportunity_name': group_key[1]
                }
            
            # Add flw_name if available
            flw_names = group_data['flw_name'].dropna()
            if len(flw_names) > 0:
                row['flw_name'] = flw_names.iloc[0]
            
            # Add each viable metric value
            for column_name, metric_info in viable_metrics.items():
                source_column = metric_info['source_column']
                source_metric_name = metric_info['metric_name']
                
                metric_rows = group_data[group_data['metric_name'] == source_metric_name]
                if len(metric_rows) > 0:
                    values = metric_rows[source_column].dropna()
                    if len(values) > 0:
                        # Use the first non-null value (should be consistent within group)
                        row[column_name] = values.iloc[0]
                    else:
                        row[column_name] = None
                else:
                    row[column_name] = None
            
            wide_rows.append(row)
        
        wide_df = pd.DataFrame(wide_rows)
        self.log(f"Created wide format data for {level_name}: {len(wide_df)} observations")
        
        return wide_df
    
    def _format_correlation_matrix(self, correlation_matrix, sample_sizes):
        """
        Format correlation matrix with additional information
        
        Args:
            correlation_matrix: Pandas correlation matrix
            sample_sizes: Matrix of sample sizes for each correlations
            
        Returns:
            DataFrame: Formatted correlation matrix with metric names as first column
        """
        
        # Round correlations to 3 decimal places
        formatted = correlation_matrix.round(3)
        
        # Reset index to make metric names a regular column
        formatted = formatted.reset_index()
        formatted = formatted.rename(columns={'index': 'metric_name'})
        
        # Add sample size information as a separate section
        # For now, just return the correlation matrix
        # Could enhance this later to include sample sizes, significance, etc.
        
        return formatted
