"""
FLW Data Quality Assessment Report - Main Orchestrator with MUAC Tampering Experiment
Coordinates all analysis modules including new MUAC fraud detection validation

Save this file as: src/reports/flw_data_quality_report.py
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
from .base_report import BaseReport
from .utils.excel_exporter import ExcelExporter

# Import analysis modules
from .flw_assessment_framework import AssessmentEngine
from .flw_metrics_engine import FLWMetricsEngine
from .flw_age_analyzer import FLWAgeAnalyzer
from .flw_muac_analyzer_enhanced import EnhancedFLWMUACAnalyzer
from .flw_longitudinal_analyzer import FLWLongitudinalAnalyzer
from .analyzers.strong_negative_analyzer import StrongNegativeAnalyzer

class FLWDataQualityReport(BaseReport):
    """FLW Data Quality Assessment Report for identifying concerning patterns and data quality issues"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for FLW Data Quality Assessment report"""
        
        # Batching parameters
        ttk.Label(parent_frame, text="Batch size (max visits per batch):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        batch_size_var = tk.StringVar(value="300")
        ttk.Entry(parent_frame, textvariable=batch_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(parent_frame, text="Min batch size (min visits):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        min_size_var = tk.StringVar(value="100")
        ttk.Entry(parent_frame, textvariable=min_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Data quality checks selection
        ttk.Label(parent_frame, text="Data quality checks:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Checkboxes for each assessment type
        assessments_frame = ttk.Frame(parent_frame)
        assessments_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        gender_var = tk.BooleanVar(value=True)
        muac_var = tk.BooleanVar(value=True)
        age_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(assessments_frame, text="Gender Balance", variable=gender_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(assessments_frame, text="MUAC Data Quality", variable=muac_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(assessments_frame, text="Age Distribution", variable=age_var).grid(row=0, column=2, sticky=tk.W)
        
        # Distribution analysis options
        ttk.Label(parent_frame, text="Distribution analysis:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        distribution_frame = ttk.Frame(parent_frame)
        distribution_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        flw_ages_var = tk.BooleanVar(value=True)
        opps_ages_var = tk.BooleanVar(value=True)
        flw_muac_var = tk.BooleanVar(value=True)
        opps_muac_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(distribution_frame, text="FLW Ages", variable=flw_ages_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="Opps Ages", variable=opps_ages_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="FLW MUAC", variable=flw_muac_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="Opps MUAC", variable=opps_muac_var).grid(row=1, column=1, sticky=tk.W)
        
        # Gender timeline analysis
        ttk.Label(parent_frame, text="Gender timeline:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        
        timeline_frame = ttk.Frame(parent_frame)
        timeline_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        include_gender_timeline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(timeline_frame, text="Generate weekly gender timeline per FLW", variable=include_gender_timeline_var).grid(row=0, column=0, sticky=tk.W)
        
        # MUAC Tampering Experiment
        ttk.Label(parent_frame, text="MUAC fraud validation:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        
        tampering_frame = ttk.Frame(parent_frame)
        tampering_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        include_tampering_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tampering_frame, text="Include tampering experiment (validates fraud detection)", variable=include_tampering_var).grid(row=0, column=0, sticky=tk.W)
        
        # Correlation analysis options
        ttk.Label(parent_frame, text="Correlation analysis:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        
        correlation_frame = ttk.Frame(parent_frame)
        correlation_frame.grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)
        
        correlation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(correlation_frame, text="Generate correlation matrices", variable=correlation_var).grid(row=0, column=0, sticky=tk.W)
        
        # Output options
        ttk.Label(parent_frame, text="Output options:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        
        output_frame = ttk.Frame(parent_frame)
        output_frame.grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)
        
        export_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Export unified longitudinal CSV", variable=export_csv_var).grid(row=0, column=0, sticky=tk.W)
        
        # Store variables for access
        parent_frame.batch_size_var = batch_size_var
        parent_frame.min_size_var = min_size_var
        parent_frame.gender_var = gender_var
        parent_frame.muac_var = muac_var
        parent_frame.age_var = age_var
        parent_frame.flw_ages_var = flw_ages_var
        parent_frame.opps_ages_var = opps_ages_var
        parent_frame.flw_muac_var = flw_muac_var
        parent_frame.opps_muac_var = opps_muac_var
        parent_frame.include_gender_timeline_var = include_gender_timeline_var
        parent_frame.include_tampering_var = include_tampering_var
        parent_frame.correlation_var = correlation_var
        parent_frame.export_csv_var = export_csv_var
    
    def _prepare_data_for_engines(self, batch_size, min_size):
        """Prepare data once for both quality assessments and performance metrics"""
        
        # Auto-detect columns using AssessmentEngine patterns
        engine = AssessmentEngine(batch_size=batch_size, min_size=min_size)
        detected_columns, missing_columns = engine.auto_detect_columns(self.df)
        
        if missing_columns:
            self.log(f"Warning: Missing columns: {missing_columns}")
        
        self.log(f"Detected columns: {detected_columns}")
        
        # Also preserve diligence columns for metrics engine
        diligence_columns = [col for col in self.df.columns if col.lower().startswith('diligence')]
        if diligence_columns:
            self.log(f"Found {len(diligence_columns)} diligence columns: {diligence_columns}")
            # Add diligence columns to detected_columns so they survive the prepare_data process
            for col in diligence_columns:
                detected_columns[col] = col  # Map column to itself (no renaming needed)
        else:
            self.log("No diligence columns found in dataset")
        
        # Also preserve health outcome columns for metrics engine
        health_patterns = {
            'under_treatment_for_mal': ['under_treatment_for_mal', 'under_malnutrition_treatment', 'malnutrition_treatment'],
            'diarrhea_last_month': ['diarrhea_last_month', 'diarrhea_recent', 'recent_diarrhea']
        }
        
        health_columns_found = []
        for field, patterns in health_patterns.items():
            found = False
            for pattern in patterns:
                matching_cols = [col for col in self.df.columns if pattern.lower() in col.lower()]
                if matching_cols:
                    detected_columns[field] = matching_cols[0]  # Use original column name
                    health_columns_found.append(matching_cols[0])
                    found = True
                    break
        
        if health_columns_found:
            self.log(f"Found {len(health_columns_found)} health outcome columns: {health_columns_found}")
        else:
            self.log("No health outcome columns found in dataset")
        
        # Prepare data using AssessmentEngine logic
        df_clean = engine.prepare_data(self.df, detected_columns)
        
        # Add GPS columns for metrics engine if available
        gps_patterns = {
            'latitude': ['latitude', 'lat', 'visit_latitude', 'visit_lat'],
            'longitude': ['longitude', 'lng', 'lon', 'long', 'visit_longitude', 'visit_lng']
        }
        
        for field, patterns in gps_patterns.items():
            for pattern in patterns:
                matching_cols = [col for col in self.df.columns if pattern.lower() in col.lower()]
                if matching_cols and field not in df_clean.columns:
                    df_clean[field] = self.df[matching_cols[0]]
                    break
        
        return df_clean
    
    def _create_gender_timeline_analysis(self, df_clean, min_size):
        """
        Create gender timeline analysis showing weekly female percentages per FLW
        
        Args:
            df_clean: Cleaned dataframe with standardized columns
            min_size: Minimum visits required per week to show percentage
            
        Returns:
            pd.DataFrame: Timeline with one row per FLW, one column per week
        """
        
        # Debug: Show available columns
        self.log(f"Available columns in df_clean: {list(df_clean.columns)}")
        
        # Try to find gender column with multiple possible names
        gender_col = None
        gender_patterns = [
            'child_gender', 'gender', 'child_sex', 'sex', 
            'beneficiary_gender', 'beneficiary_sex', 'patient_gender', 'patient_sex'
        ]
        
        for pattern in gender_patterns:
            matching_cols = [col for col in df_clean.columns if pattern.lower() in col.lower()]
            if matching_cols:
                gender_col = matching_cols[0]
                self.log(f"Found gender column: {gender_col}")
                break
        
        if gender_col is None:
            # Fallback: try to get it from original dataframe if not in cleaned version
            self.log(f"No gender column found in cleaned data. Checking original dataframe...")
            if 'child_gender' in self.df.columns:
                # Add the original gender column to df_clean
                df_clean = df_clean.copy()
                df_clean['child_gender'] = self.df['child_gender']
                gender_col = 'child_gender'
                self.log(f"Added gender column from original data: {gender_col}")
            else:
                self.log(f"No gender column found. Tried patterns: {gender_patterns}")
                self.log(f"Available columns in original df: {list(self.df.columns)}")
                return pd.DataFrame()
        
        # Check other required columns
        required_cols = ['flw_id', 'flw_name', 'opportunity_name', 'visit_date']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            self.log(f"Warning: Missing columns for gender timeline: {missing_cols}")
            return pd.DataFrame()
        
        # Filter to records with valid gender data
        gender_data = df_clean[
            df_clean[gender_col].notna() & 
            df_clean['visit_date'].notna() &
            df_clean['flw_id'].notna()
        ].copy()
        
        if len(gender_data) == 0:
            self.log("No valid gender and date data found for timeline analysis")
            return pd.DataFrame()
        
        self.log(f"Creating gender timeline analysis with {len(gender_data)} valid records")
        self.log(f"Sample gender values: {gender_data[gender_col].value_counts().head()}")
        
        # Ensure visit_date is datetime
        gender_data['visit_date'] = pd.to_datetime(gender_data['visit_date'])
        
        # Create week column (Monday start)
        gender_data['week_start'] = gender_data['visit_date'].dt.to_period('W-MON').dt.start_time
        gender_data['week_label'] = gender_data['week_start'].dt.strftime('Week_%Y-%m-%d')
        
        # Calculate female indicator (handle both 'female_child'/'male_child' and 'f'/'m' formats)
        gender_values = gender_data[gender_col].str.lower()
        gender_data['is_female'] = gender_values.isin(['f', 'female', '1', 'female_child']).astype(int)
        
        # Debug gender detection
        female_count = gender_data['is_female'].sum()
        total_count = len(gender_data)
        self.log(f"Gender detection: {female_count} female out of {total_count} total ({female_count/total_count*100:.1f}% female)")
        
        # Get date range for all weeks
        min_date = gender_data['week_start'].min()
        max_date = gender_data['week_start'].max()
        
        # Create complete week range
        week_range = pd.date_range(start=min_date, end=max_date, freq='W-MON')
        week_labels = [f"Week_{date.strftime('%Y-%m-%d')}" for date in week_range]
        
        self.log(f"Timeline covers {len(week_labels)} weeks from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Group by FLW and week, calculate stats
        flw_week_stats = gender_data.groupby(['flw_id', 'flw_name', 'opportunity_name', 'week_label']).agg({
            'is_female': ['count', 'sum'],
            gender_col: 'count'  # Use the detected gender column name
        }).reset_index()
        
        # Flatten column names
        flw_week_stats.columns = ['flw_id', 'flw_name', 'opportunity_name', 'week_label', 'total_visits', 'female_count', 'total_children']
        
        # Calculate female percentage
        flw_week_stats['female_pct'] = (flw_week_stats['female_count'] / flw_week_stats['total_visits'] * 100).round(1)
        
        # Only keep weeks with sufficient data
        sufficient_data = flw_week_stats[flw_week_stats['total_visits'] >= min_size].copy()
        
        if len(sufficient_data) == 0:
            self.log(f"No FLW/week combinations meet minimum visit threshold of {min_size}")
            return pd.DataFrame()
        
        self.log(f"Found {len(sufficient_data)} FLW/week combinations with >= {min_size} visits")
        
        # Calculate population baseline for red score calculation
        population_female_count = gender_data['is_female'].sum()
        population_total = len(gender_data)
        population_ratio = population_female_count / population_total if population_total > 0 else 0.5
        
        def is_red_score(female_count, total_visits, population_baseline=population_ratio):
            """Determine if this is a red score using same logic as gender assessment"""
            if total_visits < min_size:
                return False
            
            # Calculate sample ratio
            sample_ratio = female_count / total_visits
            
            # Use population baseline for comparison
            expected = population_baseline
            
            # Calculate z-score for significant deviation
            if expected == 0 or expected == 1:
                return False
            
            variance = expected * (1 - expected) / total_visits
            std_error = np.sqrt(variance)
            z_score = abs(sample_ratio - expected) / std_error if std_error > 0 else 0
            
            # Red score if z-score > 2.576 (99% confidence)
            return z_score > 2.576
        
        sufficient_data['is_red_score'] = sufficient_data.apply(
            lambda row: is_red_score(row['female_count'], row['total_visits']), 
            axis=1
        )
        
        # Create pivot table
        pivot_data = sufficient_data.pivot_table(
            index=['flw_id', 'flw_name', 'opportunity_name'],
            columns='week_label',
            values='female_pct',
            fill_value=np.nan
        ).reset_index()
        
        # Create red score pivot table for color coding reference
        red_score_pivot = sufficient_data.pivot_table(
            index=['flw_id', 'flw_name', 'opportunity_name'],
            columns='week_label', 
            values='is_red_score',
            fill_value=False
        ).reset_index()
        
        # Ensure all weeks are present as columns (even if no data)
        for week_label in week_labels:
            if week_label not in pivot_data.columns:
                pivot_data[week_label] = np.nan
                red_score_pivot[week_label] = False
        
        # Sort columns: identifier columns first, then weeks chronologically
        id_columns = ['flw_id', 'flw_name', 'opportunity_name']
        week_columns = sorted([col for col in pivot_data.columns if col.startswith('Week_')])
        pivot_data = pivot_data[id_columns + week_columns]
        red_score_pivot = red_score_pivot[id_columns + week_columns]
        
        # Add summary statistics
        week_cols = [col for col in pivot_data.columns if col.startswith('Week_')]
        
        pivot_data['total_weeks_with_data'] = pivot_data[week_cols].notna().sum(axis=1)
        pivot_data['avg_female_pct'] = pivot_data[week_cols].mean(axis=1).round(1)
        pivot_data['red_score_weeks'] = red_score_pivot[week_cols].sum(axis=1)
        pivot_data['red_score_rate'] = (pivot_data['red_score_weeks'] / pivot_data['total_weeks_with_data'] * 100).round(1)
        
        # Sort by total weeks with data (descending) then by FLW name
        pivot_data = pivot_data.sort_values(['total_weeks_with_data', 'flw_name'], ascending=[False, True])
        
        self.log(f"Created gender timeline with {len(pivot_data)} FLWs across {len(week_columns)} weeks")
        self.log(f"Population baseline female ratio: {population_ratio:.3f}")
        
        # Store red score data for Excel formatting
        pivot_data._red_score_data = red_score_pivot
        pivot_data._gender_timeline_formatting = True  # Flag for Excel exporter
        
        return pivot_data

    def _apply_muac_tampering(self, df, tamper_pct=20, noise_range_min=2, noise_range_max=6, seed=42):
        """Apply tampering to MUAC data for fraud detection validation"""
        
        np.random.seed(seed)
        df_tampered = df.copy()
        muac_col = 'muac_measurement_cm'
        
        if muac_col not in df_tampered.columns:
            self.log(f"Warning: MUAC column '{muac_col}' not found for tampering")
            return df_tampered
        
        # Get valid MUAC measurements
        valid_muac_mask = df_tampered[muac_col].notna()
        valid_muac_indices = df_tampered[valid_muac_mask].index
        
        if len(valid_muac_indices) == 0:
            self.log("Warning: No valid MUAC measurements found for tampering")
            return df_tampered
        
        # Select random subset to tamper (20%)
        tamper_count = int(len(valid_muac_indices) * (tamper_pct / 100))
        tamper_indices = np.random.choice(valid_muac_indices, size=tamper_count, replace=False)
        
        # Add aggressive noise (between +2cm and +6cm only - always positive to break patterns)
        noise = np.random.uniform(noise_range_min, noise_range_max, size=tamper_count)
        df_tampered.loc[tamper_indices, muac_col] += noise
        
        self.log(f"Applied extreme tampering to {tamper_count} MUAC measurements ({tamper_pct}% of {len(valid_muac_indices)} valid measurements)")
        self.log(f"Noise range: +{noise_range_min}cm to +{noise_range_max}cm (always positive), Random seed: {seed}")
        
        return df_tampered
    
    def _create_tampering_summary(self, original_df, tampered_df):
        """Create summary comparing original vs tampered MUAC analysis results"""
        
        if original_df is None or tampered_df is None:
            return pd.DataFrame()
        
        # Filter to sufficient data cases only for fair comparison
        original_sufficient = original_df[original_df['data_sufficiency'] == 'SUFFICIENT'] if 'data_sufficiency' in original_df.columns else original_df
        tampered_sufficient = tampered_df[tampered_df['data_sufficiency'] == 'SUFFICIENT'] if 'data_sufficiency' in tampered_df.columns else tampered_df
        
        if len(original_sufficient) == 0 or len(tampered_sufficient) == 0:
            return pd.DataFrame([{
                'Metric': 'Error',
                'Original': 'No sufficient data',
                'Tampered': 'No sufficient data',
                'Change': 'N/A',
                'Description': 'Insufficient data for comparison'
            }])
        
        summary_data = []
        
        # Helper function to calculate percentage change
        def pct_change(original_val, tampered_val):
            if original_val == 0 and tampered_val == 0:
                return "0.0%"
            elif original_val == 0:
                return "+8%"
            else:
                change = ((tampered_val - original_val) / original_val) * 100
                return f"{change:+.1f}%"
        
        # Sample sizes
        summary_data.append({
            'Metric': 'Sample Size (Sufficient Data)',
            'Original': len(original_sufficient),
            'Tampered': len(tampered_sufficient),
            'Change': 'N/A',
            'Description': 'FLWs with sufficient data for analysis'
        })
        
        # Key boolean flags (percentage with FALSE values - these should increase with tampering)
        boolean_metrics = [
            ('increasing_to_peak_with_wiggle', 'Increasing to Peak BROKEN'),
            ('decreasing_from_peak_with_wiggle', 'Decreasing from Peak BROKEN')
        ]
        
        for col, label in boolean_metrics:
            if col in original_sufficient.columns and col in tampered_sufficient.columns:
                # Count FALSE values (broken patterns)
                orig_false_count = (~original_sufficient[col]).sum() if original_sufficient[col].dtype == bool else (original_sufficient[col] == False).sum()
                tamp_false_count = (~tampered_sufficient[col]).sum() if tampered_sufficient[col].dtype == bool else (tampered_sufficient[col] == False).sum()
                
                orig_false_pct = (orig_false_count / len(original_sufficient)) * 100
                tamp_false_pct = (tamp_false_count / len(tampered_sufficient)) * 100
                
                summary_data.append({
                    'Metric': f'{label} (%)',
                    'Original': f"{orig_false_pct:.1f}%",
                    'Tampered': f"{tamp_false_pct:.1f}%",
                    'Change': f"{tamp_false_pct - orig_false_pct:+.1f}pp",
                    'Description': f'Percentage with {col} = FALSE (broken authentic patterns)'
                })
        
        # Bins with data (percentage with < 4 bins - should increase with tampering)
        if 'bins_with_data' in original_sufficient.columns and 'bins_with_data' in tampered_sufficient.columns:
            orig_few_bins = (original_sufficient['bins_with_data'] < 4).sum()
            tamp_few_bins = (tampered_sufficient['bins_with_data'] < 4).sum()
            
            orig_few_bins_pct = (orig_few_bins / len(original_sufficient)) * 100
            tamp_few_bins_pct = (tamp_few_bins / len(tampered_sufficient)) * 100
            
            summary_data.append({
                'Metric': 'Few Bins with Data (< 4) (%)',
                'Original': f"{orig_few_bins_pct:.1f}%",
                'Tampered': f"{tamp_few_bins_pct:.1f}%",
                'Change': f"{tamp_few_bins_pct - orig_few_bins_pct:+.1f}pp",
                'Description': 'Percentage with fewer than 4 bins containing data'
            })
        
        # Authenticity scores
        if 'authenticity_score' in original_sufficient.columns and 'authenticity_score' in tampered_sufficient.columns:
            orig_avg_score = original_sufficient['authenticity_score'].mean()
            tamp_avg_score = tampered_sufficient['authenticity_score'].mean()
            
            summary_data.append({
                'Metric': 'Average Authenticity Score',
                'Original': f"{orig_avg_score:.2f}",
                'Tampered': f"{tamp_avg_score:.2f}",
                'Change': f"{tamp_avg_score - orig_avg_score:+.2f}",
                'Description': 'Average authenticity score (0-10 scale)'
            })
            
            # Assessment categories
            assessments = [
                ('HIGHLY AUTHENTIC', lambda df: (df['authenticity_score'] >= 8).sum()),
                ('PROBABLY AUTHENTIC', lambda df: ((df['authenticity_score'] >= 6) & (df['authenticity_score'] < 8)).sum()),
                ('SUSPICIOUS', lambda df: ((df['authenticity_score'] >= 4) & (df['authenticity_score'] < 6)).sum()),
                ('LIKELY FABRICATED', lambda df: (df['authenticity_score'] < 4).sum())
            ]
            
            for assessment_name, count_func in assessments:
                orig_count = count_func(original_sufficient)
                tamp_count = count_func(tampered_sufficient)
                
                orig_pct = (orig_count / len(original_sufficient)) * 100
                tamp_pct = (tamp_count / len(tampered_sufficient)) * 100
                
                summary_data.append({
                    'Metric': f'{assessment_name} (%)',
                    'Original': f"{orig_pct:.1f}%",
                    'Tampered': f"{tamp_pct:.1f}%",
                    'Change': f"{tamp_pct - orig_pct:+.1f}pp",
                    'Description': f'Percentage classified as {assessment_name.lower()}'
                })
        
        # Problematic flag
        if 'flag_problematic' in original_sufficient.columns and 'flag_problematic' in tampered_sufficient.columns:
            orig_prob_count = original_sufficient['flag_problematic'].sum()
            tamp_prob_count = tampered_sufficient['flag_problematic'].sum()
            
            orig_prob_pct = (orig_prob_count / len(original_sufficient)) * 100
            tamp_prob_pct = (tamp_prob_count / len(tampered_sufficient)) * 100
            
            summary_data.append({
                'Metric': 'Flagged as Problematic (%)',
                'Original': f"{orig_prob_pct:.1f}%",
                'Tampered': f"{tamp_prob_pct:.1f}%",
                'Change': f"{tamp_prob_pct - orig_prob_pct:+.1f}pp",
                'Description': 'Percentage flagged with composite quality issues'
            })
        
        return pd.DataFrame(summary_data)
        
    def generate(self):
        """Generate FLW data quality assessment reports with batching"""
        output_files = []
        excel_data = {}
        from datetime import datetime 
        
        # Get parameters
        batch_size = int(self.get_parameter_value('batch_size', '300'))
        min_size = int(self.get_parameter_value('min_size', '100'))
        export_csv = self.get_parameter_value('export_csv', True)
        include_tampering = self.get_parameter_value('include_tampering', True)
        
        # Get selected assessment types
        selected_assessments = []
        if self.get_parameter_value('gender', True):
            selected_assessments.append('gender_ratio')
        if self.get_parameter_value('muac', True):
            selected_assessments.append('low_red_muac')
        if self.get_parameter_value('age', True):
            selected_assessments.append('low_young_child')
        
        # Get distribution analysis options
        include_flw_ages = self.get_parameter_value('flw_ages', True)
        include_opps_ages = self.get_parameter_value('opps_ages', True)
        include_flw_muac = self.get_parameter_value('flw_muac', True)
        include_opps_muac = self.get_parameter_value('opps_muac', True)
        
        # Get gender timeline option
        include_gender_timeline = self.get_parameter_value('include_gender_timeline', True)
        self.log(f"DEBUG SELF-log: include_gender_timeline = {include_gender_timeline}")
        print(f"DEBUG PRINT: include_gender_timeline = {include_gender_timeline}")  # Use print() instead of self.log()
        
        # Get correlation analysis option
        include_correlations = self.get_parameter_value('correlation', True)
        
        if not selected_assessments and not any([include_flw_ages, include_opps_ages, include_flw_muac, include_opps_muac]) and not export_csv and not include_correlations and not include_gender_timeline:
            self.log("Error: No data quality checks, distribution analysis, correlation analysis, gender timeline, or CSV export selected")
            return output_files
        
        self.log(f"Starting FLW data quality assessment with batching")
        self.log(f"Batch size: {batch_size}, Min size: {min_size}")
        if selected_assessments:
            self.log(f"Selected quality checks: {', '.join(selected_assessments)}")
        
        distribution_types = []
        if include_flw_ages: distribution_types.append("FLW Ages")
        if include_opps_ages: distribution_types.append("Opps Ages") 
        if include_flw_muac: distribution_types.append("FLW MUAC")
        if include_opps_muac: distribution_types.append("Opps MUAC")
        
        if distribution_types:
            self.log(f"Distribution analysis: {', '.join(distribution_types)}")
        
        if include_gender_timeline:
            self.log("Gender timeline analysis: Enabled")
        
        if include_correlations:
            self.log("Correlation analysis: Enabled")
            
        if include_tampering and (include_flw_muac or include_opps_muac):
            self.log("MUAC tampering experiment: Enabled (20% tampering, +2cm to +6cm noise)")
        
        try:
            # Initialize variables for unified CSV
            all_csv_data = []
            
            # Prepare data once for both engines (single source of truth)
            print("Preparing data for all analyses...")
            df_clean = self._prepare_data_for_engines(batch_size, min_size)
            
            # Run main quality assessments if any selected
            if selected_assessments:
                # Create assessment engine with batching parameters
                engine = AssessmentEngine(batch_size=batch_size, min_size=min_size)
                
                # Filter to selected assessments only
                engine.assessments = {k: v for k, v in engine.assessments.items() if k in selected_assessments}
                
                self.log("Running batch-based data quality assessment pipeline...")
                excel_results, all_batch_results, population_stats = engine.run_assessments(df_clean)
                
                if len(excel_results) > 0:
                    # Create opportunity summary using Excel results (most recent batches)
                    self.log("Creating opportunity summary...")
                    opportunity_summary = engine.create_opportunity_summary(excel_results, self.df)
                    
                    # Create assessment summary statistics
                    self.log("Creating assessment summary...")
                    assessment_summary = self._create_assessment_summary(excel_results, population_stats, batch_size, min_size)
                    
                    # Add to Excel data (using most recent batch results)
                    excel_data['FLW Results'] = excel_results
                    excel_data['Opportunity Summary'] = opportunity_summary
                    excel_data['Assessment Summary'] = assessment_summary
                    
                    # Create flagged FLWs summary (only strong negatives from most recent batches)
                    flagged_flws = excel_results[excel_results['has_any_strong_negative']]
                    if len(flagged_flws) > 0:
                        excel_data['Quality Issues Found'] = flagged_flws
                        self.log(f"Found {len(flagged_flws)} FLWs with data quality issues in most recent batches")
                    
                    # Create insufficient data summary
                    insufficient_data_flws = excel_results[excel_results['has_insufficient_data']]
                    if len(insufficient_data_flws) > 0:
                        excel_data['Insufficient Data'] = insufficient_data_flws
                        self.log(f"Found {len(insufficient_data_flws)} FLWs with insufficient data for assessment")
                    
                    # Add quality assessment data to CSV if requested
                    if export_csv and len(all_batch_results) > 0:
                        self.log("Transforming quality assessment results to CSV format...")
                        csv_results = engine.transform_to_csv_format(all_batch_results)
                        
                        if len(csv_results) > 0:
                            all_csv_data.append(csv_results)
                            self.log(f"Added {len(csv_results)} quality assessment records to unified CSV")
                        else:
                            self.log("No quality assessment data available for CSV transformation")
                else:
                    self.log("No FLW/opportunity pairs met the minimum batch size threshold")
            
            # Run FLW performance metrics analysis if CSV export requested
            if export_csv:
                self.log("Running FLW performance metrics analysis...")
                metrics_engine = FLWMetricsEngine(batch_size=batch_size, min_size=min_size)
                metrics_results = metrics_engine.run_metrics_analysis(df_clean)
                
                if len(metrics_results) > 0:
                    all_csv_data.append(metrics_results)
                    self.log(f"Added {len(metrics_results)} performance metric records to unified CSV")
                else:
                    self.log("No performance metrics data available for export")
            
            # Export unified CSV if we have any data
            if export_csv and all_csv_data:
                self.log("Creating unified longitudinal CSV...")
                unified_csv = pd.concat(all_csv_data, ignore_index=True)
                
                # Ensure consistent column order
                standard_columns = [
                    'flw_id',
                    'opportunity_name', 
                    'flw_name',
                    'batch_number',
                    'batch_start_date',
                    'batch_end_date',
                    'total_visits_in_batch',
                    'analysis_type',
                    'metric_name',
                    'metric_value',
                    'assessment_result',
                    'quality_score_name',
                    'quality_score_value'
                ]
                
                # Include only columns that exist
                existing_columns = [col for col in standard_columns if col in unified_csv.columns]
                unified_csv = unified_csv[existing_columns]
                
                csv_file = os.path.join(
                    self.output_dir, 
                    f"flw_longitudinal_data_{datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
                )
                unified_csv.to_csv(csv_file, index=False)
                output_files.append(csv_file)
                self.log(f"Created unified CSV: {os.path.basename(csv_file)} ({len(unified_csv)} total records)")
                
                # Run correlation analysis if requested and we have data
                if include_correlations:
                    self.log("Running correlation analysis on longitudinal data...")
                    longitudinal_analyzer = FLWLongitudinalAnalyzer(self.log)
                    correlation_results = longitudinal_analyzer.analyze_longitudinal_data(unified_csv)
                    
                    # Run strong negative temporal analysis
                    self.log("Running strong negative temporal analysis...")
                    strong_negative_analyzer = StrongNegativeAnalyzer(self.log)
                    strong_negative_results = strong_negative_analyzer.analyze_strong_negative_patterns(unified_csv)
                    
                    # Combine correlation and strong negative results
                    combined_results = {}
                    if correlation_results:
                        combined_results.update(correlation_results)
                    if strong_negative_results:
                        combined_results.update(strong_negative_results)
                    
                    if combined_results:
                        # Create separate Excel file for all longitudinal analyses
                        longitudinal_excel_file = os.path.join(
                            self.output_dir, 
                            f"flw_longitudinal_analysis_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx"
                        )
                        
                        excel_exporter = ExcelExporter(self.log)
                        longitudinal_file = excel_exporter.export_to_excel(
                            combined_results,
                            longitudinal_excel_file
                        )
                        
                        if longitudinal_file:
                            output_files.append(longitudinal_file)
                            correlation_tabs = len(correlation_results) if correlation_results else 0
                            strong_negative_tabs = len(strong_negative_results) if strong_negative_results else 0
                            self.log(f"Created longitudinal analysis Excel file: {os.path.basename(longitudinal_file)}")
                            self.log(f"  - {correlation_tabs} correlation tabs, {strong_negative_tabs} strong negative timeline tabs")
                        else:
                            self.log("Failed to create longitudinal analysis Excel file")
                    else:
                        self.log("No longitudinal analysis data generated")
                        
            elif export_csv:
                self.log("No data available for unified CSV export")
            
            # Generate age distribution analysis
            if include_flw_ages or include_opps_ages:
                self.log("Creating age distribution analysis...")
                age_analyzer = FLWAgeAnalyzer(self.df, self.log)
                
                if include_flw_ages:
                    flw_ages_df = age_analyzer.create_flw_ages_analysis()
                    if flw_ages_df is not None and len(flw_ages_df) > 0:
                        excel_data['FLW Ages'] = flw_ages_df
                
                if include_opps_ages:
                    opps_ages_df = age_analyzer.create_opps_ages_analysis()
                    if opps_ages_df is not None and len(opps_ages_df) > 0:
                        excel_data['Opps Ages'] = opps_ages_df
            
            # Generate gender timeline analysis
            if include_gender_timeline:
                self.log("Creating gender timeline analysis...")
                gender_timeline_df = self._create_gender_timeline_analysis(df_clean, min_size)
                if gender_timeline_df is not None and len(gender_timeline_df) > 0:
                    excel_data['Gender Timeline'] = gender_timeline_df
                    self.log(f"Created gender timeline with {len(gender_timeline_df)} FLWs")
                else:
                    self.log("No gender timeline data generated")
            
            # Generate MUAC distribution analysis with fraud detection
            if include_flw_muac or include_opps_muac:
                self.log("Creating MUAC distribution analysis with fraud detection...")
                muac_analyzer = EnhancedFLWMUACAnalyzer(self.df, self.log)
                
                flw_muac_original = None
                opps_muac_original = None
                flw_muac_tampered = None
                opps_muac_tampered = None
                
                # Run original analysis
                if include_flw_muac:
                    flw_muac_original = muac_analyzer.create_flw_muac_analysis()
                    if flw_muac_original is not None and len(flw_muac_original) > 0:
                        excel_data['FLW MUAC Original'] = flw_muac_original
                
                if include_opps_muac:
                    opps_muac_original = muac_analyzer.create_opps_muac_analysis()
                    if opps_muac_original is not None and len(opps_muac_original) > 0:
                        excel_data['Opps MUAC Original'] = opps_muac_original
                
                # Run tampering experiment if requested
                if include_tampering and (flw_muac_original is not None or opps_muac_original is not None):
                    self.log("Running MUAC tampering experiment...")
                    
                    # Apply tampering to the dataset
                    df_tampered = self._apply_muac_tampering(self.df, tamper_pct=20, noise_range_min=2, noise_range_max=6, seed=42)
                    
                    # Create new analyzer with tampered data
                    muac_analyzer_tampered = EnhancedFLWMUACAnalyzer(df_tampered, self.log)
                    
                    if include_flw_muac and flw_muac_original is not None:
                        flw_muac_tampered = muac_analyzer_tampered.create_flw_muac_analysis()
                        if flw_muac_tampered is not None and len(flw_muac_tampered) > 0:
                            excel_data['FLW MUAC Tampered'] = flw_muac_tampered
                    
                    if include_opps_muac and opps_muac_original is not None:
                        opps_muac_tampered = muac_analyzer_tampered.create_opps_muac_analysis()
                        if opps_muac_tampered is not None and len(opps_muac_tampered) > 0:
                            excel_data['Opps MUAC Tampered'] = opps_muac_tampered
                    
                    # Create tampering impact summaries
                    if flw_muac_original is not None and flw_muac_tampered is not None:
                        self.log("Creating FLW MUAC tampering impact summary...")
                        flw_tampering_summary = self._create_tampering_summary(flw_muac_original, flw_muac_tampered)
                        if len(flw_tampering_summary) > 0:
                            excel_data['FLW MUAC Tampering Impact'] = flw_tampering_summary
                    
                    if opps_muac_original is not None and opps_muac_tampered is not None:
                        self.log("Creating Opps MUAC tampering impact summary...")
                        opps_tampering_summary = self._create_tampering_summary(opps_muac_original, opps_muac_tampered)
                        if len(opps_tampering_summary) > 0:
                            excel_data['Opps MUAC Tampering Impact'] = opps_tampering_summary
                
                # Generate fraud detection visualizations for original data
                if flw_muac_original is not None or opps_muac_original is not None:
                    self.log("Creating MUAC fraud detection visualizations...")
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    viz_output_dir = os.path.join(self.output_dir, f"muac_fraud_analysis_{current_date}")
                    
                    # Export with visualizations (original data only for main analysis)
                    viz_results = muac_analyzer.export_detailed_results(
                        flw_muac_original, opps_muac_original, viz_output_dir
                    )
                    
                    if viz_results:
                        self.log("MUAC fraud detection analysis complete:")
                        for key, path in viz_results.items():
                            if path:
                                self.log(f"  - {key}: {os.path.basename(path)}")
                                # Add visualization files to output
                                output_files.append(path)
                
                # Log tampering experiment results
                if include_tampering and (flw_muac_tampered is not None or opps_muac_tampered is not None):
                    self.log("=== MUAC Tampering Experiment Results ===")
                    
                    if flw_muac_original is not None and flw_muac_tampered is not None:
                        # Calculate key metrics for logging
                        orig_sufficient = flw_muac_original[flw_muac_original['data_sufficiency'] == 'SUFFICIENT']
                        tamp_sufficient = flw_muac_tampered[flw_muac_tampered['data_sufficiency'] == 'SUFFICIENT']
                        
                        if len(orig_sufficient) > 0 and len(tamp_sufficient) > 0:
                            orig_avg_score = orig_sufficient['authenticity_score'].mean()
                            tamp_avg_score = tamp_sufficient['authenticity_score'].mean()
                            
                            orig_fabricated_pct = (orig_sufficient['authenticity_score'] < 4).sum() / len(orig_sufficient) * 100
                            tamp_fabricated_pct = (tamp_sufficient['authenticity_score'] < 4).sum() / len(tamp_sufficient) * 100
                            
                            self.log(f"FLW Analysis - Original avg score: {orig_avg_score:.2f}, Tampered avg score: {tamp_avg_score:.2f}")
                            self.log(f"FLW Analysis - 'Likely Fabricated': {orig_fabricated_pct:.1f}% ? {tamp_fabricated_pct:.1f}%")
                    
                    if opps_muac_original is not None and opps_muac_tampered is not None:
                        # Calculate key metrics for logging
                        orig_sufficient = opps_muac_original[opps_muac_original['data_sufficiency'] == 'SUFFICIENT']
                        tamp_sufficient = opps_muac_tampered[opps_muac_tampered['data_sufficiency'] == 'SUFFICIENT']
                        
                        if len(orig_sufficient) > 0 and len(tamp_sufficient) > 0:
                            orig_avg_score = orig_sufficient['authenticity_score'].mean()
                            tamp_avg_score = tamp_sufficient['authenticity_score'].mean()
                            
                            orig_fabricated_pct = (orig_sufficient['authenticity_score'] < 4).sum() / len(orig_sufficient) * 100
                            tamp_fabricated_pct = (tamp_sufficient['authenticity_score'] < 4).sum() / len(tamp_sufficient) * 100
                            
                            self.log(f"Opps Analysis - Original avg score: {orig_avg_score:.2f}, Tampered avg score: {tamp_avg_score:.2f}")
                            self.log(f"Opps Analysis - 'Likely Fabricated': {orig_fabricated_pct:.1f}% ? {tamp_fabricated_pct:.1f}%")
            
            # Export to Excel
            if excel_data:
                self.log("Creating Excel file with all results...")
                excel_exporter = ExcelExporter(self.log)
                excel_file = excel_exporter.export_to_excel(
                    excel_data,
                    os.path.join(self.output_dir, f"flw_data_quality_assessment_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx")
                )
                if excel_file:
                    output_files.append(excel_file)
                    self.log(f"Created Excel file: {os.path.basename(excel_file)}")
                    
                    # Log Excel tabs created
                    tab_count = len(excel_data)
                    tab_names = list(excel_data.keys())
                    self.log(f"Excel file contains {tab_count} tabs: {', '.join(tab_names)}")
            
            # Log summary statistics if assessments were run
            if selected_assessments and 'excel_results' in locals() and len(excel_results) > 0:
                self._log_summary_stats(excel_results, all_batch_results if 'all_batch_results' in locals() else pd.DataFrame(), 
                                      opportunity_summary if 'opportunity_summary' in locals() else pd.DataFrame())
            
        except Exception as e:
            self.log(f"Error during data quality assessment: {str(e)}")
            raise
        
        return output_files
    
    def _create_assessment_summary(self, assessment_results, population_stats, batch_size, min_size):
        """Create summary statistics for the assessment run"""
        
        if len(assessment_results) == 0:
            return pd.DataFrame()
        
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Metric': 'Batch Size Used',
            'Value': batch_size,
            'Description': 'Maximum visits per batch'
        })
        
        summary_data.append({
            'Metric': 'Minimum Batch Size',
            'Value': min_size,
            'Description': 'Minimum visits required for assessment'
        })
        
        summary_data.append({
            'Metric': 'Total FLW/Opportunity Pairs Assessed',
            'Value': len(assessment_results),
            'Description': 'Number of FLW/opportunity combinations evaluated (most recent batches)'
        })
        
        summary_data.append({
            'Metric': 'Pairs with Data Quality Issues',
            'Value': assessment_results['has_any_strong_negative'].sum(),
            'Description': 'Number of pairs flagged with concerning patterns in most recent batch'
        })
        
        summary_data.append({
            'Metric': 'Pairs with Insufficient Data',
            'Value': assessment_results['has_insufficient_data'].sum(),
            'Description': 'Number of pairs with insufficient data for some assessments'
        })
        
        summary_data.append({
            'Metric': 'Percentage with Issues',
            'Value': f"{(assessment_results['has_any_strong_negative'].sum() / len(assessment_results) * 100):.1f}%",
            'Description': 'Percentage of assessed pairs with quality flags in most recent batch'
        })
        
        summary_data.append({
            'Metric': 'Percentage with Insufficient Data',
            'Value': f"{(assessment_results['has_insufficient_data'].sum() / len(assessment_results) * 100):.1f}%",
            'Description': 'Percentage of assessed pairs with insufficient data'
        })
        
        # Assessment-specific statistics
        assessment_cols = [col for col in assessment_results.columns if col.endswith('_result')]
        
        for col in assessment_cols:
            assessment_name = col.replace('_result', '')
            strong_negative_count = (assessment_results[col] == 'strong_negative').sum()
            insufficient_data_count = (assessment_results[col] == 'insufficient_data').sum()
            total_with_data = assessment_results[col].notna().sum()
            
            if total_with_data > 0:
                summary_data.append({
                    'Metric': f'{assessment_name} - Issues Found',
                    'Value': strong_negative_count,
                    'Description': f'FLW/opp pairs flagged for {assessment_name} quality issues in most recent batch'
                })
                
                summary_data.append({
                    'Metric': f'{assessment_name} - Insufficient Data',
                    'Value': insufficient_data_count,
                    'Description': f'FLW/opp pairs with insufficient {assessment_name} data'
                })
                
                summary_data.append({
                    'Metric': f'{assessment_name} - Issue Rate',
                    'Value': f"{(strong_negative_count / (total_with_data - insufficient_data_count) * 100):.1f}%" if (total_with_data - insufficient_data_count) > 0 else "0.0%",
                    'Description': f'Percentage with {assessment_name} quality issues (excluding insufficient data)'
                })
        
        # Population statistics if available
        if population_stats and 'gender_ratio' in population_stats:
            gender_stats = population_stats['gender_ratio']
            summary_data.append({
                'Metric': 'Population Female Ratio',
                'Value': f"{gender_stats['female_ratio']:.3f}",
                'Description': 'Overall population female child ratio'
            })
            
            summary_data.append({
                'Metric': 'Population CI Range',
                'Value': f"{gender_stats['ci_lower']:.3f} - {gender_stats['ci_upper']:.3f}",
                'Description': '99% confidence interval for population gender ratio'
            })
        
        return pd.DataFrame(summary_data)
    
    def _log_summary_stats(self, excel_results, all_batch_results, opportunity_summary):
        """Log summary statistics to the GUI"""
        
        if len(excel_results) == 0:
            return
        
        self.log("=== Data Quality Assessment Summary ===")
        self.log(f"Total FLW/opportunity pairs assessed: {len(excel_results)} (most recent batches)")
        
        if len(all_batch_results) > 0:
            total_batches = len(all_batch_results)
            unique_pairs = all_batch_results[['flw_id', 'opportunity_name']].drop_duplicates()
            avg_batches_per_pair = total_batches / len(unique_pairs)
            self.log(f"Total batches created: {total_batches} (avg {avg_batches_per_pair:.1f} batches per FLW/opp pair)")
        
        flagged_count = excel_results['has_any_strong_negative'].sum()
        flagged_pct = (flagged_count / len(excel_results) * 100)
        self.log(f"Pairs with quality issues in most recent batch: {flagged_count} ({flagged_pct:.1f}%)")
        
        insufficient_count = excel_results['has_insufficient_data'].sum()
        insufficient_pct = (insufficient_count / len(excel_results) * 100)
        self.log(f"Pairs with insufficient data: {insufficient_count} ({insufficient_pct:.1f}%)")
        
        # Assessment-specific counts
        assessment_cols = [col for col in excel_results.columns if col.endswith('_result')]
        for col in assessment_cols:
            assessment_name = col.replace('_result', '').replace('_', ' ').title()
            issues = (excel_results[col] == 'strong_negative').sum()
            insufficient = (excel_results[col] == 'insufficient_data').sum()
            if issues > 0 or insufficient > 0:
                self.log(f"  - {assessment_name}: {issues} issues, {insufficient} insufficient data")
        
        if len(opportunity_summary) > 0:
            self.log(f"Opportunities analyzed: {len(opportunity_summary)}")
            worst_opp = opportunity_summary.iloc[0] if len(opportunity_summary) > 0 else None
            if worst_opp is not None and worst_opp['pct_flws_with_strong_negative'] > 0:
                self.log(f"Highest issue rate: {worst_opp['opportunity_name']} ({worst_opp['pct_flws_with_strong_negative']:.1f}%)")

