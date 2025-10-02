"""
ML Feature Aggregation Report

Aggregates visit-level data to FLW-level features for machine learning fraud detection.
Processes multiple CSV files and uses filename prefixes for labeling.
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
from .base_report import BaseReport


class MLFeatureAggregationReport(BaseReport):
    """
    Aggregates visit-level data to FLW-level features for ML fraud detection.
    Processes all CSV files in input directory, using filename prefixes for labels.
    """
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up GUI parameters for ML feature aggregation"""
        
        # Directory selection for CSV files
        ttk.Label(parent_frame, text="CSV Files Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        csv_dir_var = tk.StringVar()
        
        # Try to load last-used directory
        config_file = os.path.join(os.path.expanduser("~"), ".ml_features_last_dir.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    last_dir = f.read().strip()
                    if os.path.exists(last_dir):
                        csv_dir_var.set(last_dir)
            except Exception:
                pass

        dir_frame = ttk.Frame(parent_frame)
        dir_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=csv_dir_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            dir_frame,
            text="Browse...",
            command=lambda: MLFeatureAggregationReport._browse_and_save_csv_dir_static(csv_dir_var, config_file),
        ).grid(row=0, column=1, padx=(6, 0))
        dir_frame.columnconfigure(0, weight=1)

        parent_frame.csv_dir_var = csv_dir_var
        
        ttk.Label(parent_frame, text="Process all CSV files with 'real_' or 'fake_' prefixes").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        # Min visits per FLW
        ttk.Label(parent_frame, text="Min visits per FLW:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        parent_frame.min_visits_var = tk.StringVar(value="20")
        ttk.Entry(parent_frame, textvariable=parent_frame.min_visits_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Max visits per FLW  
        ttk.Label(parent_frame, text="Max visits per FLW:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        parent_frame.max_visits_var = tk.StringVar(value="200")
        ttk.Entry(parent_frame, textvariable=parent_frame.max_visits_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

    @staticmethod
    def _browse_and_save_csv_dir_static(var, config_file):
        """Browse for directory containing CSV files"""
        from tkinter import filedialog
        directory = filedialog.askdirectory(
            title="Select directory containing CSV files with real_/fake_ prefixes"
        )
        if directory:
            var.set(directory)
            try:
                with open(config_file, "w") as f:
                    f.write(directory)
            except Exception:
                pass

    def generate(self):
        """Generate ML feature aggregation"""
        output_files = []
        
        # Get parameters
        min_visits = int(self.get_parameter_value('min_visits', 20))
        max_visits = int(self.get_parameter_value('max_visits', 200))
        csv_dir = self.get_parameter_value('csv_dir', '')
        
        if not csv_dir or not os.path.exists(csv_dir):
            raise ValueError("Please specify a valid directory containing CSV files with real_/fake_ prefixes")
        
        input_dir = Path(csv_dir)
        csv_files = list(input_dir.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {input_dir}")
        
        self.log(f"Found {len(csv_files)} CSV files to process in {input_dir}")
        
        # Process each file and combine
        all_flw_data = []
        
        for csv_file in csv_files:
            filename = csv_file.name
            
            # Determine classification from filename
            if filename.lower().startswith('real'):
                classification = 'real'
            elif filename.lower().startswith('fake'):
                classification = 'fake'
            else:
                self.log(f"Warning: Skipping {filename} - no 'real' or 'fake' prefix")
                continue
            
            self.log(f"Processing {filename} as {classification}...")
            
            try:
                # Load and process this file
                df = pd.read_csv(csv_file)
                self.log(f"  -> Loaded {len(df)} records")
                
                flw_features = self._aggregate_flw_features(df, classification, min_visits, max_visits)
                
                if len(flw_features) > 0:
                    all_flw_data.append(flw_features)
                    self.log(f"  -> Extracted {len(flw_features)} FLWs")
                else:
                    self.log(f"  -> No FLWs met criteria")
                    
            except Exception as e:
                self.log(f"  -> Error processing {filename}: {str(e)}")
                continue
        
        if not all_flw_data:
            raise ValueError("No FLW data extracted from any files")
        
        # Combine all FLW data
        combined_df = pd.concat(all_flw_data, ignore_index=True)
        
        # Create output directory with date
        today = datetime.now().strftime("%Y_%m_%d")
        output_subdir = os.path.join(self.output_dir, f"ml_features_{today}")
        os.makedirs(output_subdir, exist_ok=True)
        
        total_flws = len(combined_df)
        total_visits = combined_df['visits'].sum() if 'visits' in combined_df.columns else 0
        real_flws = len(combined_df[combined_df['classification'] == 'real'])
        fake_flws = len(combined_df[combined_df['classification'] == 'fake'])
        
        self.log(f"Combined data: {total_flws} FLWs total ({total_visits:,} visits)")
        self.log(f"  - Real: {real_flws} FLWs")
        self.log(f"  - Fake: {fake_flws} FLWs")
        
        # Save aggregated features with descriptive filename
        features_file = os.path.join(output_subdir, f"ml_features_{total_flws}flws_{total_visits}visits.csv")
        combined_df.to_csv(features_file, index=False)
        output_files.append(features_file)
        
        # Generate feature summary
        feature_summary = self._generate_feature_summary(combined_df)
        summary_file = os.path.join(output_subdir, f"ml_features_summary_{total_flws}flws.csv")
        feature_summary.to_csv(summary_file, index=False)
        output_files.append(summary_file)
        
        self.log("ML feature aggregation complete!")
        return output_files
    
    def _aggregate_flw_features(self, df, classification, min_visits, max_visits):
        """Aggregate visit-level data to FLW-level features"""
        
        # Ensure we have flw_id column
        if 'flw_id' not in df.columns:
            raise ValueError("Input data must have 'flw_id' column")
        
        # Debug: Check flw_id distribution
        total_records = len(df)
        unique_flw_ids = df['flw_id'].nunique()
        null_flw_ids = df['flw_id'].isnull().sum()
        
        self.log(f"    DEBUG: Total records={total_records}, unique FLW IDs={unique_flw_ids}, null FLW IDs={null_flw_ids}")
        
        # Show visit count distribution
        visit_counts_per_flw = df.groupby('flw_id').size()
        self.log(f"    DEBUG: Visit counts per FLW - min={visit_counts_per_flw.min()}, max={visit_counts_per_flw.max()}, mean={visit_counts_per_flw.mean():.1f}")
        
        # Count how many FLWs are in each range
        too_few = (visit_counts_per_flw < min_visits).sum()
        too_many = (visit_counts_per_flw > max_visits).sum()
        valid_flws = (visit_counts_per_flw >= min_visits).sum()
        
        self.log(f"    DEBUG: FLWs with <{min_visits} visits (excluded): {too_few}")
        self.log(f"    DEBUG: FLWs with >{max_visits} visits (will be truncated): {too_many}")  
        self.log(f"    DEBUG: FLWs with >={min_visits} visits (will be included): {valid_flws}")
        
        # Group by FLW
        flw_groups = df.groupby('flw_id')
        flw_data = []
        
        for flw_id, group in flw_groups:
            visit_count = len(group)
            
            # Filter by minimum visits
            if visit_count < min_visits:
                continue
            
            # If FLW has too many visits, take only the first max_visits chronologically
            if visit_count > max_visits:
                # Sort by visit_date and take the earliest visits
                if 'visit_date' in group.columns:
                    group = group.sort_values('visit_date').head(max_visits)
                else:
                    # Fallback to first N rows if no visit_date column
                    group = group.head(max_visits)
                visit_count = max_visits
            
            # Basic FLW info
            flw_features = {
                'flw_id': flw_id,
                'flw_name': group['flw_name'].iloc[0] if 'flw_name' in group.columns else None,
                'opportunity_id': group['opportunity_id'].iloc[0] if 'opportunity_id' in group.columns else None,
                'opportunity_name': group['opportunity_name'].iloc[0] if 'opportunity_name' in group.columns else None,
                'visits': visit_count,
                'classification': classification
            }
            
            # Add all feature types
            flw_features.update(self._calculate_gender_features(group))
            flw_features.update(self._calculate_age_features(group))
            flw_features.update(self._calculate_muac_features(group))
            
            flw_data.append(flw_features)
        
        return pd.DataFrame(flw_data)
    
    def _calculate_gender_features(self, group):
        """Calculate gender-related features"""
        features = {}
        
        if 'childs_gender' in group.columns:
            gender_data = group['childs_gender'].dropna()
            if len(gender_data) > 0:
                features['pct_female'] = (gender_data == 'female_child').mean()
            else:
                features['pct_female'] = None
        else:
            features['pct_female'] = None
        
        return features
    
    def _calculate_muac_features(self, group):
        """Calculate MUAC-related features based on fraud detection patterns"""
        features = {}
        
        # Check if we have MUAC measurement data
        muac_col = None
        for col_name in ['soliciter_muac_cm', 'muac_measurement_cm', 'muac_cm', 'muac']:
            if col_name in group.columns:
                muac_col = col_name
                break
        
        if muac_col is None:
            # No MUAC column found
            features.update({
                'has_muac_data': False,
                'muac_completion_rate': 0.0,
                'muac_bins_with_data': -1,
                'muac_increasing_to_peak': -1,
                'muac_decreasing_from_peak': -1,
                'muac_has_skipped_bins': -1,
                'muac_has_plateau': -1,
                'muac_peak_concentration': -1
            })
            return features
        
        # Get valid MUAC measurements (9.5-21.5 cm range)
        muac_data = pd.to_numeric(group[muac_col], errors='coerce').dropna()
        valid_muac = muac_data[(muac_data >= 9.5) & (muac_data <= 21.5)]
        
        # Basic completion rate
        features['muac_completion_rate'] = len(muac_data) / len(group)
        
        if len(valid_muac) < 20:  # Need minimum data for distribution analysis
            features.update({
                'has_muac_data': False,
                'muac_bins_with_data': -1,
                'muac_increasing_to_peak': -1,
                'muac_decreasing_from_peak': -1,
                'muac_has_skipped_bins': -1,
                'muac_has_plateau': -1,
                'muac_peak_concentration': -1
            })
            return features
        
        features['has_muac_data'] = True
        
        # Calculate MUAC distribution features using the same logic as fraud detection
        bin_edges = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5]
        
        # Calculate bin counts
        bin_counts = []
        for i in range(len(bin_edges) - 1):
            min_val, max_val = bin_edges[i], bin_edges[i + 1]
            count = ((valid_muac >= min_val) & (valid_muac < max_val)).sum()
            bin_counts.append(count)
        
        total_count = sum(bin_counts)
        non_zero_indices = [i for i, count in enumerate(bin_counts) if count > 0]
        
        # Feature 1: Number of bins with data
        features['muac_bins_with_data'] = len(non_zero_indices)
        
        if len(non_zero_indices) == 0:
            features.update({
                'muac_increasing_to_peak': -1,
                'muac_decreasing_from_peak': -1,
                'muac_has_skipped_bins': -1,
                'muac_has_plateau': -1,
                'muac_peak_concentration': -1
            })
            return features
        
        # Find peak
        max_count = max(bin_counts)
        peak_indices = [i for i, count in enumerate(bin_counts) if count == max_count]
        peak_index = peak_indices[0]
        
        # Feature 2: Increasing to peak test
        features['muac_increasing_to_peak'] = 1 if self._test_increasing_to_peak(bin_counts, non_zero_indices, peak_index, total_count * 0.02) else 0
        
        # Feature 3: Decreasing from peak test
        features['muac_decreasing_from_peak'] = 1 if self._test_decreasing_from_peak(bin_counts, non_zero_indices, peak_index, total_count * 0.02) else 0
        
        # Feature 4: No skipped bins test (inverted - we want to know if there ARE skipped bins)
        features['muac_has_skipped_bins'] = 0 if self._test_no_skipped_bins(bin_counts, total_count) else 1
        
        # Feature 5: Has plateau test
        features['muac_has_plateau'] = 0 if self._test_no_plateau(bin_counts, max_count) else 1
        
        # Feature 6: Peak concentration
        features['muac_peak_concentration'] = max_count / total_count if total_count > 0 else 0
        
        return features
    
    def _test_increasing_to_peak(self, bin_counts, non_zero_indices, peak_index, wiggle_threshold):
        """Test for proper increasing pattern to peak (adapted from aggregation_functions.py)"""
        if peak_index == 0:
            return False
        
        bins_before_peak = [i for i in non_zero_indices if i < peak_index]
        if len(bins_before_peak) == 0:
            return False
        
        if peak_index not in non_zero_indices:
            return False
        
        peak_position_in_nonzero = non_zero_indices.index(peak_index)
        if peak_position_in_nonzero == 0:
            return False
        
        increasing_steps = 0
        big_decreases = 0
        
        for i in range(peak_position_in_nonzero):
            current_idx = non_zero_indices[i]
            next_idx = non_zero_indices[i + 1]
            step_change = bin_counts[next_idx] - bin_counts[current_idx]
            
            if step_change > 0:
                increasing_steps += 1
            if step_change < -wiggle_threshold:
                big_decreases += 1
        
        # Check for adequate buildup
        peak_value = bin_counts[peak_index]
        adequate_threshold = peak_value * 0.25
        has_adequate_buildup = any(bin_counts[i] >= adequate_threshold for i in range(peak_index))
        
        return increasing_steps >= 1 and big_decreases == 0 and has_adequate_buildup
    
    def _test_decreasing_from_peak(self, bin_counts, non_zero_indices, peak_index, wiggle_threshold):
        """Test for proper decreasing pattern from peak (adapted from aggregation_functions.py)"""
        if peak_index == len(bin_counts) - 1 or peak_index not in non_zero_indices:
            return True
        
        peak_position_in_nonzero = non_zero_indices.index(peak_index)
        if peak_position_in_nonzero == len(non_zero_indices) - 1:
            return True
        
        steps_from_peak = len(non_zero_indices) - 1 - peak_position_in_nonzero
        if steps_from_peak < 2:
            return True
        
        decreasing_steps = 0
        big_increases = 0
        
        for i in range(peak_position_in_nonzero, len(non_zero_indices) - 1):
            current_idx = non_zero_indices[i]
            next_idx = non_zero_indices[i + 1]
            step_change = bin_counts[next_idx] - bin_counts[current_idx]
            
            if step_change < 0:
                decreasing_steps += 1
            if step_change > wiggle_threshold:
                big_increases += 1
        
        return decreasing_steps >= 2 and big_increases == 0
    
    def _test_no_skipped_bins(self, bin_counts, total_count):
        """Test for no skipped bins (adapted from aggregation_functions.py)"""
        if total_count <= 0:
            return True
        
        significant_bin_threshold = total_count * 0.02
        sig_indices = [i for i, count in enumerate(bin_counts) if count >= significant_bin_threshold]
        
        if len(sig_indices) <= 1:
            return True
        
        start_idx = sig_indices[0]
        end_idx = sig_indices[-1]
        interior = bin_counts[start_idx:end_idx + 1]
        
        return all(count != 0 for count in interior)
    
    def _test_no_plateau(self, bin_counts, max_count):
        """Test for no suspicious plateaus (adapted from aggregation_functions.py)"""
        total_count = sum(bin_counts)
        if total_count == 0 or max_count == 0:
            return True
        
        change_threshold = max_count * 0.03
        min_count_threshold = total_count * 0.02
        min_peak_ratio_threshold = max_count * 0.20
        
        eligible_bins = []
        for i, count in enumerate(bin_counts):
            if count >= min_count_threshold and count >= min_peak_ratio_threshold:
                eligible_bins.append((i, count))
        
        if len(eligible_bins) < 3:
            return True
        
        for i in range(len(eligible_bins) - 2):
            bin1_idx, bin1_count = eligible_bins[i]
            bin2_idx, bin2_count = eligible_bins[i + 1]
            bin3_idx, bin3_count = eligible_bins[i + 2]
            
            if bin2_idx == bin1_idx + 1 and bin3_idx == bin2_idx + 1:
                counts = [bin1_count, bin2_count, bin3_count]
                if max(counts) - min(counts) <= change_threshold:
                    return False
        
        return True
    
    def _calculate_age_features(self, group):
        """Calculate age-related features"""
        features = {}
        
        if 'childs_age_in_month' not in group.columns:
            # No age column - set all features to sentinel values
            features.update({
                'has_age_data': False,
                'age_completion_rate': 0.0,
                'age_imbalance_6month': -1,
                'age_imbalance_12month': -1
            })
            # Add all 12 age bin percentages
            for i in range(12):
                start_month = i * 5
                end_month = start_month + 4
                features[f'pct_age_{start_month}_{end_month}'] = -1
            return features
        
        age_data = pd.to_numeric(group['childs_age_in_month'], errors='coerce').dropna()
        features['age_completion_rate'] = len(age_data) / len(group)
        
        if len(age_data) < 10:  # Need minimum data for analysis
            features.update({
                'has_age_data': False,
                'age_imbalance_6month': -1,
                'age_imbalance_12month': -1
            })
            # Add all 12 age bin percentages
            for i in range(12):
                start_month = i * 5
                end_month = start_month + 4
                features[f'pct_age_{start_month}_{end_month}'] = -1
            return features
        
        features['has_age_data'] = True
        
        # Filter to under-5 children (0-59 months)
        under_5_ages = age_data[age_data < 60]
        
        if len(under_5_ages) < 10:
            features.update({
                'age_imbalance_6month': -1,
                'age_imbalance_12month': -1
            })
            # Add all 12 age bin percentages
            for i in range(12):
                start_month = i * 5
                end_month = start_month + 4
                features[f'pct_age_{start_month}_{end_month}'] = -1
            return features
        
        # Calculate 6-month bin percentages (12 bins total: 0-4, 5-9, ..., 55-59)
        for i in range(12):
            start_month = i * 5
            end_month = start_month + 4
            if end_month >= 59:  # Last bin goes to 59
                end_month = 59
            
            bin_data = under_5_ages[(under_5_ages >= start_month) & (under_5_ages <= end_month)]
            pct = len(bin_data) / len(under_5_ages) if len(under_5_ages) > 0 else 0
            features[f'pct_age_{start_month}_{end_month}'] = pct
        
        # Calculate 6-month imbalance (deviation from uniform across 12 bins)
        bin_edges_6m = list(range(0, 61, 5))  # 0, 5, 10, ..., 60
        bin_counts_6m = []
        for i in range(len(bin_edges_6m) - 1):
            start, end = bin_edges_6m[i], bin_edges_6m[i + 1] - 1
            if i == len(bin_edges_6m) - 2:  # Last bin
                end = 59
            count = len(under_5_ages[(under_5_ages >= start) & (under_5_ages <= end)])
            bin_counts_6m.append(count)
        
        total_6m = sum(bin_counts_6m)
        if total_6m > 0:
            actual_proportions_6m = [count / total_6m for count in bin_counts_6m]
            ideal_proportion_6m = 1.0 / 12  # Uniform across 12 bins
            deviation_6m = sum(abs(prop - ideal_proportion_6m) for prop in actual_proportions_6m)
            features['age_imbalance_6month'] = min(deviation_6m / 1.83, 1.0)  # Normalize to 0-1
        else:
            features['age_imbalance_6month'] = -1
        
        # Calculate 12-month imbalance (deviation from uniform across 5 yearly bins)
        yearly_bin_edges = [0, 12, 24, 36, 48, 60]
        yearly_bin_counts = []
        for i in range(len(yearly_bin_edges) - 1):
            start, end = yearly_bin_edges[i], yearly_bin_edges[i + 1] - 1
            if i == len(yearly_bin_edges) - 2:  # Last bin
                end = 59
            count = len(under_5_ages[(under_5_ages >= start) & (under_5_ages <= end)])
            yearly_bin_counts.append(count)
        
        total_yearly = sum(yearly_bin_counts)
        if total_yearly > 0:
            actual_proportions_yearly = [count / total_yearly for count in yearly_bin_counts]
            ideal_proportion_yearly = 0.2  # Uniform across 5 bins
            deviation_yearly = sum(abs(prop - ideal_proportion_yearly) for prop in actual_proportions_yearly)
            features['age_imbalance_12month'] = min(deviation_yearly / 1.6, 1.0)  # Normalize to 0-1
        else:
            features['age_imbalance_12month'] = -1
        
        return features
    
    def _generate_feature_summary(self, df):
        """Generate summary statistics for all features"""
        
        # Separate numeric and categorical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        summary_data = []
        
        # Numeric feature summaries
        for col in numeric_cols:
            if col in ['flw_id']:  # Skip ID columns
                continue
                
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_data.append({
                    'feature': col,
                    'type': 'numeric',
                    'count': len(col_data),
                    'missing': len(df) - len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'unique_values': None
                })
        
        # Categorical feature summaries  
        for col in categorical_cols:
            if col in ['flw_id', 'flw_name']:  # Skip ID/name columns
                continue
                
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_data.append({
                    'feature': col,
                    'type': 'categorical',
                    'count': len(col_data),
                    'missing': len(df) - len(col_data),
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'unique_values': ', '.join(map(str, col_data.unique()[:10]))  # First 10 unique values
                })
        
        return pd.DataFrame(summary_data)
