"""
Enhanced FLW MUAC Analysis - MUAC measurement distribution analysis with simplified fraud detection
Analyzes MUAC measurements with streamlined authenticity scoring and visualization

Save this file as: flw_muac_analyzer_enhanced.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class EnhancedFLWMUACAnalyzer:
    """Analyzes MUAC measurement distributions with simplified fraud detection capabilities"""
    
    def __init__(self, df, log_func):
        self.df = df
        self.log = log_func
        self.muac_col = 'muac_measurement_cm'
        self.normality_threshold = 300
        
        # Define MUAC bins (1cm each, on half-centimeters)
        self.muac_bins = [
            (9.5, 10.5, 'muac_9_5_10_5'),
            (10.5, 11.5, 'muac_10_5_11_5'),
            (11.5, 12.5, 'muac_11_5_12_5'),
            (12.5, 13.5, 'muac_12_5_13_5'),
            (13.5, 14.5, 'muac_13_5_14_5'),
            (14.5, 15.5, 'muac_14_5_15_5'),
            (15.5, 16.5, 'muac_15_5_16_5'),
            (16.5, 17.5, 'muac_16_5_17_5'),
            (17.5, 18.5, 'muac_17_5_18_5'),
            (18.5, 19.5, 'muac_18_5_19_5'),
            (19.5, 20.5, 'muac_19_5_20_5'),
            (20.5, 21.5, 'muac_20_5_21_5')
        ]
    
    def _detect_column(self, possible_names):
        """Detect column name from a list of possibilities"""
        for name in possible_names:
            matching_cols = [col for col in self.df.columns if name.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        return None
    
    def _detect_plateau(self, counts, max_count, total_count):
        """
        Detect plateau in MUAC distribution
        
        Returns:
            plateau_length (int): Length of longest plateau, 1 if no significant plateau
        """
        if max_count == 0:
            return 1
        
        # Only consider bins with at least 50% of peak count
        threshold = 0.5 * max_count
        high_bins = []
        
        for i, count in enumerate(counts):
            if count >= threshold:
                high_bins.append((i, count))
        
        if len(high_bins) < 2:
            return 1  # No plateau possible
        
        # Plateau tolerance: 4% of total count
        plateau_tolerance = total_count * 0.04
        
        longest_plateau = 1
        current_plateau_start = 0
        
        for i in range(1, len(high_bins)):
            # Check if bins are consecutive
            current_bin_idx = high_bins[i][0]
            prev_bin_idx = high_bins[i-1][0]
            
            if current_bin_idx == prev_bin_idx + 1:
                # Bins are consecutive, check if counts are similar enough
                current_segment = high_bins[current_plateau_start:i+1]
                segment_counts = [item[1] for item in current_segment]
                
                if max(segment_counts) - min(segment_counts) <= plateau_tolerance:
                    # This extends the current plateau
                    plateau_length = len(current_segment)
                    longest_plateau = max(longest_plateau, plateau_length)
                else:
                    # Plateau broken by count difference, start new potential plateau
                    current_plateau_start = i - 1
            else:
                # Bins not consecutive, start new potential plateau
                current_plateau_start = i - 1
        
        return longest_plateau
    
    def _calculate_simplified_fraud_features(self, bin_counts, min_threshold=80):
        """Calculate simplified fraud detection features from MUAC bin counts"""
        counts = np.array(bin_counts)
        total_count = np.sum(counts)
        
        # Check if we have sufficient data for fraud detection
        if total_count < min_threshold:
            return self._get_insufficient_data_features(total_count, min_threshold)
        
        if total_count == 0:
            return self._get_empty_fraud_features()
        
        # Find non-zero bins and peak
        non_zero_indices = np.where(counts > 0)[0]
        if len(non_zero_indices) == 0:
            return self._get_empty_fraud_features()
        
        non_zero_counts = counts[non_zero_indices]
        max_count = np.max(non_zero_counts)
        peak_indices = np.where(counts == max_count)[0]
        peak_index = peak_indices[0] if len(peak_indices) > 0 else -1
        
        # Calculate 2% threshold for wiggle room
        wiggle_threshold = total_count * 0.02
        
        # Calculate plateau detection
        plateau_length = self._detect_plateau(counts, max_count, total_count)
        
        # 1. Calculate bins_with_data (0-2 points)
        bins_with_data = len(non_zero_indices)
        if bins_with_data >= 5:
            bins_score = 2
        elif bins_with_data >= 3:
            bins_score = 1
        else:
            bins_score = 0
        
        # 2. Calculate peak_dominance_pct (0-2 points)
        peak_dominance_pct = (max_count / total_count * 100) if total_count > 0 else 0
        if 30 <= peak_dominance_pct <= 60:
            dominance_score = 2
        elif (20 <= peak_dominance_pct < 30) or (60 < peak_dominance_pct <= 80):
            dominance_score = 1
        else:
            dominance_score = 0
        
        # 3. Calculate normalized linear regression slopes and shape indicators
        regression_slope_to_peak = None
        regression_slope_from_peak = None
        normalized_slope_to_peak = None
        normalized_slope_from_peak = None
        increasing_to_peak_with_wiggle = False
        decreasing_from_peak_with_wiggle = False
        slope_to_peak_score = 0
        slope_from_peak_score = 0
        
        if peak_index >= 0 and bins_with_data >= 3:
            # Find position of peak among non-zero bins
            non_zero_positions = non_zero_indices.tolist()
            if peak_index in non_zero_positions:
                peak_position_in_nonzero = non_zero_positions.index(peak_index)
                
                # Calculate linear regression slope to peak
                if peak_position_in_nonzero > 0:
                    # Get x (bin positions) and y (counts) for regression
                    x_to_peak = np.array(non_zero_positions[:peak_position_in_nonzero + 1])
                    y_to_peak = np.array([counts[i] for i in x_to_peak])
                    
                    if len(x_to_peak) >= 2:
                        # Calculate linear regression slope
                        regression_slope_to_peak, _ = np.polyfit(x_to_peak, y_to_peak, 1)
                        
                        # Normalize by peak count
                        normalized_slope_to_peak = regression_slope_to_peak / max_count if max_count > 0 else 0
                        
                        # Check for step-by-step violations (for wiggle room assessment)
                        violations_to_peak = 0
                        for i in range(peak_position_in_nonzero):
                            current_idx = non_zero_positions[i]
                            next_idx = non_zero_positions[i + 1]
                            step_change = counts[next_idx] - counts[current_idx]
                            
                            # Check for violations (decreases larger than wiggle threshold)
                            if step_change < -wiggle_threshold:
                                violations_to_peak += 1
                        
                        increasing_to_peak_with_wiggle = violations_to_peak == 0
                        
                        # Score normalized regression slope to peak (0-2 points)
                        # Positive slopes are good, very steep or negative are bad
                        if 0.05 <= normalized_slope_to_peak <= 0.3:
                            slope_to_peak_score = 2
                        elif (0.02 <= normalized_slope_to_peak < 0.05) or (0.3 < normalized_slope_to_peak <= 0.5):
                            slope_to_peak_score = 1
                        else:
                            slope_to_peak_score = 0
                
                # Calculate linear regression slope from peak
                if peak_position_in_nonzero < len(non_zero_positions) - 1:
                    # Get x (bin positions) and y (counts) for regression
                    x_from_peak = np.array(non_zero_positions[peak_position_in_nonzero:])
                    y_from_peak = np.array([counts[i] for i in x_from_peak])
                    
                    if len(x_from_peak) >= 2:
                        # Calculate linear regression slope
                        regression_slope_from_peak, _ = np.polyfit(x_from_peak, y_from_peak, 1)
                        
                        # Normalize by peak count
                        normalized_slope_from_peak = regression_slope_from_peak / max_count if max_count > 0 else 0
                        
                        # Check for step-by-step violations (for wiggle room assessment)
                        violations_from_peak = 0
                        for i in range(peak_position_in_nonzero, len(non_zero_positions) - 1):
                            current_idx = non_zero_positions[i]
                            next_idx = non_zero_positions[i + 1]
                            step_change = counts[next_idx] - counts[current_idx]
                            
                            # Check for violations (increases larger than wiggle threshold)
                            if step_change > wiggle_threshold:
                                violations_from_peak += 1
                        
                        decreasing_from_peak_with_wiggle = violations_from_peak == 0
                        
                        # Score normalized regression slope from peak (0-2 points)
                        # Negative slopes are good, very steep or positive are bad
                        if -0.3 <= normalized_slope_from_peak <= -0.05:
                            slope_from_peak_score = 2
                        elif (-0.5 <= normalized_slope_from_peak < -0.3) or (-0.05 < normalized_slope_from_peak <= -0.02):
                            slope_from_peak_score = 1
                        else:
                            slope_from_peak_score = 0
        
        # 4. Calculate plateau score (0-1 points)
        if plateau_length == 1:
            plateau_score = 1  # No problematic plateau
        elif plateau_length == 2:
            plateau_score = 0.5  # Acceptable short plateau
        else:  # plateau_length >= 3
            plateau_score = 0  # Suspicious plateau
        
        # Calculate total authenticity score (0-11)
        authenticity_score = (
            bins_score +  # 0-2 points
            dominance_score +  # 0-2 points
            slope_to_peak_score +  # 0-2 points
            slope_from_peak_score +  # 0-2 points
            (1 if increasing_to_peak_with_wiggle else 0) +  # 0-1 points
            (1 if decreasing_from_peak_with_wiggle else 0) +  # 0-1 points
            plateau_score  # 0-1 points
        )
        
        # Assessment categories (updated thresholds for 11-point scale)
        if authenticity_score >= 9:
            assessment = "HIGHLY AUTHENTIC"
        elif authenticity_score >= 7:
            assessment = "PROBABLY AUTHENTIC"
        elif authenticity_score >= 5:
            assessment = "SUSPICIOUS"
        else:
            assessment = "LIKELY FABRICATED"
        
        # Create risk factors list
        risk_factors = []
        if bins_score == 0:
            risk_factors.append("Too few bins with data")
        if dominance_score == 0:
            risk_factors.append("Extreme peak concentration")
        if slope_to_peak_score == 0 and normalized_slope_to_peak is not None:
            risk_factors.append("Unnatural normalized slope to peak")
        if slope_from_peak_score == 0 and normalized_slope_from_peak is not None:
            risk_factors.append("Unnatural normalized slope from peak")
        if not increasing_to_peak_with_wiggle and normalized_slope_to_peak is not None:
            risk_factors.append("Not increasing to peak")
        if not decreasing_from_peak_with_wiggle and normalized_slope_from_peak is not None:
            risk_factors.append("Not decreasing from peak")
        if plateau_length >= 3:
            risk_factors.append("Suspicious plateau detected")
        
        # Calculate flag_problematic
        flag_problematic = (
            (increasing_to_peak_with_wiggle is False) or
            (decreasing_from_peak_with_wiggle is False) or
            (bins_with_data <= 4) or
            (slope_to_peak_score == 0) or
            (slope_from_peak_score == 0) or
            (plateau_length >= 3)
        )
        
        return {
            # Simplified indicators
            'bins_with_data': bins_with_data,
            'peak_dominance_pct': round(peak_dominance_pct, 1),
            'regression_slope_to_peak': round(regression_slope_to_peak, 2) if regression_slope_to_peak is not None else None,
            'regression_slope_from_peak': round(regression_slope_from_peak, 2) if regression_slope_from_peak is not None else None,
            'normalized_slope_to_peak': round(normalized_slope_to_peak, 3) if normalized_slope_to_peak is not None else None,
            'normalized_slope_from_peak': round(normalized_slope_from_peak, 3) if normalized_slope_from_peak is not None else None,
            'increasing_to_peak_with_wiggle': increasing_to_peak_with_wiggle,
            'decreasing_from_peak_with_wiggle': decreasing_from_peak_with_wiggle,
            'plateau_length': plateau_length,
            
            # Scoring details
            'bins_score': bins_score,
            'dominance_score': dominance_score,
            'slope_to_peak_score': slope_to_peak_score,
            'slope_from_peak_score': slope_from_peak_score,
            'increasing_score': 1 if increasing_to_peak_with_wiggle else 0,
            'decreasing_score': 1 if decreasing_from_peak_with_wiggle else 0,
            'plateau_score': plateau_score,
            
            # Overall assessment
            'authenticity_score': authenticity_score,
            'max_authenticity_score': 11,
            'assessment': assessment,
            'risk_factors': '; '.join(risk_factors) if risk_factors else 'None',
            'data_sufficiency': 'SUFFICIENT',
            'flag_problematic': flag_problematic
        }
    
    def _get_insufficient_data_features(self, actual_count, required_count):
        """Return features for insufficient data cases"""
        return {
            'bins_with_data': None,
            'peak_dominance_pct': None,
            'regression_slope_to_peak': None,
            'regression_slope_from_peak': None,
            'normalized_slope_to_peak': None,
            'normalized_slope_from_peak': None,
            'increasing_to_peak_with_wiggle': None,
            'decreasing_from_peak_with_wiggle': None,
            'plateau_length': None,
            'bins_score': None,
            'dominance_score': None,
            'slope_to_peak_score': None,
            'slope_from_peak_score': None,
            'increasing_score': None,
            'decreasing_score': None,
            'plateau_score': None,
            'authenticity_score': None,
            'max_authenticity_score': 11,
            'assessment': 'INSUFFICIENT DATA',
            'risk_factors': f'Only {actual_count} measurements, need {required_count}',
            'data_sufficiency': 'INSUFFICIENT',
            'flag_problematic': None
        }
    
    def _get_empty_fraud_features(self):
        """Return default fraud features for empty data"""
        return {
            'bins_with_data': 0,
            'peak_dominance_pct': None,
            'regression_slope_to_peak': None,
            'regression_slope_from_peak': None,
            'normalized_slope_to_peak': None,
            'normalized_slope_from_peak': None,
            'increasing_to_peak_with_wiggle': None,
            'decreasing_from_peak_with_wiggle': None,
            'plateau_length': None,
            'bins_score': None,
            'dominance_score': None,
            'slope_to_peak_score': None,
            'slope_from_peak_score': None,
            'increasing_score': None,
            'decreasing_score': None,
            'plateau_score': None,
            'authenticity_score': None,
            'max_authenticity_score': 11,
            'assessment': 'NO DATA',
            'risk_factors': 'No MUAC data',
            'data_sufficiency': 'NO_DATA',
            'flag_problematic': None
        }
    
    def _calculate_additional_features(self, binnable_muac_data, min_threshold=100):
        """Calculate additional features: standard deviation, std_miss, and KS test"""
        
        features = {
            'muac_std': None,
            'std_miss': None,
            'ks_test_p_value': None
        }
        
        if len(binnable_muac_data) == 0:
            return features
        
        # Calculate standard deviation
        muac_std = binnable_muac_data.std()
        features['muac_std'] = round(muac_std, 3) if pd.notna(muac_std) else None
        
        # Calculate std_miss (distance from ideal range 1.05-1.44)
        if pd.notna(muac_std):
            if 1.05 <= muac_std <= 1.44:
                features['std_miss'] = 0.0
            elif muac_std < 1.05:
                features['std_miss'] = round(abs(muac_std - 1.05), 3)
            else:  # muac_std > 1.44
                features['std_miss'] = round(abs(muac_std - 1.44), 3)
        
        # Calculate KS test p-value (only if sufficient data)
        if len(binnable_muac_data) >= min_threshold:
            try:
                # Test against normal distribution with sample mean and std
                sample_mean = binnable_muac_data.mean()
                sample_std = binnable_muac_data.std()
                ks_stat, ks_p_value = kstest(binnable_muac_data, 
                                           lambda x: stats.norm.cdf(x, sample_mean, sample_std))
                features['ks_test_p_value'] = round(ks_p_value, 4)
            except Exception as e:
                features['ks_test_p_value'] = None
        
        return features
    
    def create_flw_muac_analysis(self):
        """Create enhanced FLW MUAC analysis with simplified fraud detection"""
        
        # Auto-detect required columns
        flw_id_col = self._detect_column(['flw_id', 'flw id', 'worker_id'])
        flw_name_col = self._detect_column(['flw_name', 'flw name', 'worker_name', 'field_worker_name'])
        opportunity_col = self._detect_column(['opportunity_name', 'opportunity', 'org'])
        
        if not flw_id_col:
            self.log("Warning: FLW ID column not found for FLW MUAC analysis")
            return None
        
        if self.muac_col not in self.df.columns:
            self.log(f"Warning: MUAC measurement column '{self.muac_col}' not found for FLW MUAC analysis")
            return None
        
        # Prepare data
        df_clean = self.df.copy()
        
        # Process each FLW
        flw_results = []
        
        for flw_id in df_clean[flw_id_col].unique():
            if pd.isna(flw_id):
                continue
            
            flw_data = df_clean[df_clean[flw_id_col] == flw_id]
            
            # Get FLW name and opportunity
            flw_name = None
            if flw_name_col and flw_name_col in df_clean.columns:
                flw_name_values = flw_data[flw_name_col].dropna()
                if len(flw_name_values) > 0:
                    flw_name = flw_name_values.mode().iloc[0] if len(flw_name_values.mode()) > 0 else flw_name_values.iloc[0]
            
            opp_name = None
            if opportunity_col and opportunity_col in df_clean.columns:
                opp_name_values = flw_data[opportunity_col].dropna()
                if len(opp_name_values) > 0:
                    opp_name = opp_name_values.mode().iloc[0] if len(opp_name_values.mode()) > 0 else opp_name_values.iloc[0]
            
            # Calculate basic metrics
            total_visits = len(flw_data)
            missing_muac_visits = flw_data[self.muac_col].isna().sum()
            valid_muac_data = flw_data[self.muac_col].dropna()
            invalid_muac_visits = ((valid_muac_data < 9.5) | (valid_muac_data > 21.5)).sum()
            binnable_muac_data = valid_muac_data[(valid_muac_data >= 9.5) & (valid_muac_data <= 21.5)]
            
            # Calculate basic statistics (simplified)
            muac_mean = round(binnable_muac_data.mean(), 2) if len(binnable_muac_data) > 0 else None
            muac_median = round(binnable_muac_data.median(), 2) if len(binnable_muac_data) > 0 else None
            
            # Calculate bin counts for fraud detection
            bin_counts = []
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = ((binnable_muac_data >= min_muac) & (binnable_muac_data < max_muac)).sum()
                bin_counts.append(bin_visits)
            
            # Calculate simplified fraud detection features
            fraud_features = self._calculate_simplified_fraud_features(bin_counts)
            
            # Calculate additional features (std dev, std_miss, KS test)
            additional_features = self._calculate_additional_features(binnable_muac_data)
            
            # Build result row
            result_row = {
                'flw_id': flw_id,
                'total_visits': total_visits
            }
            
            if flw_name:
                result_row['flw_name'] = flw_name
            if opp_name:
                result_row['opportunity_name'] = opp_name
            
            # Basic MUAC metrics
            result_row['missing_muac_visits'] = missing_muac_visits
            result_row['invalid_muac_visits'] = invalid_muac_visits
            result_row['valid_muac_count'] = len(binnable_muac_data)
            result_row['muac_mean'] = muac_mean
            result_row['muac_median'] = muac_median
            
            # Bin counts (absolute numbers only, no percentages)
            for i, (min_muac, max_muac, bin_name) in enumerate(self.muac_bins):
                result_row[f'{bin_name}_visits'] = bin_counts[i]
            
            # Add all simplified fraud detection features
            result_row.update(fraud_features)
            
            # Add additional features
            result_row.update(additional_features)
            
            flw_results.append(result_row)
        
        if not flw_results:
            return None
        
        # Convert to DataFrame and sort by authenticity score ascending (lowest first for easier problem identification)
        flw_muac_df = pd.DataFrame(flw_results)
        # Sort by authenticity_score ascending (lowest first), then by total_visits descending for ties
        flw_muac_df = flw_muac_df.sort_values(['authenticity_score', 'total_visits'], ascending=[True, False])
        
        self.log(f"Enhanced FLW MUAC analysis complete: {len(flw_muac_df)} FLWs analyzed (sorted by authenticity score, lowest first)")
        return flw_muac_df
    
    def create_opps_muac_analysis(self):
        """Create enhanced Opps MUAC analysis with simplified fraud detection"""
        
        # Auto-detect required columns
        opportunity_col = self._detect_column(['opportunity_name', 'opportunity', 'org'])
        
        if not opportunity_col:
            self.log("Warning: Opportunity column not found for Opps MUAC analysis")
            return None
        
        if self.muac_col not in self.df.columns:
            self.log(f"Warning: MUAC measurement column '{self.muac_col}' not found for Opps MUAC analysis")
            return None
        
        # Similar implementation as FLW analysis but grouped by opportunity
        df_clean = self.df.copy()
        opp_results = []
        
        for opp_name in df_clean[opportunity_col].unique():
            if pd.isna(opp_name):
                continue
            
            opp_data = df_clean[df_clean[opportunity_col] == opp_name]
            
            # Calculate metrics (similar to FLW analysis)
            total_visits = len(opp_data)
            missing_muac_visits = opp_data[self.muac_col].isna().sum()
            valid_muac_data = opp_data[self.muac_col].dropna()
            invalid_muac_visits = ((valid_muac_data < 9.5) | (valid_muac_data > 21.5)).sum()
            binnable_muac_data = valid_muac_data[(valid_muac_data >= 9.5) & (valid_muac_data <= 21.5)]
            
            # Basic statistics
            muac_mean = round(binnable_muac_data.mean(), 2) if len(binnable_muac_data) > 0 else None
            muac_median = round(binnable_muac_data.median(), 2) if len(binnable_muac_data) > 0 else None
            
            # Calculate bin counts
            bin_counts = []
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = ((binnable_muac_data >= min_muac) & (binnable_muac_data < max_muac)).sum()
                bin_counts.append(bin_visits)
            
            fraud_features = self._calculate_simplified_fraud_features(bin_counts)
            
            # Calculate additional features (std dev, std_miss, KS test)
            additional_features = self._calculate_additional_features(binnable_muac_data)
            
            # Build result row
            result_row = {
                'opportunity_name': opp_name,
                'total_visits': total_visits,
                'missing_muac_visits': missing_muac_visits,
                'invalid_muac_visits': invalid_muac_visits,
                'valid_muac_count': len(binnable_muac_data),
                'muac_mean': muac_mean,
                'muac_median': muac_median
            }
            
            # Add bin data (absolute numbers only)
            for i, (min_muac, max_muac, bin_name) in enumerate(self.muac_bins):
                result_row[f'{bin_name}_visits'] = bin_counts[i]
            
            # Add fraud features
            result_row.update(fraud_features)
            
            # Add additional features
            result_row.update(additional_features)
            
            opp_results.append(result_row)
        
        if not opp_results:
            return None
        
        # Convert to DataFrame and sort by authenticity score ascending (lowest first for easier problem identification)
        opps_muac_df = pd.DataFrame(opp_results)
        # Sort by authenticity_score ascending (lowest first), then by total_visits descending for ties
        opps_muac_df = opps_muac_df.sort_values(['authenticity_score', 'total_visits'], ascending=[True, False])
        
        self.log(f"Enhanced Opps MUAC analysis complete: {len(opps_muac_df)} opportunities analyzed (sorted by authenticity score, lowest first)")
        return opps_muac_df
    
    def export_detailed_results(self, flw_df=None, opps_df=None, output_dir="./muac_analysis_results/"):
        """Export detailed results including simplified fraud detection features"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        if flw_df is not None:
            flw_path = os.path.join(output_dir, "flw_muac_analysis_with_fraud_detection.csv")
            flw_df.to_csv(flw_path, index=False)
            self.log(f"FLW analysis exported to: {flw_path}")
            results['flw_csv'] = flw_path
            
            # Try to generate sparkline visualization automatically
            try:
                sparkline_path = self._generate_sparkline_visualization(flw_df, output_dir)
                if sparkline_path:
                    results['sparkline_png'] = sparkline_path
                    self.log(f"Sparkline visualization generated: {sparkline_path}")
            except Exception as e:
                self.log(f"Note: Could not generate sparkline visualization: {str(e)}")
        
        if opps_df is not None:
            opps_path = os.path.join(output_dir, "opps_muac_analysis_with_fraud_detection.csv")
            opps_df.to_csv(opps_path, index=False)
            self.log(f"Opportunities analysis exported to: {opps_path}")
            results['opps_csv'] = opps_path
        
        return results
    
    def _generate_sparkline_visualization(self, flw_df, output_dir):
        """Generate sparkline visualization using the MUAC sparkline generator"""
        try:
            # Try to import the sparkline generator
            try:
                from muac_sparkline_generator import MUACSparklineReport
            except ImportError:
                self.log("Sparkline generator not found (muac_sparkline_generator.py)")
                return None
            
            # Create a mock log function and params frame for the sparkline generator
            def mock_log(message):
                self.log(f"Sparkline: {message}")
            
            # Create sparkline generator with default parameters
            sparkline_gen = MUACSparklineReport(
                df=flw_df,
                output_dir=output_dir, 
                log_function=mock_log,
                params_frame=None  # We'll set parameters manually
            )
            
            # Set default parameters manually (since we don't have the UI)
            sparkline_gen.grid_cols = 15
            sparkline_gen.image_dpi = 300
            sparkline_gen.include_labels = False
            sparkline_gen.use_custom_ranges = True
            
            # Generate the sparkline visualization
            output_files = sparkline_gen.generate()
            
            return output_files[0] if output_files else None
            
        except Exception as e:
            self.log(f"Error in sparkline generation: {str(e)}")
            return None


# Example usage function
def run_enhanced_muac_analysis(df, log_func=print):
    """
    Example function showing how to use the simplified analyzer
    
    Args:
        df: DataFrame with MUAC data
        log_func: Logging function (default: print)
    
    Returns:
        Dictionary with analysis results and file paths
    """
    
    # Initialize analyzer
    analyzer = EnhancedFLWMUACAnalyzer(df, log_func)
    
    # Run analyses
    log_func("Starting simplified MUAC analysis with fraud detection...")
    
    flw_results = analyzer.create_flw_muac_analysis()
    opps_results = analyzer.create_opps_muac_analysis()
    
    # Export results
    file_paths = analyzer.export_detailed_results(flw_results, opps_results)
    
    # Print summary
    if flw_results is not None:
        sufficient_flw = flw_results[flw_results['data_sufficiency'] == 'SUFFICIENT']
        print(f"\nFLW Analysis Summary:")
        print(f"Total FLWs analyzed: {len(flw_results)}")
        print(f"With sufficient data: {len(sufficient_flw)}")
        if len(sufficient_flw) > 0:
            print(f"Highly authentic (9-11): {len(sufficient_flw[sufficient_flw['authenticity_score'] >= 9])}")
            print(f"Suspicious/Fabricated (0-6): {len(sufficient_flw[sufficient_flw['authenticity_score'] <= 6])}")
            print(f"Average authenticity score: {sufficient_flw['authenticity_score'].mean():.1f}/11")
            print(f"Flagged as problematic: {sufficient_flw['flag_problematic'].sum()}")
            
            # Show plateau analysis
            plateau_data = sufficient_flw['plateau_length'].dropna()
            if len(plateau_data) > 0:
                plateau_3_plus = (plateau_data >= 3).sum()
                print(f"FLWs with suspicious plateaus (=3): {plateau_3_plus}")
    
    if opps_results is not None:
        sufficient_opps = opps_results[opps_results['data_sufficiency'] == 'SUFFICIENT']
        print(f"\nOpportunity Analysis Summary:")
        print(f"Total opportunities analyzed: {len(opps_results)}")
        print(f"With sufficient data: {len(sufficient_opps)}")
        if len(sufficient_opps) > 0:
            print(f"Highly authentic (9-11): {len(sufficient_opps[sufficient_opps['authenticity_score'] >= 9])}")
            print(f"Suspicious/Fabricated (0-6): {len(sufficient_opps[sufficient_opps['authenticity_score'] <= 6])}")
            print(f"Average authenticity score: {sufficient_opps['authenticity_score'].mean():.1f}/11")
            print(f"Flagged as problematic: {sufficient_opps['flag_problematic'].sum()}")
            
            # Show plateau analysis
            plateau_data = sufficient_opps['plateau_length'].dropna()
            if len(plateau_data) > 0:
                plateau_3_plus = (plateau_data >= 3).sum()
                print(f"Opportunities with suspicious plateaus (=3): {plateau_3_plus}")
    
    return {
        'flw_analysis': flw_results,
        'opps_analysis': opps_results,
        'file_paths': file_paths,
        'analyzer': analyzer
    }


if __name__ == "__main__":
    # Example usage
    print("Enhanced FLW MUAC Analyzer with Simplified Fraud Detection")
    print("=" * 60)
    print("This module provides comprehensive MUAC distribution analysis")
    print("with streamlined fraud detection capabilities.")
    print("\nNew Simplified Features:")
    print("- 7 key indicators (5 quantitative, 2 boolean)")
    print("- 0-11 point scoring system")
    print("- Plateau detection for suspicious data patterns")
    print("- Normalized regression slopes (relative to peak)")
    print("- Slope analysis with wiggle room (2% tolerance)")
    print("- Cleaner output (no percentage columns)")
    print("- Simplified statistical measures")
    print("\nNew Additional Features:")
    print("- plateau_length: Length of longest plateau (1 = no plateau)")
    print("- flag_problematic: Composite boolean flag for issues")
    print("- muac_std: Standard deviation of MUAC measurements")
    print("- std_miss: Distance from ideal std range [1.05, 1.44]")
    print("- ks_test_p_value: Kolmogorov-Smirnov test for normality")
    print("\nScoring System:")
    print("- Bins with data: 0-2 points")
    print("- Peak dominance: 0-2 points") 
    print("- Normalized slope to peak: 0-2 points")
    print("- Normalized slope from peak: 0-2 points")
    print("- Increasing to peak (with wiggle): 0-1 point")
    print("- Decreasing from peak (with wiggle): 0-1 point")
    print("- Plateau detection: 0-1 point")
    print("\nPlateau Detection:")
    print("- Only considers bins with =50% of peak count")
    print("- Tolerance: 4% of total measurements")
    print("- Length 1: No plateau (1 point)")
    print("- Length 2: Acceptable (0.5 points)")
    print("- Length =3: Suspicious (0 points, flagged as risk)")
    print("\nNormalized Slope Thresholds:")
    print("- To peak: 0.05-0.3 (ideal), 0.02-0.05 or 0.3-0.5 (acceptable)")
    print("- From peak: -0.3 to -0.05 (ideal), -0.5 to -0.3 or -0.05 to -0.02 (acceptable)")
    print("\nAssessment Categories:")
    print("- 9-11: Highly Authentic")
    print("- 7-8: Probably Authentic")
    print("- 5-6: Suspicious")
    print("- 0-4: Likely Fabricated")
    print("\nTo use: analyzer = EnhancedFLWMUACAnalyzer(your_dataframe, print)")
    print("Then: results = analyzer.create_flw_muac_analysis()")
    print("\nExample usage:")
    print("import pandas as pd")
    print("df = pd.read_csv('your_muac_data.csv')")
    print("results = run_enhanced_muac_analysis(df)")
    print("flw_analysis = results['flw_analysis']")
    print("print(flw_analysis[['flw_id', 'plateau_length', 'authenticity_score', 'assessment']].head())")
