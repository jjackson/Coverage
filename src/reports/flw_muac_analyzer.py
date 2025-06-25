"""
FLW MUAC Analysis - MUAC measurement distribution analysis
Analyzes MUAC measurements in 1cm bins on half-centimeters with normality testing

Save this file as: src/reports/flw_muac_analyzer.py
"""

import pandas as pd
import numpy as np
from scipy import stats


class FLWMUACAnalyzer:
    """Analyzes MUAC measurement distributions for FLWs and opportunities"""
    
    def __init__(self, df, log_func):
        self.df = df
        self.log = log_func
        self.muac_col = 'muac_measurement_cm'
        self.normality_threshold = 300  # Minimum visits for normality testing
        
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
    
    def _assess_normality(self, data):
        """Assess normality of MUAC data using Shapiro-Wilk test and descriptive stats"""
        
        if len(data) < self.normality_threshold:
            return {
                'test_pvalue': None,
                'normality_result': 'Insufficient data',
                'skewness': None,
                'kurtosis': None,
                'mean': None,
                'median': None,
                'std': None,
                'valid_count': len(data)
            }
        
        # Calculate descriptive statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skewness_val = stats.skew(data)
        kurtosis_val = stats.kurtosis(data, fisher=False)  # Pearson's kurtosis (normal = 3)
        
        # Perform Shapiro-Wilk test
        try:
            # Shapiro-Wilk works best with samples <= 5000
            if len(data) <= 5000:
                shapiro_stat, shapiro_pvalue = stats.shapiro(data)
            else:
                # Use Anderson-Darling for larger samples
                ad_result = stats.anderson(data, dist='norm')
                # Convert to approximate p-value
                shapiro_pvalue = 1.0 if ad_result.statistic < ad_result.critical_values[2] else 0.01
        except Exception:
            shapiro_pvalue = None
        
        # Determine normality result
        if shapiro_pvalue is None:
            normality_result = 'Test failed'
        elif shapiro_pvalue > 0.05:
            normality_result = 'Normal'
        else:
            normality_result = 'Non-normal'
        
        return {
            'test_pvalue': round(shapiro_pvalue, 4) if shapiro_pvalue is not None else None,
            'normality_result': normality_result,
            'skewness': round(skewness_val, 4),
            'kurtosis': round(kurtosis_val, 4),
            'mean': round(mean_val, 2),
            'median': round(median_val, 2),
            'std': round(std_val, 2),
            'valid_count': len(data)
        }
        """Detect column name from a list of possibilities"""
        for name in possible_names:
            matching_cols = [col for col in self.df.columns if name.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        return None
    
    def create_flw_muac_analysis(self):
        """Create FLW MUAC analysis with MUAC measurement distribution"""
        
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
            
            # Get FLW name if available
            flw_name = None
            if flw_name_col and flw_name_col in df_clean.columns:
                flw_name_values = flw_data[flw_name_col].dropna()
                if len(flw_name_values) > 0:
                    flw_name = flw_name_values.mode().iloc[0] if len(flw_name_values.mode()) > 0 else flw_name_values.iloc[0]
            
            # Get opportunity name if available
            opp_name = None
            if opportunity_col and opportunity_col in df_clean.columns:
                opp_name_values = flw_data[opportunity_col].dropna()
                if len(opp_name_values) > 0:
                    opp_name = opp_name_values.mode().iloc[0] if len(opp_name_values.mode()) > 0 else opp_name_values.iloc[0]
            
            # Calculate metrics for this FLW
            total_visits = len(flw_data)
            
            # Count missing MUAC values
            missing_muac_visits = flw_data[self.muac_col].isna().sum()
            
            # Get valid MUAC data
            valid_muac_data = flw_data[self.muac_col].dropna()
            
            # Count invalid MUAC values (outside reasonable range)
            invalid_muac_visits = ((valid_muac_data < 9.5) | (valid_muac_data > 21.5)).sum()
            
            # Filter to valid range for binning
            binnable_muac_data = valid_muac_data[(valid_muac_data >= 9.5) & (valid_muac_data <= 21.5)]
            
            # Perform normality assessment
            normality_results = self._assess_normality(binnable_muac_data)
            
            result_row = {
                'flw_id': flw_id,
                'total_visits': total_visits
            }
            
            if flw_name:
                result_row['flw_name'] = flw_name
            
            if opp_name:
                result_row['opportunity_name'] = opp_name
            
            # Add missing and invalid counts first
            result_row['missing_muac_visits'] = missing_muac_visits
            result_row['invalid_muac_visits'] = invalid_muac_visits
            
            # Count visits in each MUAC bin - visits first
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = ((binnable_muac_data >= min_muac) & (binnable_muac_data < max_muac)).sum()
                result_row[f'{bin_name}_visits'] = bin_visits
            
            # Add normality assessment results
            result_row['valid_muac_count'] = normality_results['valid_count']
            result_row['muac_normality_test_pvalue'] = normality_results['test_pvalue']
            result_row['muac_normality_result'] = normality_results['normality_result']
            result_row['muac_skewness'] = normality_results['skewness']
            result_row['muac_kurtosis'] = normality_results['kurtosis']
            result_row['muac_mean'] = normality_results['mean']
            result_row['muac_median'] = normality_results['median']
            result_row['muac_std'] = normality_results['std']
            
            # Then add percentages
            result_row['missing_muac_pct'] = round(missing_muac_visits / total_visits, 4) if total_visits > 0 else 0
            result_row['invalid_muac_pct'] = round(invalid_muac_visits / total_visits, 4) if total_visits > 0 else 0
            
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = result_row[f'{bin_name}_visits']
                result_row[f'{bin_name}_pct'] = round(bin_visits / total_visits, 4) if total_visits > 0 else 0
            
            flw_results.append(result_row)
        
        if not flw_results:
            return None
        
        # Convert to DataFrame and sort by total visits descending
        flw_muac_df = pd.DataFrame(flw_results)
        flw_muac_df = flw_muac_df.sort_values('total_visits', ascending=False)
        
        self.log(f"FLW MUAC analysis complete: {len(flw_muac_df)} FLWs analyzed for MUAC measurement distribution")
        return flw_muac_df
    
    def create_opps_muac_analysis(self):
        """Create Opps MUAC analysis with MUAC measurement distribution"""
        
        # Auto-detect required columns
        opportunity_col = self._detect_column(['opportunity_name', 'opportunity', 'org'])
        
        if not opportunity_col:
            self.log("Warning: Opportunity column not found for Opps MUAC analysis")
            return None
        
        if self.muac_col not in self.df.columns:
            self.log(f"Warning: MUAC measurement column '{self.muac_col}' not found for Opps MUAC analysis")
            return None
        
        # Prepare data
        df_clean = self.df.copy()
        
        # Process each opportunity
        opp_results = []
        
        for opp_name in df_clean[opportunity_col].unique():
            if pd.isna(opp_name):
                continue
            
            opp_data = df_clean[df_clean[opportunity_col] == opp_name]
            
            # Calculate metrics for this opportunity
            total_visits = len(opp_data)
            
            # Count missing MUAC values
            missing_muac_visits = opp_data[self.muac_col].isna().sum()
            
            # Get valid MUAC data
            valid_muac_data = opp_data[self.muac_col].dropna()
            
            # Count invalid MUAC values (outside reasonable range)
            invalid_muac_visits = ((valid_muac_data < 9.5) | (valid_muac_data > 21.5)).sum()
            
            # Filter to valid range for binning
            binnable_muac_data = valid_muac_data[(valid_muac_data >= 9.5) & (valid_muac_data <= 21.5)]
            
            # Perform normality assessment
            normality_results = self._assess_normality(binnable_muac_data)
            
            result_row = {
                'opportunity_name': opp_name,
                'total_visits': total_visits
            }
            
            # Add missing and invalid counts first
            result_row['missing_muac_visits'] = missing_muac_visits
            result_row['invalid_muac_visits'] = invalid_muac_visits
            
            # Count visits in each MUAC bin - visits first
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = ((binnable_muac_data >= min_muac) & (binnable_muac_data < max_muac)).sum()
                result_row[f'{bin_name}_visits'] = bin_visits
            
            # Add normality assessment results
            result_row['valid_muac_count'] = normality_results['valid_count']
            result_row['muac_normality_test_pvalue'] = normality_results['test_pvalue']
            result_row['muac_normality_result'] = normality_results['normality_result']
            result_row['muac_skewness'] = normality_results['skewness']
            result_row['muac_kurtosis'] = normality_results['kurtosis']
            result_row['muac_mean'] = normality_results['mean']
            result_row['muac_median'] = normality_results['median']
            result_row['muac_std'] = normality_results['std']
            
            # Then add percentages
            result_row['missing_muac_pct'] = round(missing_muac_visits / total_visits, 4) if total_visits > 0 else 0
            result_row['invalid_muac_pct'] = round(invalid_muac_visits / total_visits, 4) if total_visits > 0 else 0
            
            for min_muac, max_muac, bin_name in self.muac_bins:
                bin_visits = result_row[f'{bin_name}_visits']
                result_row[f'{bin_name}_pct'] = round(bin_visits / total_visits, 4) if total_visits > 0 else 0
            
            opp_results.append(result_row)
        
        if not opp_results:
            return None
        
        # Convert to DataFrame and sort by total visits descending
        opps_muac_df = pd.DataFrame(opp_results)
        opps_muac_df = opps_muac_df.sort_values('total_visits', ascending=False)
        
        self.log(f"Opps MUAC analysis complete: {len(opps_muac_df)} opportunities analyzed for MUAC measurement distribution")
        return opps_muac_df
