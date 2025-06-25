"""
FLW Age Analysis - Child age distribution analysis
Analyzes child ages in 6-month intervals

Save this file as: src/reports/flw_age_analyzer.py
"""

import pandas as pd
import numpy as np


class FLWAgeAnalyzer:
    """Analyzes child age distributions for FLWs and opportunities"""
    
    def __init__(self, df, log_func):
        self.df = df
        self.log = log_func
        
        # Define age brackets (10 six-month intervals)
        self.age_brackets = [
            (0, 5, 'age_0_5'),
            (6, 11, 'age_6_11'),
            (12, 17, 'age_12_17'),
            (18, 23, 'age_18_23'),
            (24, 29, 'age_24_29'),
            (30, 35, 'age_30_35'),
            (36, 41, 'age_36_41'),
            (42, 47, 'age_42_47'),
            (48, 53, 'age_48_53'),
            (54, 59, 'age_54_59')
        ]
    
    def _detect_column(self, possible_names):
        """Detect column name from a list of possibilities"""
        for name in possible_names:
            matching_cols = [col for col in self.df.columns if name.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        return None
    
    def create_flw_ages_analysis(self):
        """Create FLW Ages analysis with child age distribution"""
        
        # Auto-detect required columns
        flw_id_col = self._detect_column(['flw_id', 'flw id', 'worker_id'])
        flw_name_col = self._detect_column(['flw_name', 'flw name', 'worker_name', 'field_worker_name'])
        opportunity_col = self._detect_column(['opportunity_name', 'opportunity', 'org'])
        child_age_col = self._detect_column(['childs_age_in_month', 'child_age_months', 'childs_age_in_months', 'age_month', 'age_in_month'])
        
        if not flw_id_col:
            self.log("Warning: FLW ID column not found for FLW Ages analysis")
            return None
        
        if not child_age_col:
            self.log("Warning: Child age column not found for FLW Ages analysis")
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
            
            # Count missing age values
            missing_age_visits = flw_data[child_age_col].isna().sum()
            
            # Get valid age data
            valid_age_data = flw_data[child_age_col].dropna()
            valid_age_data = valid_age_data[(valid_age_data >= 0) & (valid_age_data <= 120)]  # Reasonable range
            
            result_row = {
                'flw_id': flw_id,
                'total_visits': total_visits
            }
            
            if flw_name:
                result_row['flw_name'] = flw_name
            
            if opp_name:
                result_row['opportunity_name'] = opp_name
            
            # Add missing count first
            result_row['missing_age_visits'] = missing_age_visits
            
            # Count visits in each age bracket - visits first
            for min_age, max_age, bracket_name in self.age_brackets:
                bracket_visits = ((valid_age_data >= min_age) & (valid_age_data <= max_age)).sum()
                result_row[f'{bracket_name}_visits'] = bracket_visits
            
            # Then add percentages
            result_row['missing_age_pct'] = round(missing_age_visits / total_visits, 4) if total_visits > 0 else 0
            
            for min_age, max_age, bracket_name in self.age_brackets:
                bracket_visits = result_row[f'{bracket_name}_visits']
                result_row[f'{bracket_name}_pct'] = round(bracket_visits / total_visits, 4) if total_visits > 0 else 0
            
            flw_results.append(result_row)
        
        if not flw_results:
            return None
        
        # Convert to DataFrame and sort by total visits descending
        flw_ages_df = pd.DataFrame(flw_results)
        flw_ages_df = flw_ages_df.sort_values('total_visits', ascending=False)
        
        self.log(f"FLW Ages analysis complete: {len(flw_ages_df)} FLWs analyzed for child age distribution")
        return flw_ages_df
    
    def create_opps_ages_analysis(self):
        """Create Opps Ages analysis with child age distribution"""
        
        # Auto-detect required columns
        opportunity_col = self._detect_column(['opportunity_name', 'opportunity', 'org'])
        child_age_col = self._detect_column(['childs_age_in_month', 'child_age_months', 'childs_age_in_months', 'age_month', 'age_in_month'])
        
        if not opportunity_col:
            self.log("Warning: Opportunity column not found for Opps Ages analysis")
            return None
        
        if not child_age_col:
            self.log("Warning: Child age column not found for Opps Ages analysis")
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
            
            # Count missing age values
            missing_age_visits = opp_data[child_age_col].isna().sum()
            
            # Get valid age data
            valid_age_data = opp_data[child_age_col].dropna()
            valid_age_data = valid_age_data[(valid_age_data >= 0) & (valid_age_data <= 120)]  # Reasonable range
            
            result_row = {
                'opportunity_name': opp_name,
                'total_visits': total_visits
            }
            
            # Add missing count first
            result_row['missing_age_visits'] = missing_age_visits
            
            # Count visits in each age bracket - visits first
            for min_age, max_age, bracket_name in self.age_brackets:
                bracket_visits = ((valid_age_data >= min_age) & (valid_age_data <= max_age)).sum()
                result_row[f'{bracket_name}_visits'] = bracket_visits
            
            # Then add percentages
            result_row['missing_age_pct'] = round(missing_age_visits / total_visits, 4) if total_visits > 0 else 0
            
            for min_age, max_age, bracket_name in self.age_brackets:
                bracket_visits = result_row[f'{bracket_name}_visits']
                result_row[f'{bracket_name}_pct'] = round(bracket_visits / total_visits, 4) if total_visits > 0 else 0
            
            opp_results.append(result_row)
        
        if not opp_results:
            return None
        
        # Convert to DataFrame and sort by total visits descending
        opps_ages_df = pd.DataFrame(opp_results)
        opps_ages_df = opps_ages_df.sort_values('total_visits', ascending=False)
        
        self.log(f"Opps Ages analysis complete: {len(opps_ages_df)} opportunities analyzed for child age distribution")
        return opps_ages_df