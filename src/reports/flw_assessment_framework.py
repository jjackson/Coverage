"""
FLW Assessment Framework - Updated Version
Implements the assessment pipeline for Front Line Worker analysis

Save this file as: src/reports/flw_assessment_framework.py
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class Assessment(ABC):
    """Base class for all assessments"""
    
    def __init__(self, name, description, required_columns):
        self.name = name
        self.description = description
        self.required_columns = required_columns
    
    @abstractmethod
    def assess(self, visits_df, threshold, population_stats=None):
        """
        Assess a group of visits for this FLW/opportunity pair
        
        Args:
            visits_df: DataFrame with visits for this FLW/opp pair
            threshold: Minimum number of valid values required
            population_stats: Overall population statistics (for comparative assessments)
            
        Returns:
            dict: {
                'result': 'strong_negative', 'insufficient_data', or 'no_assessment',
                'indicator_name': str,
                'indicator_value': float,
                'valid_count': int,
                'description': str
            }
        """
        pass


class GenderRatioAssessment(Assessment):
    """Gender ratio assessment using Wilson confidence intervals"""
    
    def __init__(self):
        super().__init__(
            name="gender_ratio",
            description="Evaluates whether the ratio of girls to boys follows expected patterns",
            required_columns=['childs_gender']
        )
    
    def assess(self, visits_df, threshold, population_stats=None):
        """Assess gender ratio using Wilson CI comparison"""
        
        # Check if required column exists
        if 'childs_gender' not in visits_df.columns:
            return self._no_assessment("childs_gender column not found")
        
        # Get valid gender data
        gender_data = visits_df['childs_gender'].dropna()
        gender_data = gender_data[gender_data.isin(['female_child', 'male_child'])]
        
        # Check minimum threshold
        if len(gender_data) < threshold:
            return self._insufficient_data(f"Insufficient valid gender data: {len(gender_data)} < {threshold}")
        
        # Calculate female ratio
        female_count = (gender_data == 'female_child').sum()
        total_count = len(gender_data)
        female_ratio = female_count / total_count
        
        # Need population stats for comparison
        if population_stats is None or 'gender_ratio' not in population_stats:
            return self._no_assessment("No population statistics available for comparison")
        
        pop_stats = population_stats['gender_ratio']
        
        # Calculate Wilson confidence interval for this FLW
        flw_ci = self._wilson_ci(female_count, total_count, confidence=0.99)
        
        # Compare with population CI
        pop_ci = (pop_stats['ci_lower'], pop_stats['ci_upper'])
        
        # Check if CIs don't overlap (strong_negative)
        if flw_ci[1] < pop_ci[0] or flw_ci[0] > pop_ci[1]:
            result = "strong_negative"
            description = f"Female ratio {female_ratio:.3f} is significantly different from population (CI: {flw_ci[0]:.3f}-{flw_ci[1]:.3f} vs pop: {pop_ci[0]:.3f}-{pop_ci[1]:.3f})"
        else:
            result = "."
            description = ""  # No description for normal cases
        
        return {
            'result': result,
            'indicator_name': 'female_child_ratio',
            'indicator_value': round(female_ratio, 4),
            'valid_count': total_count,
            'description': description
        }
    
    def _wilson_ci(self, successes, total, confidence=0.99):
        """Calculate Wilson score confidence interval"""
        if total == 0:
            return (0, 0)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / total
        
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        halfwidth = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        
        return (max(0, center - halfwidth), min(1, center + halfwidth))
    
    def _insufficient_data(self, reason):
        """Return insufficient_data result"""
        return {
            'result': 'insufficient_data',
            'indicator_name': 'female_child_ratio',
            'indicator_value': None,
            'valid_count': 0,
            'description': reason
        }
    
    def _no_assessment(self, reason):
        """Return no_assessment result"""
        return {
            'result': '.',
            'indicator_name': 'female_child_ratio',
            'indicator_value': None,
            'valid_count': 0,
            'description': ""  # Empty description for display as "."
        }


class LowRedMUACAssessment(Assessment):
    """Low red MUAC assessment using simple threshold"""
    
    def __init__(self):
        super().__init__(
            name="low_red_muac",
            description="Flags cases where red MUAC percentage is suspiciously low",
            required_columns=['muac_color']
        )
    
    def assess(self, visits_df, threshold, population_stats=None):
        """Assess red MUAC percentage"""
        
        # Check if required column exists
        if 'muac_color' not in visits_df.columns:
            return self._no_assessment("muac_color column not found")
        
        # Get valid MUAC data
        muac_data = visits_df['muac_color'].dropna()
        muac_data = muac_data[muac_data.isin(['Red', 'Yellow', 'Green'])]
        
        # Check minimum threshold
        if len(muac_data) < threshold:
            return self._insufficient_data(f"Insufficient valid MUAC data: {len(muac_data)} < {threshold}")
        
        # Calculate red percentage
        red_count = (muac_data == 'Red').sum()
        total_count = len(muac_data)
        red_ratio = red_count / total_count
        
        # Check threshold (1% = 0.01)
        if red_ratio < 0.01:
            result = "strong_negative"
            description = f"Red MUAC rate {red_ratio:.3f} ({red_ratio*100:.1f}%) is suspiciously low (<1%)"
        else:
            result = "."
            description = ""  # No description - not "within expected range"
        
        return {
            'result': result,
            'indicator_name': 'red_muac_percentage',
            'indicator_value': round(red_ratio, 4),
            'valid_count': total_count,
            'description': description
        }
    
    def _insufficient_data(self, reason):
        """Return insufficient_data result"""
        return {
            'result': 'insufficient_data',
            'indicator_name': 'red_muac_percentage',
            'indicator_value': None,
            'valid_count': 0,
            'description': reason
        }
    
    def _no_assessment(self, reason):
        """Return no_assessment result"""
        return {
            'result': '.',
            'indicator_name': 'red_muac_percentage',
            'indicator_value': None,
            'valid_count': 0,
            'description': ""  # Empty description for display as "."
        }


class LowYoungChildAssessment(Assessment):
    """Low young child assessment using simple threshold"""
    
    def __init__(self):
        super().__init__(
            name="low_young_child",
            description="Flags cases where percentage of children under 12 months is suspiciously low",
            required_columns=['childs_age_in_month']
        )
    
    def assess(self, visits_df, threshold, population_stats=None):
        """Assess young child percentage"""
        
        # Check if required column exists
        if 'childs_age_in_month' not in visits_df.columns:
            return self._no_assessment("childs_age_in_month column not found")
        
        # Get valid age data
        age_data = visits_df['childs_age_in_month'].dropna()
        # Filter reasonable age range (0-120 months = 0-10 years)
        age_data = age_data[(age_data >= 0) & (age_data <= 120)]
        
        # Check minimum threshold
        if len(age_data) < threshold:
            return self._insufficient_data(f"Insufficient valid age data: {len(age_data)} < {threshold}")
        
        # Calculate under 12 months percentage
        under_12_count = (age_data < 12).sum()
        total_count = len(age_data)
        under_12_ratio = under_12_count / total_count
        
        # Check threshold (5% = 0.05)
        if under_12_ratio < 0.05:
            result = "strong_negative"
            description = f"Under-12-months rate {under_12_ratio:.3f} ({under_12_ratio*100:.1f}%) is suspiciously low (<5%)"
        else:
            result = "."
            description = ""  # No description - not "within expected range"
        
        return {
            'result': result,
            'indicator_name': 'under_12_months_percentage',
            'indicator_value': round(under_12_ratio, 4),
            'valid_count': total_count,
            'description': description
        }
    
    def _insufficient_data(self, reason):
        """Return insufficient_data result"""
        return {
            'result': 'insufficient_data',
            'indicator_name': 'under_12_months_percentage',
            'indicator_value': None,
            'valid_count': 0,
            'description': reason
        }
    
    def _no_assessment(self, reason):
        """Return no_assessment result"""
        return {
            'result': 'no_assessment',
            'indicator_name': 'under_12_months_percentage',
            'indicator_value': None,
            'valid_count': 0,
            'description': ""  # Empty description for display as "."
        }


class AssessmentEngine:
    """Main engine for running assessments on FLW data"""
    
    def __init__(self, visit_threshold=300):
        self.visit_threshold = visit_threshold
        self.assessments = {
            'gender_ratio': GenderRatioAssessment(),
            'low_red_muac': LowRedMUACAssessment(),
            'low_young_child': LowYoungChildAssessment()
        }
        self.column_patterns = {
            'flw_id': ['flw_id', 'flw id', 'worker_id'],
            'opportunity_name': ['opportunity_name', 'opportunity', 'org'],
            'visit_date': ['visit_date', 'date', 'visit_time'],
            'status': ['Status', 'status', 'visit_status'],
            'childs_gender': ['childs_gender', 'gender', 'child_gender'],
            'childs_age_in_month': ['childs_age_in_month', 'age_month', 'age_in_month'],
            'muac_color': ['muac_color', 'muac_colour']
        }
    
    def auto_detect_columns(self, df):
        """Auto-detect required columns"""
        detected = {}
        missing = []
        
        for field, patterns in self.column_patterns.items():
            found = False
            for pattern in patterns:
                matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
                if matching_cols:
                    detected[field] = matching_cols[0]
                    found = True
                    break
            
            if not found:
                missing.append(field)
        
        return detected, missing
    
    def prepare_data(self, df, detected_columns):
        """Clean and prepare data for assessment"""
        
        # Rename columns to standard names
        rename_map = {v: k for k, v in detected_columns.items()}
        df_clean = df.rename(columns=rename_map).copy()
        
        # Filter to approved visits only
        if 'status' in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean[df_clean['status'].str.lower() == 'approved']
            after_count = len(df_clean)
            print(f"Filtered to approved visits: {before_count} ? {after_count}")
        
        # Parse visit dates
        if 'visit_date' in df_clean.columns:
            df_clean['visit_date'] = pd.to_datetime(df_clean['visit_date'], errors='coerce')
            df_clean = df_clean.dropna(subset=['visit_date'])
        
        # Split FLWs by opportunity (create flw_123_1, flw_123_2 etc.)
        df_clean = self._split_flws_by_opportunity(df_clean)
        
        return df_clean
    
    def _split_flws_by_opportunity(self, df):
        """Split FLWs who worked on multiple opportunities"""
        
        if 'opportunity_name' not in df.columns or 'flw_id' not in df.columns:
            return df
        
        # Find FLWs that worked on multiple opportunities
        flw_opp_counts = df.groupby('flw_id')['opportunity_name'].nunique()
        multi_opp_flws = flw_opp_counts[flw_opp_counts > 1].index
        
        if len(multi_opp_flws) == 0:
            print("No FLWs found working on multiple opportunities")
            return df
        
        print(f"Found {len(multi_opp_flws)} FLWs working on multiple opportunities")
        
        # Process the split
        split_rows = []
        
        for _, row in df.iterrows():
            flw_id = row['flw_id']
            
            if pd.isna(flw_id) or flw_id not in multi_opp_flws:
                # Single opportunity FLW - keep as is
                split_rows.append(row)
            else:
                # Multi-opportunity FLW - needs splitting
                flw_opportunities = df[df['flw_id'] == flw_id]['opportunity_name'].unique()
                flw_opportunities = [opp for opp in flw_opportunities if pd.notna(opp)]
                
                current_opp = row['opportunity_name']
                if pd.isna(current_opp):
                    continue
                
                try:
                    opp_index = list(flw_opportunities).index(current_opp) + 1
                except ValueError:
                    opp_index = 1
                
                # Create new FLW ID with suffix
                new_row = row.copy()
                new_row['flw_id'] = f"{flw_id}_{opp_index}"
                split_rows.append(new_row)
        
        result_df = pd.DataFrame(split_rows)
        
        original_flw_count = df['flw_id'].nunique()
        new_flw_count = result_df['flw_id'].nunique()
        print(f"FLW split complete: {original_flw_count} ? {new_flw_count} FLW entities")
        
        return result_df
    
    def calculate_population_stats(self, df):
        """Calculate population-level statistics for comparative assessments"""
        
        population_stats = {}
        
        # Gender ratio population stats
        if 'childs_gender' in df.columns:
            gender_data = df['childs_gender'].dropna()
            gender_data = gender_data[gender_data.isin(['female_child', 'male_child'])]
            
            if len(gender_data) > 0:
                female_count = (gender_data == 'female_child').sum()
                total_count = len(gender_data)
                female_ratio = female_count / total_count
                
                # Calculate Wilson CI for population
                z = stats.norm.ppf(0.995)  # 99% confidence
                p = female_ratio
                n = total_count
                
                denominator = 1 + z**2 / n
                center = (p + z**2 / (2 * n)) / denominator
                halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
                
                population_stats['gender_ratio'] = {
                    'female_ratio': female_ratio,
                    'ci_lower': max(0, center - halfwidth),
                    'ci_upper': min(1, center + halfwidth),
                    'total_count': total_count
                }
        
        return population_stats
    
    def run_assessments(self, df):
        """Run complete assessment pipeline"""
        
        print("=== FLW Assessment Pipeline ===")
        
        # Auto-detect columns
        print("Auto-detecting columns...")
        detected_columns, missing_columns = self.auto_detect_columns(df)
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        print(f"Detected columns: {detected_columns}")
        
        # Prepare data
        print("Preparing data...")
        df_clean = self.prepare_data(df, detected_columns)
        
        # Calculate population statistics
        print("Calculating population statistics...")
        population_stats = self.calculate_population_stats(df_clean)
        
        # Get FLW/opportunity pairs with sufficient visits
        print("Identifying FLW/opportunity pairs with sufficient visits...")
        flw_opp_visits = df_clean.groupby(['flw_id', 'opportunity_name']).size().reset_index(name='visit_count')
        eligible_pairs = flw_opp_visits[flw_opp_visits['visit_count'] >= self.visit_threshold]
        
        print(f"Found {len(eligible_pairs)} FLW/opportunity pairs with >= {self.visit_threshold} visits")
        
        # Run assessments for each eligible pair
        assessment_results = []
        
        for _, pair in eligible_pairs.iterrows():
            flw_id = pair['flw_id']
            opp_name = pair['opportunity_name']
            
            # Get this FLW's visits (most recent N)
            flw_visits = df_clean[
                (df_clean['flw_id'] == flw_id) & 
                (df_clean['opportunity_name'] == opp_name)
            ].sort_values('visit_date', ascending=False).head(self.visit_threshold)
            
            # Run each assessment
            flw_result = {
                'flw_id': flw_id,
                'opportunity_name': opp_name,
                'total_visits': len(flw_visits),
                'assessment_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            strong_negatives = 0
            insufficient_data_count = 0
            
            for assessment_name, assessment in self.assessments.items():
                result = assessment.assess(flw_visits, self.visit_threshold, population_stats)
                
                # Add assessment-specific columns
                flw_result[f"{result['indicator_name']}"] = result['indicator_value']
                flw_result[f"{result['indicator_name']}_result"] = result['result']
                flw_result[f"{result['indicator_name']}_valid_count"] = result['valid_count']
                
                # Handle description display
                if result['result'] == 'insufficient_data':
                    flw_result[f"{result['indicator_name']}_description"] = result['description']
                    insufficient_data_count += 1
                elif result['result'] == 'strong_negative':
                    flw_result[f"{result['indicator_name']}_description"] = result['description']
                    strong_negatives += 1
                else:
                    # no_assessment - show as "."
                    flw_result[f"{result['indicator_name']}_description"] = "."
            
            flw_result['num_strong_negative'] = strong_negatives
            flw_result['num_insufficient_data'] = insufficient_data_count
            flw_result['has_any_strong_negative'] = strong_negatives > 0
            flw_result['has_insufficient_data'] = insufficient_data_count > 0
            
            assessment_results.append(flw_result)
        
        results_df = pd.DataFrame(assessment_results)
        
        print(f"Assessment complete: {len(results_df)} FLW/opportunity pairs assessed")
        
        return results_df, population_stats
    
    def create_opportunity_summary(self, assessment_results_df, original_df):
        """Create opportunity-level summary"""
        
        if len(assessment_results_df) == 0:
            return pd.DataFrame()
        
        # Get all FLWs by opportunity from original data
        all_flws = original_df.groupby('opportunity_name')['flw_id'].nunique().reset_index()
        all_flws.columns = ['opportunity_name', 'total_flws']
        
        # Get FLWs with minimum visits
        min_visit_flws = assessment_results_df.groupby('opportunity_name').size().reset_index()
        min_visit_flws.columns = ['opportunity_name', 'flws_with_min_visits']
        
        # Get FLWs with strong negatives
        strong_neg_flws = assessment_results_df[
            assessment_results_df['has_any_strong_negative']
        ].groupby('opportunity_name').size().reset_index()
        strong_neg_flws.columns = ['opportunity_name', 'flws_with_strong_negative']
        
        # Get FLWs with insufficient data
        insufficient_data_flws = assessment_results_df[
            assessment_results_df['has_insufficient_data']
        ].groupby('opportunity_name').size().reset_index()
        insufficient_data_flws.columns = ['opportunity_name', 'flws_with_insufficient_data']
        
        # Combine all summaries
        opp_summary = all_flws.merge(min_visit_flws, on='opportunity_name', how='left')
        opp_summary = opp_summary.merge(strong_neg_flws, on='opportunity_name', how='left')
        opp_summary = opp_summary.merge(insufficient_data_flws, on='opportunity_name', how='left')
        
        # Fill NAs and calculate percentages
        opp_summary['flws_with_min_visits'] = opp_summary['flws_with_min_visits'].fillna(0)
        opp_summary['flws_with_strong_negative'] = opp_summary['flws_with_strong_negative'].fillna(0)
        opp_summary['flws_with_insufficient_data'] = opp_summary['flws_with_insufficient_data'].fillna(0)
        
        opp_summary['pct_flws_with_strong_negative'] = (
            opp_summary['flws_with_strong_negative'] / 
            opp_summary['flws_with_min_visits'] * 100
        ).round(1)
        
        opp_summary['pct_flws_with_insufficient_data'] = (
            opp_summary['flws_with_insufficient_data'] / 
            opp_summary['flws_with_min_visits'] * 100
        ).round(1)
        
        # Handle division by zero
        opp_summary['pct_flws_with_strong_negative'] = opp_summary['pct_flws_with_strong_negative'].fillna(0)
        opp_summary['pct_flws_with_insufficient_data'] = opp_summary['pct_flws_with_insufficient_data'].fillna(0)
        
        return opp_summary.sort_values('pct_flws_with_strong_negative', ascending=False)
