import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from .aggregation_functions import (
        gender_sva_score,
        yearly_age_imbalance_score,
        monthly_age_perfection_score,
        unwell_completion_anomaly_score,
        child_unwell_overreporting_score,
        completion_anomaly_score,
        yes_no_sva_score,	
         all_six_no_overreporting_score,
        muac_distribution_fraud_score  # NEW: Add the MUAC function
    )
    FRAUD_SCORING_AVAILABLE = True
except ImportError:
    FRAUD_SCORING_AVAILABLE = False
    print("WARNING: Fraud scoring functions not available")

try:
    from .field_utils import normalize_field_key, prepare_categorical_values, process_custom_binning
    FIELD_UTILS_AVAILABLE = True
except ImportError:
    FIELD_UTILS_AVAILABLE = False
    print("WARNING: field_utils not available")

# Fraud Scoring Algorithm Registry
FRAUD_SCORING_ALGORITHMS = {
    'gender_sva': {
        'weight': 12.0,
        'type': 'sva',
        'field': 'childs_gender',
        'description': 'Gender distribution deviation from baseline'
    },
    'yearly_age_imbalance': {
        'weight': 4.0,
        'type': 'custom',
        'function': yearly_age_imbalance_score,
        'description': 'Yearly age distribution imbalance'
    },
    'all_six_no_overreporting': {
        'weight': 8.0,  # High weight since you want strong detection
        'type': 'custom',
        'function': all_six_no_overreporting_score,
        'description': 'Detects unusually high rates of all_six_no = true responses'
    },
    'monthly_age_perfection': {
        'weight': 1.0,
        'type': 'custom', 
        'function': monthly_age_perfection_score,
        'description': 'Suspiciously perfect monthly age distribution'
    },
    'va_child_unwell_today_sva': {
        'weight': 1.0,
        'type': 'sva',
        'field': 'va_child_unwell_today',
        'description': 'Unwell response distribution deviation from baseline'
    },
    'va_child_unwell_today_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'va_child_unwell_today'),
        'description': 'Inconsistent completion of unwell field'
    },
    # New completion anomaly algorithms
    'glasses_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'have_glasses'),
        'description': 'Inconsistent completion of glasses field'
    },
    'diarrhea_completion_anomaly': {
        'weight': .1, 
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'diarrhea_last_month'),
        'description': 'Inconsistent completion of diarrhea field'
    },
    'mal_completion_anomaly': {
        'weight': .1,
        'type': 'custom', 
        'function': lambda group: completion_anomaly_score(group, 'diagnosed_with_mal_past_3_months'),
        'description': 'Inconsistent completion of malnutrition diagnosis field'
    },
    
    # New SVA algorithms for remaining yes/no fields
    'muac_colour_sva': {
        'weight': 2.0,
        'type': 'sva',
        'field': 'muac_colour',
        'description': 'MUAC colour response distribution deviation from baseline'
    },
    'recent_va_dose_sva': {
        'weight': 1.0,
        'type': 'sva',
        'field': 'recent_va_dose',
        'description': 'Recent VA dose response distribution deviation from baseline'
    },
    'received_va_dose_before_sva': {
        'weight': 1.0,
        'type': 'sva',
        'field': 'received_va_dose_before',
        'description': 'Received VA dose before response distribution deviation from baseline'
    },
    'received_any_vaccine_sva': {
        'weight': 6.0,
        'type': 'sva',
        'field': 'received_any_vaccine',
        'description': 'Received any vaccine response distribution deviation from baseline'
    },
    'under_treatment_for_mal_sva': {
        'weight': 1.0,
        'type': 'sva',
        'field': 'under_treatment_for_mal',
        'description': 'Under treatment for malnutrition response distribution deviation from baseline'
    },
    
    # New completion anomaly algorithms for remaining yes/no fields
    'muac_colour_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'muac_colour'),
        'description': 'Inconsistent completion of MUAC colour field'
    },
    'recent_va_dose_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'recent_va_dose'),
        'description': 'Inconsistent completion of recent VA dose field'
    },
    'received_va_dose_before_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'received_va_dose_before'),
        'description': 'Inconsistent completion of received VA dose before field'
    },
    'received_any_vaccine_completion_anomaly': {
        'weight': 1.0,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'received_any_vaccine'),
        'description': 'Inconsistent completion of received any vaccine field'
    },
    'diarrhea_sva': {
        'weight': 2.0,
        'type': 'sva', 
        'field': 'diarrhea_last_month',
        'description': 'Diarrhea response distribution deviation from baseline'
    },
    'glasses_sva': {
        'weight': 2.0,
        'type': 'sva',
        'field': 'have_glasses',
        'description': 'Glasses response distribution deviation from baseline'
    },
    'mal_sva': {
        'weight': 2.0,
        'type': 'sva',
        'field': 'diagnosed_with_mal_past_3_months', 
        'description': 'Malnutrition diagnosis distribution deviation from baseline'
    },
    'muac_distribution': {  # NEW: Add the MUAC distribution algorithm
        'weight': 15.0,
        'type': 'custom',
        'function': muac_distribution_fraud_score,
        'description': 'MUAC distribution pattern analysis (bins, slopes, plateaus)'
    },
    'under_treatment_for_mal_completion_anomaly': {
        'weight': .1,
        'type': 'custom',
        'function': lambda group: completion_anomaly_score(group, 'under_treatment_for_mal'),
        'description': 'Inconsistent completion of under treatment for malnutrition field'
    }
}

# Add this to fraud_detection_core.py

# Field Analysis Configuration Registry
# This centralizes all field analysis configuration in one place
FIELD_ANALYSIS_CONFIG = {
    'pct_female': {
        'field_name': 'Gender Distribution',
        'fraud_score': 'fraud_gender_sva',
        'completion_field': 'pct_childs_gender_blank',
        'completion_fraud_score': None,
        'correlation_scores': ['fraud_muac_distribution'],
        'description': 'Percentage of female children per FLW',
        'create_page': True
    },
    'yearly_age_total_deviation': {
        'field_name': 'Age Yearly Deviation',
        'fraud_score': 'fraud_yearly_age_imbalance',
        'completion_field': 'pct_childs_age_in_month_blank',
        'completion_fraud_score': None,
        'correlation_scores': ['fraud_gender_sva', 'fraud_muac_distribution'],
        'description': 'Total deviation from uniform yearly age distribution',
        'create_page': True
    },
    'yearly_age_variance': {
        'field_name': 'Age Yearly Variance',
        'fraud_score': 'fraud_yearly_age_imbalance',
        'completion_field': 'pct_childs_age_in_month_blank',
        'completion_fraud_score': None,
        'correlation_scores': ['fraud_gender_sva', 'fraud_muac_distribution'],
        'description': 'Variance in yearly age group proportions',
        'create_page': True
    },
    'monthly_age_total_deviation': {
        'field_name': 'Age Monthly Deviation',
        'fraud_score': 'fraud_monthly_age_perfection',
        'completion_field': 'pct_childs_age_in_month_blank',
        'completion_fraud_score': None,
        'correlation_scores': ['fraud_gender_sva', 'fraud_muac_distribution'],
        'description': 'Total deviation from perfect monthly age uniformity',
        'create_page': True
    },
    'monthly_age_variance': {
        'field_name': 'Age Monthly Variance',
        'fraud_score': 'fraud_monthly_age_perfection',
        'completion_field': 'pct_childs_age_in_month_blank',
        'completion_fraud_score': None,
        'correlation_scores': ['fraud_gender_sva', 'fraud_muac_distribution'],
        'description': 'Variance in monthly age bin proportions',
        'create_page': True
    },
    'pct_have_glasses_yes': {
        'field_name': 'Have Glasses',
        'fraud_score': 'fraud_glasses_sva',
        'completion_field': 'pct_have_glasses_blank',
        'completion_fraud_score': 'fraud_glasses_completion_anomaly',
        'correlation_scores': ['fraud_gender_sva', 'fraud_muac_distribution'],
        'description': 'Share of clients reported to have glasses',
        'create_page': True
    },
    
    # Fields that get baseline distributions but no analysis pages
    'pct_diagnosed_with_mal_past_3_months_yes': {
        'field_name': 'Malnutrition Diagnosis',
        'fraud_score': 'fraud_mal_sva',
        'completion_field': 'pct_diagnosed_with_mal_past_3_months_blank',
        'completion_fraud_score': 'fraud_mal_completion_anomaly',
        'description': 'Share diagnosed with malnutrition in past 3 months',
        'create_page': False
    },
    'pct_diarrhea_last_month_yes': {
        'field_name': 'Diarrhea Last Month',
        'fraud_score': 'fraud_diarrhea_sva',
        'completion_field': 'pct_diarrhea_last_month_blank',
        'completion_fraud_score': 'fraud_diarrhea_completion_anomaly',
        'description': 'Share with diarrhea in last month',
        'create_page': False
    },
    'pct_va_child_unwell_today_yes': {
        'field_name': 'Child Unwell Today',
        'fraud_score': 'fraud_va_child_unwell_today_sva',
        'completion_field': 'pct_va_child_unwell_today_blank',
        'completion_fraud_score': 'fraud_va_child_unwell_today_completion_anomaly',
        'description': 'Share of children unwell today',
        'create_page': False
    },
    'pct_received_any_vaccine_yes': {
        'field_name': 'Received Any Vaccine',
        'fraud_score': 'fraud_received_any_vaccine_sva',
        'completion_field': 'pct_received_any_vaccine_blank',
        'completion_fraud_score': 'fraud_received_any_vaccine_completion_anomaly',
        'description': 'Share who received any vaccine',
        'create_page': False
    }
}


# Helper function to get fields that should have baseline distributions
def get_baseline_distribution_fields():
    """Return list of fields that should have FLW baseline distributions generated"""
    return list(FIELD_ANALYSIS_CONFIG.keys())


# Helper function to get fields that should have analysis pages
def get_analysis_page_fields():
    """Return list of fields that should have analysis pages created"""
    return [field for field, config in FIELD_ANALYSIS_CONFIG.items() 
            if config.get('create_page', False)]


# Helper function to get fraud score for a value field
def get_fraud_score_for_field(value_field):
    """Return the fraud score column name for a given value field"""
    config = FIELD_ANALYSIS_CONFIG.get(value_field)
    return config['fraud_score'] if config else None


class FraudDetectionCore:
    """Core fraud detection functionality"""
    
    def __init__(self, df, get_parameter_func, log_func):
        self.df = df
        self.get_parameter_value = get_parameter_func
        self.log = log_func
        
        if not FRAUD_SCORING_AVAILABLE:
            raise ImportError("Fraud scoring functions not available. Please ensure aggregation_functions.py contains the required scoring functions.")

    def generate_baseline_data(self):
        """Generate baseline statistical data from current dataset"""
        print("DEBUG: generate_baseline_data() start")
        min_responses = int(self.get_parameter_value('min_responses', 50))
        bin_size = float(self.get_parameter_value('bin_size', 1))
        print(f"DEBUG: min_responses={min_responses}, bin_size={bin_size}")
        
        special_numeric_fields = {"soliciter_muac_cm", "childs_age_in_month"}
        max_categories = 20
        
        fields_with_blank_as_category = {
            "muac_colour",
            "va_child_unwell_today",
        }
        
        custom_binning = {
            "childs_age_in_month": {
                "type": "interval",
                "start": 0,
                "size": 6,
                "description": "6-month age groups"
            },
            "soliciter_muac_cm": {
                "type": "fixed_bins", 
                "start": 8.5,
                "size": 1.0,
                "description": "1cm bins starting from 8.5cm"
            }
        }
        
        # Target fields using new field names only
        target_fields = {
           "opportunity_id",
           "childs_age_in_month",
           "childs_gender", 
           "diagnosed_with_mal_past_3_months",
           "muac_colour",
           "all_six_no",
           "soliciter_muac_cm",
           "under_treatment_for_mal",
           "have_glasses",
           "no_of_children",
           "received_va_dose_before", 
           "recent_va_dose",
           "va_child_unwell_today",
           "diarrhea_last_month",
           "child_name",
           "household_phone",
           "household_name",
           "received_any_vaccine",
       }
        
        df = self.df.copy()
        total_visits = len(df)
        print(f"DEBUG: total_visits={total_visits}; columns={list(df.columns)}")
        print(f"DEBUG: target_fields: {target_fields}")
        
        baseline_data = {
            "metadata": {
                "source_file": None,
                "date_generated": datetime.utcnow().isoformat(),
                "total_visits": int(total_visits),
                "total_flws": df['flw_id'].nunique() if 'flw_id' in df.columns else None,
                "notes": "Unified baseline with fraud detection scores",
                "included_fields": [],
                "method": "Composite fraud scoring (gender SVA + age imbalance + perfection detection)",
                "fields_with_blank_as_category": list(fields_with_blank_as_category),
                "custom_binning": custom_binning,
                "fraud_algorithms": {
                    algo_name: {
                        'weight': config['weight'],
                        'type': config['type'],
                        'description': config['description'],
                        'field': config.get('field', None)  # Only for SVA algorithms
                    }
                    for algo_name, config in FRAUD_SCORING_ALGORITHMS.items()
                }
            },
            "fields": {}
        }
        
        included = 0
        processed = 0
        for col in df.columns:
            if col.lower() in ['flw_id', 'visit_id']:
                continue
            if col not in target_fields:
                continue
            processed += 1
            series = df[col]
            col_lower = col.lower()
            
            if pd.api.types.is_datetime64_any_dtype(series):
                print(f"DEBUG: skip datetime field: {col}")
                continue
            
            # Use field_utils function for normalization
            if FIELD_UTILS_AVAILABLE:
                norm_key = normalize_field_key(col_lower)
            else:
                # Fallback if field_utils not available
                norm_key = col_lower
                if norm_key.startswith('pct_'):
                    norm_key = norm_key[4:]
            
            if norm_key in fields_with_blank_as_category:
                series = series.fillna("__BLANK__")
                print(f"DEBUG: treating blanks as category for '{col}'")
            else:
                nulls = series.isna().sum()
                series = series.dropna()
                if nulls:
                    print(f"DEBUG: dropped {nulls} nulls for '{col}'")
            
            if len(series) < min_responses:
                print(f"DEBUG: field '{col}' skipped (n={len(series)} < {min_responses})")
                continue
                
            field_entry = None
            if col in custom_binning:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric_series) >= min_responses:
                    field_entry = self._process_custom_binning(numeric_series, custom_binning[col])
            
            if field_entry is None:
                if col_lower in special_numeric_fields:
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    if len(numeric_series) < min_responses:
                        print(f"DEBUG: special numeric '{col}' insufficient after coercion")
                        continue
                    field_entry = self._process_numeric(numeric_series, bin_size)
                elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                    if series.nunique() <= max_categories:
                        field_entry = self._process_categorical(series)
                    else:
                        print(f"DEBUG: numeric '{col}' has too many categories ({series.nunique()})")
                        continue
                else:
                    if series.nunique() <= max_categories:
                        field_entry = self._process_categorical(series)
                    else:
                        print(f"DEBUG: categorical '{col}' has too many categories ({series.nunique()})")
                        continue
            
            if field_entry is not None:
                baseline_data["fields"][col] = field_entry
                baseline_data["metadata"]["included_fields"].append(col)
                included += 1
                print(f"DEBUG: included field '{col}' in baseline")
        
        print(f"DEBUG: generate_baseline_data() done; processed={processed}, included={included}")
        return baseline_data

    def load_baseline_data(self, output_dir):
        """Load baseline data from JSON file"""
        baseline_path = os.path.join(output_dir, "baseline_stats.json")
        print(f"DEBUG: load_baseline_data() reading {baseline_path}")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline stats file not found at {baseline_path}")
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("DEBUG: baseline loaded OK")    
        return data

    def save_baseline_csv(self, baseline_data, output_dir):
        """Save flattened baseline data as CSV"""
        print("DEBUG: save_baseline_csv() start")
        flat_rows = []
        for field, info in baseline_data.get("fields", {}).items():
            if info["type"] == "categorical":
                for cat, stats in info["categories"].items():
                    flat_rows.append([field, cat, stats["count"], stats["proportion"]])
            else:
                for bin_label, stats in info["bins"].items():
                    flat_rows.append([field, bin_label, stats["count"], stats["proportion"]])
        csv_df = pd.DataFrame(flat_rows, columns=["field_name", "category_or_bin", "count", "proportion"])
        csv_path = os.path.join(output_dir, "baseline_stats.csv")
        print(f"DEBUG: writing baseline CSV -> {csv_path} rows={len(flat_rows)}")
        csv_df.to_csv(csv_path, index=False)
        return csv_path

    def calculate_fraud_scores(self, baseline_data, is_baseline=False):
        """
        Calculate composite fraud scores for each FLW using multiple algorithms.
        
        Args:
            baseline_data: Baseline statistics for comparison
            is_baseline: Whether this is baseline generation mode
            
        Returns:
            pd.DataFrame: FLW results with individual and composite fraud scores
        """
        print(f"DEBUG: calculate_fraud_scores(is_baseline={is_baseline}) start")
        
        excluded_fields = {
            "opportunity_name", "status", "unit_name", 
            "opportunity_id", "flagged"
        }
        
        fields_with_blank_as_category = set(
            baseline_data.get("metadata", {}).get("fields_with_blank_as_category", [])
        )
        
        df = self.df.copy()
        results = []
        groups = list(df.groupby("flw_id"))
        
        print(f"DEBUG: calculating fraud scores for {len(groups)} FLWs")
        print(f"DEBUG: Available columns: {list(df.columns)}")
        
        # Calculate individual fraud scores for each FLW
        for idx, (flw_id, group) in enumerate(groups, 1):
            if idx % 50 == 0:
                print(f"DEBUG: ... FLW {idx}/{len(groups)} -> {flw_id}")
                
            flw_stats = {"flw_id": flw_id}
            
            # Get basic FLW info
            most_recent = self._get_most_recent_visit_info(group)
            flw_stats.update(most_recent)
            flw_stats["n_visits"] = len(group)
            
            # Calculate individual fraud scores
            individual_scores = {}
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for algo_name, algo_config in FRAUD_SCORING_ALGORITHMS.items():
                score = None
                details = None
                
                if algo_config['type'] == 'sva':
                    # Handle SVA-based scoring
                    field_name = algo_config['field']
                    
                    if field_name in baseline_data.get("fields", {}) and field_name in group.columns:
                        field_info = baseline_data["fields"][field_name]
                        if field_info["type"] == "categorical":
                            # Calculate FLW distribution (ignoring blanks for yes/no fields)
                            values = self._prepare_categorical_values(
                                group[field_name], field_name, fields_with_blank_as_category
                            )
                            if len(values) > 0:
                                baseline_dist = {
                                    cat: info["proportion"] 
                                    for cat, info in field_info["categories"].items()
                                    if str(cat) != '__BLANK__'  # Ignore blanks for SVA calculation
                                }
                                
                                # Renormalize baseline distribution without blanks
                                total_baseline = sum(baseline_dist.values())
                                if total_baseline > 0:
                                    baseline_dist = {cat: prop/total_baseline for cat, prop in baseline_dist.items()}
                                
                                # Calculate FLW distribution without blanks
                                non_blank_values = values[values != '__BLANK__'] if '__BLANK__' in values.values else values
                                if len(non_blank_values) > 0:
                                    flw_dist = non_blank_values.value_counts(normalize=True).to_dict()
                                    
                                    # Use appropriate SVA function
                                    if field_name == 'childs_gender':
                                        score = gender_sva_score(baseline_dist, flw_dist)
                                    else:
                                        # Use generic yes/no SVA for other fields
                                        score = yes_no_sva_score(baseline_dist, flw_dist)
                
                elif algo_config['type'] == 'custom':
                    # Handle custom function-based scoring
                    score_function = algo_config['function']
                    try:
                        result = score_function(group)
                        
                        # NEW: Handle tuple returns (score, details)
                        if isinstance(result, tuple) and len(result) == 2:
                            score, details = result
                        else:
                            score = result
                        
                        # Add age distribution details for yearly age imbalance
                        if algo_name == 'yearly_age_imbalance' and score is not None:
                            age_col = 'childs_age_in_month'
                            
                            if age_col in group.columns:
                                ages = pd.to_numeric(group[age_col], errors='coerce').dropna()
                                if len(ages) > 0:
                                    under_5_ages = ages[ages < 60]
                                    if len(under_5_ages) > 0:
                                        # Calculate percentages for each yearly age group (0-59 months)
                                        age_groups = [
                                            (0, 12, "pct_0_11_month"),
                                            (12, 24, "pct_12_23_month"), 
                                            (24, 36, "pct_24_35_month"),
                                            (36, 48, "pct_36_47_month"),
                                            (48, 60, "pct_48_59_month")
                                        ]
                                        
                                        for start_age, end_age, col_name in age_groups:
                                            age_group = under_5_ages[(under_5_ages >= start_age) & (under_5_ages < end_age)]
                                            pct = len(age_group) / len(under_5_ages)
                                            flw_stats[col_name] = round(pct, 4)
                        
                    except Exception as e:
                        print(f"DEBUG: Error calculating {algo_name} for FLW {flw_id}: {e}")
                        score = None
                        details = f"Error: {str(e)}"
                
                # Store individual score and details
                individual_scores[algo_name] = score
                flw_stats[f"fraud_{algo_name}"] = round(score, 4) if score is not None else None
                
                # NEW: Store details if available
                if details is not None:
                    flw_stats[f"fraud_{algo_name}_details"] = details

                # Extract specific metrics for age algorithms
                if algo_name == 'yearly_age_imbalance' and isinstance(details, dict):
                    flw_stats["yearly_age_total_deviation"] = round(details.get('total_deviation', 0), 4)
                    flw_stats["yearly_age_variance"] = round(details.get('variance', 0), 6)
    
                elif algo_name == 'monthly_age_perfection' and isinstance(details, dict):
                    flw_stats["monthly_age_total_deviation"] = round(details.get('total_deviation', 0), 4)
                    flw_stats["monthly_age_variance"] = round(details.get('variance', 0), 6)

                elif algo_name == 'muac_distribution' and isinstance(details, dict):
                    flw_stats["muac_features_passed"] = details.get('features_passed', 0)
                    flw_stats["muac_bin_counts"] = details.get('bin_counts', '')
                    # Keep the failure reasons in the existing details column
                    flw_stats[f"fraud_{algo_name}_details"] = details.get('failure_reasons', '')

                # Add to weighted total if score is valid
                if score is not None:
                    weight = algo_config['weight']
                    total_weighted_score += score * weight
                    total_weight += weight
            
            # Calculate composite fraud score
            if total_weight > 0:
                composite_score = total_weighted_score / total_weight
                flw_stats["fraud_composite_score"] = round(composite_score, 4)
            else:
                flw_stats["fraud_composite_score"] = None
            
            # Store algorithm weights used (for transparency)
            flw_stats["fraud_total_weight"] = round(total_weight, 1)
            
            # ALWAYS CALCULATE PCT_FEMALE (independent of fraud algorithms)
            if 'childs_gender' in group.columns:
                gender_values = group['childs_gender'].dropna()
                if len(gender_values) > 0:
                    female_count = (gender_values == 'female_child').sum()
                    pct_female = female_count / len(gender_values)
                    flw_stats["pct_female"] = round(pct_female, 4)
                else:
                    flw_stats["pct_female"] = None
            else:
                flw_stats["pct_female"] = None

            # CALCULATE PCT_MUAC_RED (independent of fraud algorithms)
            if 'muac_colour' in group.columns:
                muac_values = group['muac_colour'].dropna()
                muac_values = muac_values[muac_values != '']  # Remove blanks
                if len(muac_values) > 0:
                    red_count = muac_values.astype(str).str.lower().eq('red').sum()
                    pct_red = red_count / len(muac_values)
                    flw_stats["pct_muac_red"] = round(pct_red, 4)
                else:
                    flw_stats["pct_muac_red"] = None
            else:
                flw_stats["pct_muac_red"] = None
        
            
            # CALCULATE AVERAGE NUMBER OF CHILDREN (excluding blanks)
            if 'no_of_children' in group.columns:
                children_values = pd.to_numeric(group['no_of_children'], errors='coerce').dropna()
                if len(children_values) > 0:
                    avg_children = children_values.mean()
                    flw_stats["avg_no_of_children"] = round(avg_children, 2)
                else:
                    flw_stats["avg_no_of_children"] = None
            else:
                flw_stats["avg_no_of_children"] = None
            
            # ADD PCT_BLANK FOR ALL TARGET FIELDS
            all_target_fields = [
                'opportunity_id', 'childs_age_in_month', 'childs_gender', 
                'diagnosed_with_mal_past_3_months', 'muac_colour', 'soliciter_muac_cm',
                'under_treatment_for_mal', 'have_glasses', 'no_of_children',
                'received_va_dose_before', 'recent_va_dose', 'va_child_unwell_today',
                'diarrhea_last_month', 'child_name', 'household_phone',
                'household_name', 'received_any_vaccine', 'all_six_no'  # FIXED: Added all_six_no
            ]
            
            for field in all_target_fields:
                if field in group.columns:
                    total_records = len(group)
                    
                    # Calculate blank percentage (null + empty string) - scale 0-1
                    non_null = group[field].dropna()
                    non_blank = non_null[non_null != '']
                    blank_count = total_records - len(non_blank)
                    pct_blank = blank_count / total_records  # 0-1 scale
                    
                    flw_stats[f"pct_{field}_blank"] = round(pct_blank, 3)

            # ADD PCT_YES FOR SPECIFIC YES/NO FIELDS ONLY (of non-blank records)
            yes_no_fields = [
                'diagnosed_with_mal_past_3_months',
                'diarrhea_last_month', 
                'have_glasses',
                'under_treatment_for_mal',
                'va_child_unwell_today',
                'all_six_no',
                'received_any_vaccine'
            ]
            
            for field in yes_no_fields:
                if field in group.columns:
                    # Get non-blank records
                    non_null = group[field].dropna()
                    non_blank = non_null[non_null != '']
                    
                    if len(non_blank) > 0:
                        # Handle both yes/no and true/false fields - FIXED
                        if field == 'all_six_no':
                            # For true/false field, count "true" responses
                            positive_count = non_blank.astype(str).str.lower().isin(['true', '1', 'yes']).sum()
                        else:
                            # For yes/no fields, count "yes" responses
                            positive_count = (non_blank == 'yes').sum()
                        
                        pct_yes = positive_count / len(non_blank)  # 0-1 scale
                        flw_stats[f"pct_{field}_yes"] = round(pct_yes, 3)
                    else:
                        flw_stats[f"pct_{field}_yes"] = None
            
            results.append(flw_stats)
        
        # Convert to DataFrame and sort by composite fraud score
        results_df = pd.DataFrame(results)
        # Sort with NaNs at the end (compatible with older pandas)
        results_df = results_df.sort_values(by="fraud_composite_score", ascending=False)
        
        print(f"DEBUG: fraud scoring complete, results_df shape={results_df.shape}")
        
        # Add fraud score percentiles if we have valid scores
        valid_scores = results_df["fraud_composite_score"].dropna()
        if len(valid_scores) > 0:
            if is_baseline:
                # Store fraud score percentiles in baseline data for future use
                percentiles = {
                    f"fraud_p{p}": float(np.percentile(valid_scores, p)) 
                    for p in [1,5,10,20,30,40,50,60,70,80,90,95,99]
                }
                baseline_data["fraud_score_percentiles"] = percentiles
                print("DEBUG: stored baseline fraud score percentiles")
                # NEW: Generate FLW distributions for field analysis  

                try:
                    from .baseline_enhancement import generate_flw_distributions
                    baseline_data = generate_flw_distributions(results_df, baseline_data)
                except ImportError:
                    print("DEBUG: baseline_enhancement not available - skipping FLW distributions")
                except Exception as e:
                    print(f"DEBUG: Error generating FLW distributions: {e}")

            # Add percentile columns
            results_df = self._add_fraud_percentile_columns(results_df, valid_scores, is_baseline, baseline_data)
        
        # Add baseline reference row
        baseline_row = self._create_fraud_baseline_row(baseline_data, results_df.columns)
        results_df = pd.concat([baseline_row, results_df], ignore_index=True)
        
        # Reorder columns to put fraud scores first
        results_df = self._reorder_fraud_columns(results_df)
        
        print("DEBUG: calculate_fraud_scores() complete")
        return results_df

    def _add_fraud_percentile_columns(self, df, valid_scores, is_baseline, baseline_data=None):
        """Add fraud score percentile columns to results DataFrame"""
        if len(valid_scores) == 0:
            df["fraud_percentile"] = None
            df["fraud_risk_category"] = None
            return df
        
        if is_baseline:
            # For baseline generation, use current data percentiles
            v = sorted(valid_scores)
            def get_percentile(score):
                if pd.isna(score): 
                    return None
                return round((np.searchsorted(v, score) / len(v)) * 100, 1)
        else:
            # For ranking mode, use stored baseline percentiles
            baseline_percentiles = baseline_data.get("fraud_score_percentiles", {}) if baseline_data else {}
            print(f"DEBUG: Using baseline fraud percentiles: {list(baseline_percentiles.keys())}")
            
            def get_percentile(score):
                if pd.isna(score): 
                    return None
                if not baseline_percentiles:
                    return None
                    
                # Find percentile based on baseline thresholds
                percentile = 0
                for p in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                    threshold = baseline_percentiles.get(f"fraud_p{p}", 0)
                    if score >= threshold:
                        percentile = p
                    else:
                        break
                return percentile
        
        df["fraud_percentile"] = df["fraud_composite_score"].apply(get_percentile)
        df["fraud_risk_category"] = df["fraud_percentile"].apply(self._categorize_fraud_risk)
        
        return df

    def _categorize_fraud_risk(self, percentile):
        """Categorize fraud risk based on percentile"""
        if pd.isna(percentile): 
            return None
        if percentile >= 95: 
            return "Very High Risk (top 5%)"
        elif percentile >= 90: 
            return "High Risk (top 10%)"
        elif percentile >= 75: 
            return "Moderate Risk (top 25%)"
        elif percentile >= 50: 
            return "Above Average Risk"
        elif percentile >= 25: 
            return "Below Average Risk"
        else: 
            return "Low Risk (bottom 25%)"

    def _create_fraud_baseline_row(self, baseline_data, columns):
        """Create baseline reference row for fraud scoring results"""
        baseline_row = pd.DataFrame([{
            'flw_id': 'BASELINE VALUES',
            'name': None,
            'opportunity_name': None,
            'last_visit_date': None,
            'n_visits': None,
            'fraud_composite_score': 0.0000,
            'fraud_percentile': 50.0,
            'fraud_risk_category': '50th percentile (baseline reference)',
            'fraud_total_weight': sum(config['weight'] for config in FRAUD_SCORING_ALGORITHMS.values())
        }])
        
        # Set individual fraud algorithm scores to baseline values
        for algo_name in FRAUD_SCORING_ALGORITHMS.keys():
            baseline_row[f"fraud_{algo_name}"] = 0.0000
        
        # Calculate baseline age distribution from actual baseline data
        age_columns = ["pct_0_11_month", "pct_12_23_month", "pct_24_35_month", "pct_36_47_month", "pct_48_59_month"]
        
        if 'childs_age_in_month' in baseline_data.get('fields', {}):
            age_info = baseline_data['fields']['childs_age_in_month']
            
            # If it's binned data, try to extract age group proportions
            if age_info.get('type') == 'numeric' and 'bins' in age_info:
                bins = age_info['bins']
                
                # Initialize age group counts
                age_group_counts = {
                    "pct_0_11_month": 0,
                    "pct_12_23_month": 0, 
                    "pct_24_35_month": 0,
                    "pct_36_47_month": 0,
                    "pct_48_59_month": 0
                }
                
                total_under_60 = 0
                
                # Map bins to age groups
                for bin_label, bin_info in bins.items():
                    try:
                        # Parse bin range (e.g., "0.0-6.0" or "12-18")
                        if '-' in str(bin_label):
                            start_age = float(str(bin_label).split('-')[0])
                            count = bin_info.get('count', 0)
                            
                            # Only count ages under 60 months
                            if start_age < 60:
                                total_under_60 += count
                                
                                # Assign to appropriate age group
                                if start_age < 12:
                                    age_group_counts["pct_0_11_month"] += count
                                elif start_age < 24:
                                    age_group_counts["pct_12_23_month"] += count
                                elif start_age < 36:
                                    age_group_counts["pct_24_35_month"] += count
                                elif start_age < 48:
                                    age_group_counts["pct_36_47_month"] += count
                                else:  # 48-59
                                    age_group_counts["pct_48_59_month"] += count
                    except (ValueError, IndexError):
                        continue
                
                # Calculate proportions
                if total_under_60 > 0:
                    for col in age_columns:
                        proportion = age_group_counts[col] / total_under_60
                        baseline_row[col] = round(proportion, 4)
                else:
                    # Fallback to uniform if no valid data
                    for col in age_columns:
                        baseline_row[col] = 0.2000
            else:
                # Fallback to uniform if not binned numeric data
                for col in age_columns:
                    baseline_row[col] = 0.2000
        else:
            # Fallback to uniform if no age data in baseline
            for col in age_columns:
                baseline_row[col] = 0.2000
        
        # Calculate pct_female from baseline gender distribution
        if 'childs_gender' in baseline_data.get('fields', {}):
            gender_info = baseline_data['fields']['childs_gender']
            if gender_info.get('type') == 'categorical':
                baseline_pct_female = gender_info['categories'].get('female_child', {}).get('proportion', 0.5)
                baseline_row["pct_female"] = round(baseline_pct_female, 4)
        else:
            baseline_row["pct_female"] = 0.5000  # Default 50/50 if no gender data

        # Calculate avg_no_of_children from baseline data
        if 'no_of_children' in baseline_data.get('fields', {}):
            children_info = baseline_data['fields']['no_of_children']
            if children_info.get('type') == 'categorical':
                # Calculate weighted average from categorical data
                total_weighted = 0
                total_count = 0
                for value_str, value_info in children_info['categories'].items():
                    try:
                        value = float(value_str)
                        count = value_info.get('count', 0)
                        total_weighted += value * count
                        total_count += count
                    except (ValueError, TypeError):
                        continue
                if total_count > 0:
                    avg_children = total_weighted / total_count
                    baseline_row["avg_no_of_children"] = round(avg_children, 2)
                else:
                    baseline_row["avg_no_of_children"] = 3.50  # Default
            elif children_info.get('type') == 'numeric' and 'summary' in children_info:
                # Use mean from numeric summary
                baseline_row["avg_no_of_children"] = round(children_info['summary'].get('mean', 3.50), 2)
            else:
                baseline_row["avg_no_of_children"] = 3.50  # Default
        else:
            baseline_row["avg_no_of_children"] = 3.50  # Default if no data
        
        # Calculate baseline blank percentages from actual baseline data
        total_visits = baseline_data.get('metadata', {}).get('total_visits', 1)
        fields_data = baseline_data.get('fields', {})
        
        all_target_fields = [
            'opportunity_id', 'childs_age_in_month', 'childs_gender', 
            'diagnosed_with_mal_past_3_months', 'muac_colour', 'soliciter_muac_cm',
            'under_treatment_for_mal', 'have_glasses', 'no_of_children',
            'received_va_dose_before', 'recent_va_dose', 'va_child_unwell_today',
            'diarrhea_last_month', 'child_name', 'household_phone',
            'household_name', 'received_any_vaccine', 'all_six_no'  # FIXED: Added all_six_no
        ]
        
        for field in all_target_fields:
            col_name = f"pct_{field}_blank"
            if field in fields_data:
                field_info = fields_data[field]
                total_responses = field_info.get('total_responses', 0)
                # Blank percentage = (total_visits - total_responses) / total_visits
                blank_pct = (total_visits - total_responses) / total_visits if total_visits > 0 else 0.0
                baseline_row[col_name] = round(blank_pct, 3)
            else:
                # Field not in baseline - assume high blank rate
                baseline_row[col_name] = 0.900
        
        # Calculate baseline yes percentages from actual baseline categorical data - FIXED
        yes_no_fields = [
            'diagnosed_with_mal_past_3_months',
            'diarrhea_last_month', 
            'have_glasses',
            'under_treatment_for_mal',
            'va_child_unwell_today',
            'all_six_no',
            'received_any_vaccine'
        ]

        for field in yes_no_fields:
            col_name = f"pct_{field}_yes"
            if field in fields_data:
                field_info = fields_data[field]
                if field_info.get('type') == 'categorical':
                    categories = field_info.get('categories', {})
                    
                    # Calculate total non-blank responses
                    total_non_blank = 0
                    positive_responses = 0
                    
                    for cat, cat_info in categories.items():
                        if str(cat) != '__BLANK__':  # Exclude blank responses
                            total_non_blank += cat_info.get('count', 0)
                            
                            # Handle both yes/no and true/false fields - FIXED
                            cat_lower = str(cat).lower()
                            if field == 'all_six_no':
                                # For true/false field, count "true" responses
                                if cat_lower in ['true', '1', 'yes']:
                                    positive_responses += cat_info.get('count', 0)
                            else:
                                # For yes/no fields, count "yes" responses
                                if cat_lower == 'yes':
                                    positive_responses += cat_info.get('count', 0)
                    
                    # Calculate positive percentage of non-blank responses
                    if total_non_blank > 0:
                        positive_pct = positive_responses / total_non_blank
                        baseline_row[col_name] = round(positive_pct, 3)
                    else:
                        baseline_row[col_name] = 0.000
                else:
                    # Field exists but not categorical - default low rate
                    baseline_row[col_name] = 0.050
            else:
                # Field not in baseline - default low rate
                baseline_row[col_name] = 0.050
        
        # Fill any missing columns with None
        for col in columns:
            if col not in baseline_row.columns:
                baseline_row[col] = None
        
        return baseline_row

    def _reorder_fraud_columns(self, df):
        """Reorder columns to put fraud scores prominently at the front"""
        column_order = [
            "flw_id", "name", "opportunity_name", "last_visit_date",
            "n_visits", "fraud_composite_score", "fraud_percentile", "fraud_risk_category"
        ]
        
        # Add individual fraud algorithm columns
        fraud_algo_cols = [f"fraud_{algo}" for algo in FRAUD_SCORING_ALGORITHMS.keys()]
        column_order.extend(sorted(fraud_algo_cols))
        
        # Add fraud metadata
        column_order.append("fraud_total_weight")
        
        # NEW: Add detail columns right after the main fraud scores
        detail_cols = [c for c in df.columns if c.endswith('_details')]
        column_order.extend(sorted(detail_cols))
        
        # Add remaining columns...
        pct_cols = [c for c in df.columns if c.startswith('pct_')]
        column_order.extend(sorted(pct_cols))
        
        # Add any remaining columns
        remaining_cols = [c for c in df.columns if c not in column_order]
        column_order.extend(sorted(remaining_cols))
        
        # Return only columns that actually exist
        final_columns = [c for c in column_order if c in df.columns]
        return df[final_columns]

    # Helper methods for baseline processing
    def _get_most_recent_visit_info(self, group):
        """Get info from most recent visit for an FLW"""
        group_sorted = group.sort_values(by='visit_date', ascending=False)
        most_recent_row = group_sorted.iloc[0]
        info = {
            'name': most_recent_row.get('flw_name', None),
            'opportunity_id': most_recent_row.get('opportunity_id', None),
            'opportunity_name': most_recent_row.get('opportunity_name', None),
            'last_visit_date': self._format_date(most_recent_row.get('visit_date', None))
        }
        return info

    def _format_date(self, date_value):
        """Format date value consistently"""
        if pd.isna(date_value) or date_value is None:
            return None
        if isinstance(date_value, str):
            try:
                date_value = pd.to_datetime(date_value)
            except:
                return date_value
        if hasattr(date_value, 'date'):
            return date_value.date().strftime('%Y-%m-%d')
        else:
            return str(date_value)

    def _prepare_categorical_values(self, series, field_name, fields_with_blank_as_category):
        """Prepare categorical values for analysis"""
        if FIELD_UTILS_AVAILABLE:
            return prepare_categorical_values(series, field_name, fields_with_blank_as_category)
        else:
            # Fallback if field_utils not available
            if field_name in fields_with_blank_as_category:
                values = series.fillna("__BLANK__").astype(str)
            else:
                values = series.dropna().astype(str)
            return values

    def _process_categorical(self, series):
        """Process categorical field for baseline"""
        counts = series.value_counts(dropna=False)
        total = counts.sum()
        categories = {str(cat): {"count": int(count), "proportion": float(count / total)} for cat, count in counts.items()}
        return {"type": "categorical", "categories": categories, "total_responses": int(total)}

    def _process_numeric(self, series, bin_size):
        """Process numeric field for baseline"""
        total = series.count()
        summary = {"mean": float(series.mean()), "std": float(series.std(ddof=0)), "min": float(series.min()), "max": float(series.max()), "quartiles": [float(q) for q in np.percentile(series, [25, 50, 75])]}
        bin_edges = np.arange(series.min(), series.max() + bin_size, bin_size)
        if len(bin_edges) < 2:
            bin_edges = [series.min(), series.max() + bin_size]
        bin_labels = [f"{round(bin_edges[i],1)}-{round(bin_edges[i+1],1)}" for i in range(len(bin_edges)-1)]
        counts = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True).value_counts().sort_index()
        bins_dict = {label: {"count": int(count), "proportion": float(count / total)} for label, count in counts.items()}
        return {"type": "numeric", "summary": summary, "bins": bins_dict, "total_responses": int(total)}

    def _process_custom_binning(self, series, bin_config):
        """Process field with custom binning for baseline"""
        if FIELD_UTILS_AVAILABLE:
            return process_custom_binning(series, bin_config)
        else:
            # Fallback if field_utils not available
            return self._process_numeric(series, 1.0)
