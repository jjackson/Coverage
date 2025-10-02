import pandas as pd
import numpy as np

def completion_anomaly_score(flw_group, field_name):
    """
    Calculate fraud score based on inconsistent completion of specified field.
    Uses bimodal logic: penalizes medium completion rates (20-80% blank), 
    rewards very high (80-100% blank) or very low (0-20% blank) completion rates.
    
    Args:
        flw_group: DataFrame with FLW's visit data
        field_name: Name of field to analyze
        
    Returns:
        Float: Fraud score (0.0 to 1.0)
    """
    if field_name not in flw_group.columns:
        return None
    
    total_records = len(flw_group)
    if total_records == 0:
        return None
    
    # Calculate completion rate
    non_null = flw_group[field_name].dropna()
    non_blank = non_null[non_null != '']
    completion_rate = len(non_blank) / total_records
    completion_pct = completion_rate * 100
    
    # Bimodal scoring: penalize medium completion rates most heavily
    # All values capped at 1.0 for consistency with other fraud algorithms
    if completion_pct <= 0:
        return 0.0              # No data = no fraud score
    elif completion_pct <= 20:
        return completion_pct * 0.005  # Low completion = low penalty (0.0-0.1)
    elif completion_pct <= 40:
        return 0.1 + (completion_pct - 20) * 0.02  # Medium-low = moderate penalty (0.1-0.5)
    elif completion_pct <= 60:
        return 0.5 + (completion_pct - 40) * 0.025  # Medium = high penalty (0.5-1.0)
    elif completion_pct <= 80:
        return 1.0 - (completion_pct - 60) * 0.025  # Medium-high = decreasing penalty (1.0-0.5)
    else:
        return max(0.0, 0.5 - (completion_pct - 80) * 0.025)  # High completion = low penalty (0.5-0.0)



def yes_no_sva_score(baseline_dist, flw_dist):
    """
    Calculate fraud score based on yes/no distribution deviation from baseline.
    Similar to gender_sva_score but optimized for binary yes/no fields.
    
    Args:
        baseline_dist: Dict with baseline proportions (e.g., {'yes': 0.15, 'no': 0.85})
        flw_dist: Dict with FLW proportions (e.g., {'yes': 0.60, 'no': 0.40})
        
    Returns:
        Float: SVA fraud score (0.0 to 1.0)
    """
    sva_score = 0.0
    
    # Calculate sum of absolute deviations
    for category in baseline_dist:
        baseline_freq = baseline_dist[category]
        flw_freq = flw_dist.get(category, 0)
        deviation = abs(flw_freq - baseline_freq)
        sva_score += deviation
    
    # Normalize score to 0-1 range
    # For binary fields, max possible SVA is 2.0 (if completely opposite)
    normalized_score = min(sva_score / 2.0, 1.0)
    
    return normalized_score

def age_variance_6month_bins(flw_group):
    """
    Calculate variance of age distribution across 6-month bins
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    max_age = ages.max()
    bin_edges = list(range(0, int(max_age) + 7, 6))
    
    if len(bin_edges) < 2:
        bin_edges = [0, 6]
    
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    
    try:
        binned_ages = pd.cut(ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        proportions = bin_counts / len(ages)
        return float(proportions.var())
    except Exception as e:
        print(f"Error in age_variance_6month_bins: {e}")
        return None


def age_variance_yearly_bins(flw_group):
    """
    Calculate variance of age distribution across yearly bins
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    max_age = ages.max()
    bin_edges = list(range(0, int(max_age) + 13, 12))
    
    if len(bin_edges) < 2:
        bin_edges = [0, 12]
    
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    
    try:
        binned_ages = pd.cut(ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        proportions = bin_counts / len(ages)
        return float(proportions.var())
    except Exception as e:
        print(f"Error in age_variance_yearly_bins: {e}")
        return None


def age_variance_monthly_bins(flw_group):
    """
    Calculate variance of age distribution across monthly bins
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    max_age = ages.max()
    bin_edges = list(range(0, int(max_age) + 2, 1))
    
    if len(bin_edges) < 2:
        bin_edges = [0, 1]
    
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    
    try:
        binned_ages = pd.cut(ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        proportions = bin_counts / len(ages)
        return float(proportions.var())
    except Exception as e:
        print(f"Error in age_variance_monthly_bins: {e}")
        return None


def pct_child_unwell_blank(flw_group):
    """
    Calculate percentage of unwell field that are blank
    """
    col_name = 'va_child_unwell_today'
    
    if col_name not in flw_group.columns:
        return None
    
    total_records = len(flw_group)
    if total_records == 0:
        return None
    
    null_count = flw_group[col_name].isna().sum()
    blank_count = (flw_group[col_name] == '').sum()
    
    total_blank = null_count + blank_count
    percentage = (total_blank / total_records) * 100
    
    return float(percentage)


def pct_child_unwell_yes_of_nonblank(flw_group):
    """
    Calculate percentage of non-blank unwell responses that are yes
    """
    col_name = 'va_child_unwell_today'
    
    if col_name not in flw_group.columns:
        return None
    
    non_blank = flw_group[col_name].dropna()
    non_blank = non_blank[non_blank != '']
    
    if len(non_blank) == 0:
        return None
    
    yes_count = (non_blank == 'yes').sum()
    percentage = (yes_count / len(non_blank)) * 100
    
    return float(percentage)


def unwell_completion_anomaly_score(flw_group):
    """
    Calculate fraud score based on inconsistent completion of unwell field
    """
    col_name = 'va_child_unwell_today'
    
    if col_name not in flw_group.columns:
        return None
    
    total_records = len(flw_group)
    if total_records == 0:
        return None
    
    non_blank = flw_group[col_name].dropna()
    non_blank = non_blank[non_blank != '']
    
    completion_rate = len(non_blank) / total_records
    completion_pct = completion_rate * 100
    
    if completion_pct <= 0:
        return 0.0
    elif completion_pct >= 85:
        return 0.0
    elif completion_pct <= 50:
        return 2.0 * (completion_pct / 50)
    else:
        remaining_distance = 85 - completion_pct
        total_distance = 85 - 50
        return remaining_distance / total_distance


def child_unwell_overreporting_score(flw_group):
    """
    Calculate fraud score based on impossible rates of unwell reporting
    """
    col_name = 'va_child_unwell_today'
    
    if col_name not in flw_group.columns:
        return None
    
    non_blank = flw_group[col_name].dropna()
    non_blank = non_blank[non_blank != '']
    
    if len(non_blank) == 0:
        return None
    
    yes_count = (non_blank == 'yes').sum()
    yes_rate = (yes_count / len(non_blank)) * 100
    
    if yes_rate <= 10:
        return yes_rate * 0.01
    elif yes_rate <= 12:
        return 0.1 + (yes_rate - 10) * 0.05
    elif yes_rate <= 15:
        return 0.2 + (yes_rate - 12) * 0.1
    elif yes_rate <= 20:
        return 0.5 + (yes_rate - 15) * 0.06
    else:
        return min(1.0, 0.8 + (yes_rate - 20) * 0.01)


def distance_from_uniform_age_distribution(flw_group, bin_size_months, max_age_months=60):
    """
    Calculate deviation from uniform age distribution
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    target_ages = ages[ages < max_age_months]
    
    if len(target_ages) == 0:
        return None
    
    bin_edges = list(range(0, max_age_months + bin_size_months, bin_size_months))
    
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        bin_labels.append(f"{start}-{end}mo")
    
    num_bins = len(bin_labels)
    ideal_proportion = 1.0 / num_bins
    
    try:
        binned_ages = pd.cut(target_ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        
        actual_proportions = bin_counts / len(target_ages)
        
        for label in bin_labels:
            if label not in actual_proportions.index:
                actual_proportions[label] = 0.0
        
        actual_proportions = actual_proportions.reindex(bin_labels, fill_value=0.0)
        
        deviation_from_uniform = abs(actual_proportions - ideal_proportion).sum()
        
        return float(deviation_from_uniform)
    except Exception as e:
        print(f"Error in distance_from_uniform_age_distribution: {e}")
        return None


def distance_from_uniform_yearly_bins(flw_group):
    """
    Deviation from uniform for yearly bins
    """
    return distance_from_uniform_age_distribution(flw_group, bin_size_months=12, max_age_months=60)


def distance_from_uniform_6month_bins(flw_group):
    """
    Deviation from uniform for 6-month bins
    """
    return distance_from_uniform_age_distribution(flw_group, bin_size_months=6, max_age_months=60)


def distance_from_uniform_3month_bins(flw_group):
    """
    Deviation from uniform for 3-month bins
    """
    return distance_from_uniform_age_distribution(flw_group, bin_size_months=3, max_age_months=60)


def distance_from_uniform_monthly_bins(flw_group):
    """
    Deviation from uniform for monthly bins
    """
    return distance_from_uniform_age_distribution(flw_group, bin_size_months=1, max_age_months=60)


def gender_sva_score(baseline_dist, flw_dist):
    """
    Calculate fraud score based on gender distribution deviation from baseline
    """
    sva_score = 0.0
    
    for category in baseline_dist:
        baseline_freq = baseline_dist[category]
        flw_freq = flw_dist.get(category, 0)
        deviation = abs(flw_freq - baseline_freq)
        sva_score += deviation
    
    normalized_score = min(sva_score / 1.0, 1.0)
    
    return normalized_score


def yearly_age_imbalance_score(flw_group):
    """
    Calculate fraud score based on deviation from uniform yearly age distribution
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    under_5_ages = ages[ages < 60]
    
    if len(under_5_ages) == 0:
        return None
    
    bin_edges = [0, 12, 24, 36, 48, 60]
    bin_labels = ['0-12mo', '12-24mo', '24-36mo', '36-48mo', '48-60mo']
    
    try:
        binned_ages = pd.cut(under_5_ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        
        actual_proportions = bin_counts / len(under_5_ages)
        
        ideal_proportion = 0.2
        
        for label in bin_labels:
            if label not in actual_proportions.index:
                actual_proportions[label] = 0.0
        
        actual_proportions = actual_proportions.reindex(bin_labels, fill_value=0.0)
        
        deviation_from_uniform = abs(actual_proportions - ideal_proportion).sum()
        
        fraud_score = min(deviation_from_uniform / 1.6, 1.0)
        variance_val = float(actual_proportions.var())
        details = {
            'total_deviation': float(deviation_from_uniform),
            'variance': variance_val
        }
        return (fraud_score, details)
    except Exception as e:
        print(f"Error in yearly_age_imbalance_score: {e}")
        return None


def monthly_age_perfection_score(flw_group):
    """
    Calculate fraud score for suspiciously perfect monthly age distribution
    """
    age_col = 'childs_age_in_month'
    
    if age_col not in flw_group.columns:
        return None
    
    ages = pd.to_numeric(flw_group[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return None
    
    under_5_ages = ages[ages < 60]
    
    if len(under_5_ages) < 20:
        return None
    
    bin_edges = list(range(0, 61, 1))
    bin_labels = [f"{i}-{i+1}mo" for i in range(60)]
    
    try:
        binned_ages = pd.cut(under_5_ages, bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_counts = binned_ages.value_counts()
        
        actual_proportions = bin_counts / len(under_5_ages)
        
        ideal_proportion = 1.0 / 60
        
        for label in bin_labels:
            if label not in actual_proportions.index:
                actual_proportions[label] = 0.0
        
        actual_proportions = actual_proportions.reindex(bin_labels, fill_value=0.0)
        
        deviation_from_uniform = abs(actual_proportions - ideal_proportion).sum()
        
        theoretical_max = 2.0 - (2.0 / 60)
        normalized_deviation = deviation_from_uniform / theoretical_max
        
        perfection_threshold = 0.15
        
        if normalized_deviation < perfection_threshold:
            penalty = (perfection_threshold - normalized_deviation) / perfection_threshold
            fraud_score = min(penalty, 1.0)
        else:
            fraud_score = 0.0
        
        variance_val = float(actual_proportions.var())
        details = {
            'total_deviation': float(deviation_from_uniform),
            'variance': variance_val
        }
        return (fraud_score, details)
    except Exception as e:
        print(f"Error in monthly_age_perfection_score: {e}")
        return None


def calculate_categorical_sva(baseline_dist, flw_dist):
    """
    Calculate Sum of absolute differences between two categorical distributions
    """
    sva_score = 0.0
    for category in baseline_dist:
        baseline_freq = baseline_dist[category]
        flw_freq = flw_dist.get(category, 0)
        sva_score += abs(flw_freq - baseline_freq)
    return sva_score


def test_scoring_functions():
    """
    Test function to validate the scoring functions work correctly
    """
    print("Testing fraud detection scoring functions...")
    
    test_data = pd.DataFrame({
        'flw_id': ['TEST_001'] * 100,
        'childs_age_in_month': [6, 18, 30, 42, 54] * 20,
        'childs_gender': ['male_child', 'female_child'] * 50
    })
    
    imbalance_score = yearly_age_imbalance_score(test_data)
    print(f"Yearly imbalance score: {imbalance_score:.4f}")
    
    perfection_score = monthly_age_perfection_score(test_data)
    print(f"Monthly perfection score: {perfection_score:.4f}")
    
    baseline_gender = {'male_child': 0.5, 'female_child': 0.5}
    flw_gender = {'male_child': 0.5, 'female_child': 0.5}
    gender_score = gender_sva_score(baseline_gender, flw_gender)
    print(f"Gender SVA score: {gender_score:.4f}")
    
    flw_gender_skewed = {'male_child': 0.8, 'female_child': 0.2}
    gender_score_skewed = gender_sva_score(baseline_gender, flw_gender_skewed)
    print(f"Gender SVA score (skewed): {gender_score_skewed:.4f}")
    
    print("Testing complete!")



def muac_distribution_fraud_score(flw_group):
    """
    Calculate MUAC distribution fraud score based on 6 key properties.
    
    Scoring system:
    - 6/6 features pass: 0.0 fraud score (authentic)
    - 5/6 features pass: 0.5 fraud score (minor issue)
    - 4/6 features pass: 0.67 fraud score (moderate issue)
    - 3/6 features pass: 0.75 fraud score (major issue)
    - 2/6 features pass: 0.8 fraud score (severe issue)
    - 1/6 features pass: 0.9 fraud score (very severe issue)
    - 0/6 features pass: 1.0 fraud score (fabricated)
    
    Args:
        flw_group: DataFrame with FLW's visit data
        
    Returns:
        Tuple: (fraud_score, details_dict)
            fraud_score: Float 0.0-1.0 (0.0=authentic, 1.0=fabricated)
            details_dict: Dict with bin counts and failure reasons
    """
    # Define MUAC column - try multiple possible names
    muac_col = None
    possible_muac_cols = ['soliciter_muac_cm', 'muac_measurement_cm', 'muac_cm', 'muac']
    
    for col_name in possible_muac_cols:
        if col_name in flw_group.columns:
            muac_col = col_name
            break
    
    if muac_col is None:
        return None, "no MUAC data"
    
    # Get valid MUAC measurements (9.5-21.5 cm range)
    muac_data = flw_group[muac_col].dropna()
    valid_muac = muac_data[(muac_data >= 9.5) & (muac_data <= 21.5)]
    
    if len(valid_muac) < 20:  # Need minimum data for analysis
        return None, f"insufficient data ({len(valid_muac)} measurements)"
    
    # Define MUAC bins (1cm each, on half-centimeters)
    bin_edges = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5]
    
    # Calculate bin counts
    bin_counts = []
    for i in range(len(bin_edges) - 1):
        min_val, max_val = bin_edges[i], bin_edges[i + 1]
        count = ((valid_muac >= min_val) & (valid_muac < max_val)).sum()
        bin_counts.append(count)
    
    total_count = sum(bin_counts)
    if total_count == 0:
        return None, "no valid measurements"
    
    # Find non-zero bins and peak
    non_zero_indices = [i for i, count in enumerate(bin_counts) if count > 0]
    if not non_zero_indices:
        return None, "no data in bins"
    
    max_count = max(bin_counts)
    peak_indices = [i for i, count in enumerate(bin_counts) if count == max_count]
    peak_index = peak_indices[0]  # Take first peak if multiple
    
    # Calculate thresholds
    wiggle_threshold = total_count * 0.02  # 2% for slope violations
    plateau_threshold = max_count * 0.04   # 4% of peak for plateau detection
    
    # Create bin counts string for details
    bin_counts_str = ",".join(map(str, bin_counts))
    
    # Test 1: Has 5+ bins with data
    has_5plus_bins = len(non_zero_indices) >= 5
    
    # Test 2: Increasing to peak (UPDATED - requires at least one value before peak)
    increasing_to_peak = test_increasing_to_peak(bin_counts, non_zero_indices, peak_index, wiggle_threshold)
    
    # Test 3: Decreasing from peak
    decreasing_from_peak = test_decreasing_from_peak(bin_counts, non_zero_indices, peak_index, wiggle_threshold)
    
    # Test 4: No skipped bins
    no_skipped_bins = test_no_skipped_bins(bin_counts, total_count)
    
    # Test 5: No plateau (UPDATED - 3% threshold, 2% total + 20% peak criteria)
    no_plateau = test_no_plateau(bin_counts, max_count, plateau_threshold)
    
    # Test 6: Peak not over-concentrated (NEW - =50% of total)
    peak_not_overconcentrated = test_peak_not_overconcentrated(bin_counts)
    
    # Calculate points and fraud score
    tests = [has_5plus_bins, increasing_to_peak, decreasing_from_peak, no_skipped_bins, no_plateau, peak_not_overconcentrated]
    points_earned = sum(tests)
    
    # Scoring system for 6 tests: missing even 1 feature = 0.5 fraud score
    if points_earned == 6:
        fraud_score = 0.0      # Perfect - authentic
    elif points_earned == 5:
        fraud_score = 0.5      # Minor issue
    elif points_earned == 4:
        fraud_score = 0.67     # Moderate issue
    elif points_earned == 3:
        fraud_score = 0.75     # Major issue
    elif points_earned == 2:
        fraud_score = 0.8      # Severe issue
    elif points_earned == 1:
        fraud_score = 0.9      # Very severe issue
    else:  # points_earned == 0
        fraud_score = 1.0      # Fabricated
    
    # Generate details string with bin counts and failures
    failed_tests = []
    if not has_5plus_bins:
        failed_tests.append(f"only_{len(non_zero_indices)}_bins")
    if not increasing_to_peak:
        failed_tests.append("increasing")
    if not decreasing_from_peak:
        failed_tests.append("decreasing")
    if not no_skipped_bins:
        failed_tests.append("skipped_bins")
    if not no_plateau:
        failed_tests.append("plateau")
    if not peak_not_overconcentrated:
        failed_tests.append("peak_overconcentrated")
    
    if points_earned == 6:
        details = f"score_6/6: bins={bin_counts_str}, all_tests_passed"
    else:
        details = f"score_{points_earned}/6: bins={bin_counts_str}, failed_{','.join(failed_tests)}"
    
    details_dict = {
        'features_passed': points_earned,
        'bin_counts': bin_counts_str,
        'failure_reasons': ','.join(failed_tests) if failed_tests else 'all_tests_passed'
    }

    return fraud_score, details_dict


def test_increasing_to_peak(bin_counts, non_zero_indices, peak_index, wiggle_threshold):
    """
    Test for proper increasing pattern to peak.
    Requirements:
    - Must have at least one non-zero bin BEFORE the peak
    - Must have at least 1 actual steps UP (>0 change)
    - Can't have any steps DOWN more than 2% threshold
    - Must have at least one bin before peak with =25% of peak value (adequate buildup)
    """
    # NEW: Ensure peak is not the first bin overall
    if peak_index == 0:
        return False  # Peak cannot be in first bin - no room for increasing pattern
    
    # NEW: Ensure there's at least one non-zero bin before the peak
    bins_before_peak = [i for i in non_zero_indices if i < peak_index]
    if len(bins_before_peak) == 0:
        return False  # No data before peak - cannot have increasing pattern
    
    if peak_index not in non_zero_indices:
        return False  # Peak index should be in non-zero indices
    
    peak_position_in_nonzero = non_zero_indices.index(peak_index)
    
    if peak_position_in_nonzero == 0:
        return False  # Peak is first non-zero bin - no increasing pattern possible
    
    steps_to_peak = peak_position_in_nonzero
    if steps_to_peak < 1:
        return False  # Need at least 1 step to have an increasing pattern
    
    increasing_steps = 0
    big_decreases = 0
    
    for i in range(peak_position_in_nonzero):
        current_idx = non_zero_indices[i]
        next_idx = non_zero_indices[i + 1]
        step_change = bin_counts[next_idx] - bin_counts[current_idx]
        
        # Count actual increasing steps (>0)
        if step_change > 0:
            increasing_steps += 1
        
        # Count big decreases (more than wiggle threshold)
        if step_change < -wiggle_threshold:
            big_decreases += 1
    
    # Requirements:
    # 1. At least 1 actual increasing step
    # 2. No big decreases (>2% threshold)
    has_enough_increases = increasing_steps >= 1
    no_big_decreases = big_decreases == 0
    
    # NEW: Additional constraint - need at least one bin before peak with =25% of peak value
    peak_value = bin_counts[peak_index]
    adequate_threshold = peak_value * 0.25
    has_adequate_buildup = False
    
    for i in range(peak_index):
        if bin_counts[i] >= adequate_threshold:
            has_adequate_buildup = True
            break
    
    return has_enough_increases and no_big_decreases and has_adequate_buildup

def test_decreasing_from_peak(bin_counts, non_zero_indices, peak_index, wiggle_threshold):
    """
    Test for proper decreasing pattern from peak.
    Requirements:
    - Must have at least 2 actual steps DOWN (<0 change)
    - Can't have any steps UP more than 2% threshold
    """
    if peak_index == len(bin_counts) - 1 or peak_index not in non_zero_indices:
        return True  # No decreasing needed if peak is last bin
    
    peak_position_in_nonzero = non_zero_indices.index(peak_index)
    
    if peak_position_in_nonzero == len(non_zero_indices) - 1:
        return True  # Peak is last non-zero bin
    
    steps_from_peak = len(non_zero_indices) - 1 - peak_position_in_nonzero
    if steps_from_peak < 2:
        return True  # Need at least 2 steps to evaluate
    
    decreasing_steps = 0
    big_increases = 0
    
    for i in range(peak_position_in_nonzero, len(non_zero_indices) - 1):
        current_idx = non_zero_indices[i]
        next_idx = non_zero_indices[i + 1]
        step_change = bin_counts[next_idx] - bin_counts[current_idx]
        
        # Count actual decreasing steps (<0)
        if step_change < 0:
            decreasing_steps += 1
        
        # Count big increases (more than wiggle threshold)
        if step_change > wiggle_threshold:
            big_increases += 1
    
    # Requirements:
    # 1. At least 2 actual decreasing steps
    # 2. No big increases (>2% threshold)
    has_enough_decreases = decreasing_steps >= 2
    no_big_increases = big_increases == 0
    
    return has_enough_decreases and no_big_increases


def test_no_skipped_bins(bin_counts, total_count):
    """
    Test for no skipped bins using interior window logic.
    - Find first and last significant bins (=2% of total)
    - Check interior window for any zeros (complete gaps)
    - Sub-2% but non-zero values are allowed
    """
    if total_count <= 0:
        return True
    
    significant_bin_threshold = total_count * 0.02  # 2% threshold
    sig_indices = [i for i, count in enumerate(bin_counts) if count >= significant_bin_threshold]
    
    if len(sig_indices) <= 1:
        return True  # No interior to check
    
    # Define interior window from first to last significant bin
    start_idx = sig_indices[0]
    end_idx = sig_indices[-1]
    interior = bin_counts[start_idx:end_idx + 1]
    
    # Fail if ANY zero is found inside the interior window
    return all(count != 0 for count in interior)

def test_no_plateau(bin_counts, max_count, plateau_threshold):
    """
    Test for suspicious plateaus using new criteria:
    - 3% of peak change threshold for any 3 consecutive bins
    - Bins must be =2% of total count AND =20% of peak count
    
    Args:
        bin_counts: List of counts per bin
        max_count: Maximum count (peak value)
        plateau_threshold: Not used in new version (kept for compatibility)
        
    Returns:
        bool: True if no suspicious plateau found, False if plateau detected
    """
    total_count = sum(bin_counts)
    if total_count == 0 or max_count == 0:
        return True
    
    # New thresholds
    change_threshold = max_count * 0.03  # 3% of peak for similarity
    min_count_threshold = total_count * 0.02  # 2% of total
    min_peak_ratio_threshold = max_count * 0.20  # 20% of peak
    
    # Find eligible bins (must meet both count thresholds)
    eligible_bins = []
    for i, count in enumerate(bin_counts):
        if count >= min_count_threshold and count >= min_peak_ratio_threshold:
            eligible_bins.append((i, count))
    
    if len(eligible_bins) < 3:
        return True  # Need at least 3 bins to form a plateau
    
    # Check for plateaus: 3 consecutive eligible bins with similar values
    for i in range(len(eligible_bins) - 2):
        # Get three consecutive eligible bins
        bin1_idx, bin1_count = eligible_bins[i]
        bin2_idx, bin2_count = eligible_bins[i + 1]
        bin3_idx, bin3_count = eligible_bins[i + 2]
        
        # Check if they are actually consecutive in the original bin_counts
        if bin2_idx == bin1_idx + 1 and bin3_idx == bin2_idx + 1:
            # Check if all three values are within 3% of peak of each other
            counts = [bin1_count, bin2_count, bin3_count]
            max_in_group = max(counts)
            min_in_group = min(counts)
            
            if max_in_group - min_in_group <= change_threshold:
                # Found a plateau: 3 consecutive bins with similar values
                return False
    
    return True  # No plateau found

def test_peak_not_overconcentrated(bin_counts):
    """
    Test 6: Peak concentration check.
    Ensures peak bin doesn't account for more than 55% of total measurements.
    
    Args:
        bin_counts: List of counts per bin
        
    Returns:
        bool: True if peak is reasonably distributed, False if over-concentrated
    """
    total_count = sum(bin_counts)
    if total_count == 0:
        return True
    
    max_count = max(bin_counts)
    peak_concentration = max_count / total_count
    
    # Fail if peak accounts for more than 55% of all measurements
    return peak_concentration <= 0.60



# Add these functions to aggregation_functions.py

def all_six_no_overreporting_score(flw_group):
    """
    Calculate fraud score based on unusually high rates of all_six_no = true reporting.
    Strong penalty for rates above 40%.
    
    Args:
        flw_group: DataFrame with FLW's visit data
        
    Returns:
        Float: Fraud score (0.0 to 1.0)
    """
    col_name = 'all_six_no'
    
    if col_name not in flw_group.columns:
        return None
    
    # Get non-blank records
    non_blank = flw_group[col_name].dropna()
    non_blank = non_blank[non_blank != '']
    
    if len(non_blank) == 0:
        return None
    
    # Count "true" responses (could be True, "true", "True", etc.)
    true_responses = non_blank.astype(str).str.lower().isin(['true', '1', 'yes'])
    true_count = true_responses.sum()
    true_rate = (true_count / len(non_blank)) * 100
    
    # Scoring system with strong penalty above 40%
    if true_rate <= 10:
        return true_rate * 0.01  # Very low penalty (0.0-0.1)
    elif true_rate <= 20:
        return 0.1 + (true_rate - 10) * 0.02  # Low penalty (0.1-0.3)
    elif true_rate <= 30:
        return 0.3 + (true_rate - 20) * 0.03  # Moderate penalty (0.3-0.6)
    elif true_rate <= 40:
        return 0.6 + (true_rate - 30) * 0.02  # High penalty (0.6-0.8)
    else:
        # Strong penalty above 40%
        return min(1.0, 0.8 + (true_rate - 40) * 0.02)


def pct_all_six_no_true_of_nonblank(flw_group):
    """
    Calculate percentage of non-blank all_six_no responses that are true.
    Helper function for calculating the actual percentage.
    
    Args:
        flw_group: DataFrame with FLW's visit data
        
    Returns:
        Float: Percentage (0-100) or None if no data
    """
    col_name = 'all_six_no'
    
    if col_name not in flw_group.columns:
        return None
    
    # Get non-blank records
    non_blank = flw_group[col_name].dropna()
    non_blank = non_blank[non_blank != '']
    
    if len(non_blank) == 0:
        return None
    
    # Count "true" responses
    true_responses = non_blank.astype(str).str.lower().isin(['true', '1', 'yes'])
    true_count = true_responses.sum()
    percentage = (true_count / len(non_blank)) * 100
    
    return float(percentage)


def pct_all_six_no_blank(flw_group):
    """
    Calculate percentage of all_six_no field that are blank.
    
    Args:
        flw_group: DataFrame with FLW's visit data
        
    Returns:
        Float: Percentage (0-100) or None if no data
    """
    col_name = 'all_six_no'
    
    if col_name not in flw_group.columns:
        return None
    
    total_records = len(flw_group)
    if total_records == 0:
        return None
    
    null_count = flw_group[col_name].isna().sum()
    blank_count = (flw_group[col_name] == '').sum()
    
    total_blank = null_count + blank_count
    percentage = (total_blank / total_records) * 100
    
    return float(percentage)