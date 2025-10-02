
"""
baseline_enhancement.py (cleaned)
---------------------------------
Generates FLW-level distributions (50-bin histograms) for numeric fields and
augments each bin with median fraud scores for overlay lines in the UI.
"""

from __future__ import annotations

import math
from typing import Dict, List, Iterable
import numpy as np
import pandas as pd


# -----------------------------
# Public API
# -----------------------------

# Replace the generate_flw_distributions function in baseline_enhancement.py

def generate_flw_distributions(flw_data: pd.DataFrame, baseline_data: dict) -> dict:
    """
    Build baseline FLW distributions for configured fields using centralized config.

    Args:
        flw_data: DataFrame at the FLW (one row per FLW) level containing numeric columns
        baseline_data: dict-like baseline structure to enhance

    Returns:
        baseline_data with key 'flw_distributions' -> {field_name: distribution_dict}
    """
    if baseline_data is None:
        baseline_data = {}

    if flw_data is None or not isinstance(flw_data, pd.DataFrame) or flw_data.empty:
        print("DEBUG: No FLW data provided or empty DataFrame; skipping distributions")
        return baseline_data

    # Ensure container exists
    if 'flw_distributions' not in baseline_data:
        baseline_data['flw_distributions'] = {}

    try:
        from .fraud_detection_core import FIELD_ANALYSIS_CONFIG
        
        # Generate distributions for all configured fields
        for field_name, field_config in FIELD_ANALYSIS_CONFIG.items():
            if field_name not in flw_data.columns:
                continue
                
            values = flw_data[field_name].dropna()
            if len(values) < 5:
                # not enough data to build a sensible distribution
                continue

            # Use only rows with a non-null value for this field to align fraud stats
            valid_idx = values.index
            flw_subset = flw_data.loc[valid_idx]
            
            # Pass only the specific fraud score this field needs
            fraud_columns = [field_config['fraud_score']]

            dist = create_field_distribution_with_fraud_scores(
                values, flw_subset, field_name, fraud_columns
            )
            baseline_data['flw_distributions'][field_name] = dist
            
            print(f"DEBUG: Generated distribution for {field_name} with fraud score {field_config['fraud_score']}")

        return baseline_data

    except Exception as e:
        print(f"DEBUG: Error generating FLW distributions: {e}")
        raise


# Also replace get_available_fraud_columns since it's no longer needed
def get_available_fraud_columns(flw_data: pd.DataFrame) -> List[str]:
    """
    This function is deprecated - fraud columns are now specified per field in FIELD_ANALYSIS_CONFIG.
    Keeping for backward compatibility but it should not be used for new distributions.
    """
    if flw_data is None or not isinstance(flw_data, pd.DataFrame):
        return []

    # Common fraud score columns; include those present
    preferred = [
        'fraud_composite_score',
        'fraud_gender_sva',
        'fraud_yearly_age_imbalance',
        'fraud_monthly_age_perfection',
        'fraud_muac_distribution',
    ]
    return [c for c in preferred if c in flw_data.columns and pd.api.types.is_numeric_dtype(flw_data[c])]


def add_muac_features_distribution(flw_data: pd.DataFrame, baseline_data: dict) -> dict:
    """
    Optional: add distribution for 'muac_features_passed' when present.
    """
    if baseline_data is None:
        baseline_data = {}

    if 'muac_features_passed' not in flw_data.columns:
        print("DEBUG: muac_features_passed column not found - skipping MUAC distribution")
        return baseline_data

    values = flw_data['muac_features_passed'].dropna()
    if len(values) == 0:
        return baseline_data

    fraud_columns = get_available_fraud_columns(flw_data)
    dist = create_field_distribution_with_fraud_scores(values, flw_data.loc[values.index], 'muac_features_passed', fraud_columns)
    baseline_data.setdefault('flw_distributions', {})['muac_features_passed'] = dist
    print(f"DEBUG: Added MUAC features distribution: {len(values)} FLWs")
    return baseline_data


def validate_flw_distributions(baseline_data: dict) -> List[str]:
    """
    Validate the generated FLW distributions for consistency.
    Returns a list of issues (empty if all good).
    """
    issues: List[str] = []

    if not isinstance(baseline_data, dict):
        return ["baseline_data should be a dict"]

    if 'flw_distributions' not in baseline_data:
        issues.append("Missing flw_distributions section")
        return issues

    flw_distributions = baseline_data['flw_distributions']
    for field_name, distribution in flw_distributions.items():
        # Required top-level keys
        required_keys = ['min_value', 'max_value', 'bin_size', 'bins', 'statistics']
        for key in required_keys:
            if key not in distribution:
                issues.append(f"{field_name}: Missing {key}")

        # Bins structure
        bins = distribution.get('bins', [])
        if len(bins) != 50:
            issues.append(f"{field_name}: Expected 50 bins, got {len(bins)}")

        total_pct = 0.0
        for i, bin_data in enumerate(bins):
            bin_required_keys = ['bin_start', 'bin_end', 'count', 'percentage', 'median_fraud_scores']
            for key in bin_required_keys:
                if key not in bin_data:
                    issues.append(f"{field_name} bin {i}: Missing {key}")

            if 'percentage' in bin_data:
                total_pct += float(bin_data['percentage'])

            if 'median_fraud_scores' in bin_data and not isinstance(bin_data['median_fraud_scores'], dict):
                issues.append(f"{field_name} bin {i}: median_fraud_scores should be dict")

        # Percentages sum check (~100% within 1% tolerance)
        if abs(total_pct - 100.0) > 1.0:
            issues.append(f"{field_name}: Bin percentages sum to {total_pct:.1f}%, expected ~100%")

        # Bin size check
        if all(k in distribution for k in ['min_value', 'max_value', 'bin_size']):
            expected = (float(distribution['max_value']) - float(distribution['min_value'])) / 50.0
            actual = float(distribution['bin_size'])
            if abs(expected - actual) > 1e-4:
                issues.append(f"{field_name}: bin_size mismatch - expected {expected:.6f}, got {actual:.6f}")

    return issues


# -----------------------------
# Helpers
# -----------------------------

def get_distribution_field_list(flw_data: pd.DataFrame) -> List[str]:
    """
    Heuristically determine which columns are numeric enough to warrant a distribution.
    Rule of thumb: >=80% of non-null values parse as numeric (and not on the skip list).
    """
    if flw_data is None or not isinstance(flw_data, pd.DataFrame):
        return []

    skip_fields: set = {
        'flw_id', 'opportunity_id', 'opportunity_name', 'name', 'username',
        'household_name', 'household_phone', 'child_name'
    }

    candidate_fields: List[str] = []
    for col in flw_data.columns:
        if col in skip_fields:
            continue

        series = flw_data[col].dropna()
        if len(series) < 5:
            continue

        # Attempt to coerce to numeric
        numeric = pd.to_numeric(series, errors='coerce')
        valid_ratio = float(numeric.notna().mean()) if len(series) > 0 else 0.0
        if valid_ratio >= 0.8:
            candidate_fields.append(col)

    return candidate_fields


def _compute_buffered_min_max(values: pd.Series) -> (float, float):
    """
    Expand min/max slightly to avoid edge effects; ensures non-zero bin_size.
    """
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        # Expand around a single value
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        return vmin - eps, vmax + eps

    # Add a 2% buffer on either side
    span = vmax - vmin
    return vmin - 0.02 * span, vmax + 0.02 * span


def create_field_distribution_with_fraud_scores(
    field_values: pd.Series,
    flw_subset: pd.DataFrame,
    field_name: str,
    fraud_columns: Iterable[str]
) -> Dict:
    """
    Create a 50-bin histogram for `field_values` and compute median fraud
    scores per bin using aligned rows in `flw_subset`.
    """
    # Sanitize
    values = pd.to_numeric(field_values, errors='coerce').dropna()
    total = len(values)
    if total == 0:
        # Return an empty shell (should be filtered upstream)
        return {
            'min_value': 0.0, 'max_value': 0.0, 'bin_size': 0.0,
            'bins': [], 'bin_percentages': [], 'bin_labels': [],
            'statistics': {'mean': 0.0, 'std': 0.0, 'count': 0, 'original_min': 0.0, 'original_max': 0.0}
        }

    min_val, max_val = _compute_buffered_min_max(values)
    bin_edges = np.linspace(min_val, max_val, 51)  # 51 edges => 50 bins
    bin_indices = pd.cut(values, bins=bin_edges, include_lowest=True, right=True, labels=False)

    # Precompute stats
    mean_val = float(values.mean())
    std_val = float(values.std(ddof=0)) if total > 1 else 0.0
    bins: List[Dict] = []

    for i in range(50):
        start = float(bin_edges[i])
        end = float(bin_edges[i + 1])
        member_idx = values.index[bin_indices == i]
        count = int(len(member_idx))
        pct = (count / total) * 100.0 if total else 0.0

        bin_data = {
            'bin_start': start,
            'bin_end': end,
            'count': count,
            'percentage': pct,
            'median_fraud_scores': {}
        }

        # Compute medians for each fraud column, aligned to members in this bin
        if count > 0 and fraud_columns:
            sub = flw_subset.loc[member_idx]
            for col in fraud_columns:
                if col in sub.columns:
                    col_vals = pd.to_numeric(sub[col], errors='coerce').dropna()
                    if len(col_vals) > 0:
                        bin_data['median_fraud_scores'][col] = float(col_vals.median())
                    else:
                        bin_data['median_fraud_scores'][col] = None
                else:
                    bin_data['median_fraud_scores'][col] = None

        bins.append(bin_data)

    # Percentages and labels for backward compatibility / debugging
    bin_percentages = [b['percentage'] for b in bins]
    bin_labels = [f"{b['bin_start']:.4f}-{b['bin_end']:.4f}" for b in bins]

    # Compose distribution object
    distribution = {
        'min_value': float(min_val),
        'max_value': float(max_val),
        'bin_size': float((max_val - min_val) / 50.0),
        'bins': bins,
        'bin_percentages': bin_percentages,
        'bin_labels': bin_labels,
        'statistics': {
            'mean': mean_val,
            'std': std_val,
            'count': total,
            'original_min': float(values.min()),
            'original_max': float(values.max())
        }
    }
    return distribution
