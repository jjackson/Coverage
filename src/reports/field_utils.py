import re
import pandas as pd
import numpy as np

def normalize_field_key(field_name):
    """
    Normalize field name for categorization logic.
    This function contains the problematic regex that keeps getting corrupted.
    """
    norm_key = field_name.lower()
    if norm_key.startswith('pct_'):
        norm_key = norm_key[4:]
    norm_key = re.sub(r'_(yes|no|green|red)$', '', norm_key)
    return norm_key

def prepare_categorical_values(series, field_name, fields_with_blank_as_category):
    """
    Prepare categorical values for analysis.
    Moved here to avoid regex corruption issues.
    """
    norm_key = normalize_field_key(field_name)
    
    if norm_key in fields_with_blank_as_category:
        values = series.fillna("__BLANK__").astype(str)
    else:
        values = series.dropna().astype(str)
    return values

def process_custom_binning(series, bin_config):
    """
    Process field with custom binning for baseline.
    Moved here to avoid corruption issues.
    """
    total = series.count()
    summary = {
        "mean": float(series.mean()), 
        "std": float(series.std(ddof=0)), 
        "min": float(series.min()), 
        "max": float(series.max()), 
        "quartiles": [float(q) for q in np.percentile(series, [25, 50, 75])]
    }
    
    if bin_config["type"] == "interval":
        start = bin_config["start"]
        size = bin_config["size"]
        max_val = series.max()
        bin_edges = []
        current = start
        while current < max_val + size:
            bin_edges.append(current)
            current += size
        if len(bin_edges) < 2:
            bin_edges = [start, start + size]
        bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    elif bin_config["type"] == "fixed_bins":
        start = bin_config["start"]
        size = bin_config["size"]
        max_val = series.max()
        bin_edges = []
        current = start
        while current < max_val + size:
            bin_edges.append(current)
            current += size
        if len(bin_edges) < 2:
            bin_edges = [start, start + size]
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    else:
        # Fallback - use simple numeric processing
        bin_edges = np.arange(series.min(), series.max() + 1.0, 1.0)
        if len(bin_edges) < 2:
            bin_edges = [series.min(), series.max() + 1.0]
        bin_labels = [f"{round(bin_edges[i],1)}-{round(bin_edges[i+1],1)}" for i in range(len(bin_edges)-1)]
    
    counts = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True).value_counts().sort_index()
    bins_dict = {label: {"count": int(count), "proportion": float(count / total)} for label, count in counts.items()}
    
    return {"type": "numeric", "summary": summary, "bins": bins_dict, "total_responses": int(total), "custom_binning": bin_config}
