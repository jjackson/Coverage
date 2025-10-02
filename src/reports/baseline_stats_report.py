"""
Enhanced Baseline Stats Report
Generates aggregated distributions for categorical and numeric fields
from a trusted dataset, to be used as a baseline reference for
similarity/outlier detection algorithms (e.g., MMA/SVA).

Enhanced features:
- Stores which fields treat blanks as valid categories
- Calculates similarity scores for all baseline FLWs
- Custom binning for child_age_months and muac_measurement_cm
- Generates synthetic test datasets
"""

import os
import json
import re
import math
import random
import pandas as pd
import numpy as np
from datetime import datetime
from .base_report import BaseReport

class BaselineStatsReport(BaseReport):
    """Generates baseline distributions for each field and optional synthetic datasets"""

    @staticmethod
    def setup_parameters(parent_frame):
        import tkinter as tk
        from tkinter import ttk

        # Minimum number of responses for field to be included
        ttk.Label(parent_frame, text="Min responses to include field:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        min_responses_var = tk.StringVar(value="50")
        ttk.Entry(parent_frame, textvariable=min_responses_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Bin size for numeric variables (default, overridden for special fields)
        ttk.Label(parent_frame, text="Default numeric bin size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        bin_size_var = tk.StringVar(value="1")
        ttk.Entry(parent_frame, textvariable=bin_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Synthetic data generation controls
        ttk.Label(parent_frame, text="Synthetic test datasets:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=6)
        gen_synth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, text="Generate glaring + subtle synthetic CSVs", variable=gen_synth_var).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(parent_frame, text="# FLWs (per synthetic set):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        synth_flws_var = tk.StringVar(value="15")
        ttk.Entry(parent_frame, textvariable=synth_flws_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(parent_frame, text="# visits per FLW:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        synth_visits_var = tk.StringVar(value="200")
        ttk.Entry(parent_frame, textvariable=synth_visits_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(parent_frame, text="Random seed:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        seed_var = tk.StringVar(value="42")
        ttk.Entry(parent_frame, textvariable=seed_var, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # Store variables
        parent_frame.min_responses_var = min_responses_var
        parent_frame.bin_size_var = bin_size_var
        parent_frame.gen_synth_var = gen_synth_var
        parent_frame.synth_flws_var = synth_flws_var
        parent_frame.synth_visits_var = synth_visits_var
        parent_frame.seed_var = seed_var

    def generate(self):
        output_files = []

        min_responses = int(self.get_parameter_value('min_responses', 50))
        bin_size = float(self.get_parameter_value('bin_size', 1))
        gen_synth = self.get_parameter_value('gen_synth', True)
        synth_flws = int(self.get_parameter_value('synth_flws', 15))
        synth_visits = int(self.get_parameter_value('synth_visits', 200))
        seed = int(self.get_parameter_value('seed', 42))

        # Special handling lists
        special_numeric_fields = {"muac_cm", "age_months", "age_years"}
        max_categories = 20

        # Fields where we treat blanks as a valid category (value suffixes removed)
        fields_with_blank_as_category = {
            "muac_vitals_color",
            "child_unwell_today",
        }

        # Custom binning configuration
        custom_binning = {
            "child_age_months": {
                "type": "interval",
                "start": 0,
                "size": 6,
                "description": "6-month age groups"
            },
            "muac_measurement_cm": {
                "type": "fixed_bins", 
                "start": 8.5,
                "size": 1.0,
                "description": "1cm bins starting from 8.5cm"
            }
        }

        # Focus on specific fields for initial analysis
        target_fields = {
            "child_age_months",      # numeric with custom binning
            "muac_measurement_cm",   # numeric with custom binning  
            "child_gender",          # categorical
            "malnutrition_diagnosed_recently",  # categorical
            "muac_vitals_color",     # categorical, blank-sensitive
            "child_unwell_today",    # categorical, blank-sensitive
        }

        df = self.df.copy()
        total_visits = len(df)

        baseline_data = {
            "metadata": {
                "source_file": None,
                "date_generated": datetime.utcnow().isoformat(),
                "total_visits": int(total_visits),
                "total_flws": df['flw_id'].nunique() if 'flw_id' in df.columns else None,
                "notes": "Enhanced baseline with FLW similarity scores and custom binning",
                "included_fields": [],
                "method": "Aggregated per-question distributions; special fields include '__BLANK__' as category",
                "fields_with_blank_as_category": list(fields_with_blank_as_category),
                "custom_binning": custom_binning
            },
            "fields": {}
        }

        flat_rows = []  # for CSV output

        for col in df.columns:
            if col.lower() in ['flw_id', 'visit_id']:
                continue

            # Only process target fields for focused analysis
            if col not in target_fields:
                continue

            # Decide how to read series for this column
            series = df[col]
            col_lower = col.lower()

            # Skip datetime entirely
            if pd.api.types.is_datetime64_any_dtype(series):
                continue

            # Normalize a key to check against the blank-sensitive set
            norm_key = col_lower
            if norm_key.startswith('pct_'):
                norm_key = norm_key[4:]
            norm_key = re.sub(r'_(yes|no|green|red)$', '', norm_key)

            # For special blank-sensitive fields: keep NaN as '__BLANK__' so it becomes a valid category
            if norm_key in fields_with_blank_as_category:
                series = series.fillna("__BLANK__")
            else:
                series = series.dropna()

            if len(series) < min_responses:
                continue  # skip sparse fields

            # Check for custom binning first
            field_entry = None
            if col in custom_binning:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric_series) >= min_responses:
                    field_entry = self._process_custom_binning(numeric_series, custom_binning[col])

            # If no custom binning, use standard logic
            if field_entry is None:
                if (col_lower in special_numeric_fields):
                    # Force numeric binning for important continuous fields
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    if len(numeric_series) < min_responses:
                        continue
                    field_entry = self._process_numeric(numeric_series, bin_size)
                elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                    # For general numeric columns, only include if low-cardinality; otherwise skip
                    if series.nunique() <= max_categories:
                        field_entry = self._process_categorical(series)
                    else:
                        continue
                else:
                    # Treat as categorical if unique values are limited
                    if series.nunique() <= max_categories:
                        field_entry = self._process_categorical(series)
                    else:
                        continue

            if field_entry is not None:
                baseline_data["fields"][col] = field_entry
                baseline_data["metadata"]["included_fields"].append(col)

                # Flatten for CSV
                if field_entry["type"] == "categorical":
                    for cat, stats in field_entry["categories"].items():
                        flat_rows.append([col, cat, stats["count"], stats["proportion"]])
                else:
                    for bin_label, stats in field_entry["bins"].items():
                        flat_rows.append([col, bin_label, stats["count"], stats["proportion"]])

        # Calculate similarity scores for all baseline FLWs
        self.log("Calculating similarity scores for baseline FLWs...")
        baseline_similarities = self._calculate_baseline_similarities(df, baseline_data, fields_with_blank_as_category)
        baseline_data["baseline_flw_similarities"] = baseline_similarities

        # Save JSON
        json_path = os.path.join(self.output_dir, "baseline_stats.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        output_files.append(json_path)

        # Save CSV
        csv_df = pd.DataFrame(flat_rows, columns=["field_name", "category_or_bin", "count", "proportion"])
        csv_path = os.path.join(self.output_dir, "baseline_stats.csv")
        csv_df.to_csv(csv_path, index=False)
        output_files.append(csv_path)

        self.log(f"Baseline stats saved to {json_path} and {csv_path}")

        # Optionally generate synthetic datasets using the baseline
        if gen_synth:
            try:
                bad_path, subtle_path = self._generate_synthetic_datasets(
                    baseline_data=baseline_data,
                    n_flws=synth_flws,
                    visits_per_flw=synth_visits,
                    seed=seed
                )
                if bad_path:
                    output_files.append(bad_path)
                if subtle_path:
                    output_files.append(subtle_path)
            except Exception as e:
                self.log(f"Synthetic data generation failed: {e}")

        return output_files

    def _process_categorical(self, series):
        counts = series.value_counts(dropna=False)
        total = counts.sum()
        categories = {
            str(cat): {
                "count": int(count),
                "proportion": float(count / total)
            }
            for cat, count in counts.items()
        }
        return {
            "type": "categorical",
            "categories": categories,
            "total_responses": int(total)
        }

    def _process_numeric(self, series, bin_size):
        total = series.count()
        summary = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "quartiles": [float(q) for q in np.percentile(series, [25, 50, 75])]
        }

        # Create bins
        bin_edges = np.arange(series.min(), series.max() + bin_size, bin_size)
        if len(bin_edges) < 2:
            bin_edges = [series.min(), series.max() + bin_size]
        bin_labels = [f"{round(bin_edges[i],1)}-{round(bin_edges[i+1],1)}" for i in range(len(bin_edges)-1)]
        counts = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True).value_counts().sort_index()
        bins_dict = {
            label: {
                "count": int(count),
                "proportion": float(count / total)
            }
            for label, count in counts.items()
        }

        return {
            "type": "numeric",
            "summary": summary,
            "bins": bins_dict,
            "total_responses": int(total)
        }

    def _process_custom_binning(self, series, bin_config):
        """Process numeric field with custom binning rules"""
        total = series.count()
        summary = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "quartiles": [float(q) for q in np.percentile(series, [25, 50, 75])]
        }

        if bin_config["type"] == "interval":
            # Age groups: 0-6, 6-12, 12-18, etc.
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
            # MUAC: 8.5-9.5, 9.5-10.5, etc.
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
            # Fallback to standard binning
            return self._process_numeric(series, 1.0)

        counts = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True).value_counts().sort_index()
        bins_dict = {
            label: {
                "count": int(count),
                "proportion": float(count / total)
            }
            for label, count in counts.items()
        }

        return {
            "type": "numeric",
            "summary": summary,
            "bins": bins_dict,
            "total_responses": int(total),
            "custom_binning": bin_config
        }

    def _calculate_baseline_similarities(self, df, baseline_data, fields_with_blank_as_category):
        """Calculate similarity scores for all FLWs in the baseline dataset"""
        excluded_fields = {
            "opportunity_name",
            "status", 
            "unit_name",
            "opportunity_id",
            "flagged"
        }

        similarities = []
        flw_details = []

        for flw_id, group in df.groupby("flw_id"):
            flw_similarities = []
            
            for field_name, field_info in baseline_data["fields"].items():
                if field_name.lower() in excluded_fields:
                    continue
                    
                if field_info["type"] == "categorical":
                    if field_name in group.columns:
                        # Use same logic as ranking report
                        norm_key = field_name.lower()
                        if norm_key.startswith('pct_'):
                            norm_key = norm_key[4:]
                        norm_key = re.sub(r'_(yes|no|green|red)$', '', norm_key)
                        
                        if norm_key in fields_with_blank_as_category:
                            values = group[field_name].fillna("__BLANK__").astype(str)
                        else:
                            values = group[field_name].dropna().astype(str)
                            
                        if len(values) > 0:
                            baseline_dist = {cat: info["proportion"] for cat, info in field_info["categories"].items()}
                            flw_dist = values.value_counts(normalize=True).to_dict()
                            sim_score = sum(min(baseline_dist.get(cat, 0), flw_dist.get(cat, 0)) for cat in baseline_dist)
                            flw_similarities.append(sim_score)
                            
                elif field_info["type"] == "numeric":
                    if field_name in group.columns:
                        values = pd.to_numeric(group[field_name], errors="coerce").dropna()
                        if len(values) > 0:
                            baseline_bins = field_info["bins"]
                            flw_bins = self._calculate_numeric_similarity_distribution(values, baseline_bins)
                            sim_score = sum(min(baseline_bins.get(bin_label, {}).get("proportion", 0),
                                                flw_bins.get(bin_label, 0)) for bin_label in baseline_bins)
                            flw_similarities.append(sim_score)
            
            # Calculate overall similarity for this FLW
            overall_similarity = sum(flw_similarities) / len(flw_similarities) if flw_similarities else None
            if overall_similarity is not None:
                similarities.append(overall_similarity)
                flw_details.append({
                    "flw_id": flw_id,
                    "similarity_score": round(overall_similarity, 9),
                    "n_visits": len(group),
                    "fields_evaluated": len(flw_similarities)
                })

        # Calculate statistics
        if similarities:
            similarities_array = np.array(similarities)
            percentiles = {
                f"p{p}": float(np.percentile(similarities_array, p))
                for p in [5, 10, 25, 50, 75, 90, 95]
            }
            
            stats = {
                "scores": [round(s, 9) for s in similarities],
                "count": len(similarities),
                "mean": float(np.mean(similarities_array)),
                "std": float(np.std(similarities_array, ddof=1)) if len(similarities) > 1 else 0.0,
                "min": float(np.min(similarities_array)),
                "max": float(np.max(similarities_array)),
                "percentiles": percentiles,
                "flw_details": flw_details
            }
        else:
            stats = {
                "scores": [],
                "count": 0,
                "mean": None,
                "std": None,
                "min": None, 
                "max": None,
                "percentiles": {},
                "flw_details": []
            }

        return stats

    def _calculate_numeric_similarity_distribution(self, values, baseline_bins):
        """Calculate numeric distribution for similarity comparison"""
        # Parse bin edges from baseline bin labels
        bin_edges = []
        bin_labels = list(baseline_bins.keys())
        
        for label in bin_labels:
            try:
                start = float(label.split('-')[0])
                bin_edges.append(start)
            except (ValueError, IndexError):
                continue
        
        if bin_labels:
            try:
                end = float(bin_labels[-1].split('-')[1])
                bin_edges.append(end)
            except (ValueError, IndexError):
                if bin_edges:
                    bin_edges.append(bin_edges[-1] + 1)
        
        if len(bin_edges) < 2:
            return {}
        
        try:
            binned = pd.cut(values, bins=bin_edges, labels=bin_labels, include_lowest=True)
            flw_distribution = binned.value_counts(normalize=True).to_dict()
        except Exception:
            flw_distribution = {}
            
        return flw_distribution

    # -------------------------- Synthetic data generation --------------------------
    def _generate_synthetic_datasets(self, baseline_data, n_flws, visits_per_flw, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Build a sampling schema from baseline distributions
        fields = baseline_data.get('fields', {})
        if not fields:
            self.log("No fields found in baseline for synthetic generation")
            return None, None

        schema = {}
        numeric_edges = {}
        for field, info in fields.items():
            if info['type'] == 'categorical':
                cats = list(info['categories'].keys())
                probs = [info['categories'][c]['proportion'] for c in cats]
                # Normalize to protect against rounding
                s = sum(probs)
                probs = [p / s if s > 0 else 0 for p in probs]
                schema[field] = ('categorical', cats, probs)
            elif info['type'] == 'numeric':
                bins = list(info['bins'].keys())
                probs = [info['bins'][b]['proportion'] for b in bins]
                s = sum(probs)
                probs = [p / s if s > 0 else 0 for p in probs]
                # Parse edges from labels like "13.0-14.0"
                edges = [float(b.split('-')[0]) for b in bins]
                edges.append(float(bins[-1].split('-')[1]))
                schema[field] = ('numeric', bins, probs)
                numeric_edges[field] = edges

        # File paths
        bad_path = os.path.join(self.output_dir, "synthetic_test_glaring.csv")
        subtle_path = os.path.join(self.output_dir, "synthetic_test_subtle.csv")

        # Generate both datasets
        bad_df = self._sample_dataset(schema, numeric_edges, n_flws, visits_per_flw, mode='bad')
        subtle_df = self._sample_dataset(schema, numeric_edges, n_flws, visits_per_flw, mode='subtle')

        # Save
        bad_df.to_csv(bad_path, index=False)
        subtle_df.to_csv(subtle_path, index=False)
        self.log(f"Synthetic datasets saved: {bad_path}, {subtle_path}")
        return bad_path, subtle_path

    def _sample_dataset(self, schema, numeric_edges, n_flws, visits_per_flw, mode='bad'):
        from datetime import datetime, timedelta
        
        rows = []

        # Parameters controlling anomaly strength
        if mode == 'bad':
            # Very strong anomalies: almost-constant values, improbable choices, high blanks share
            p_constant = 0.95
            blank_boost = 0.5   # extra probability mass to BLANK where available (capped)
            improbable_boost = 0.8  # shift mass to least-likely category/bin
        else:
            # Subtle anomalies: small shifts and modest blanks
            p_constant = 0.70
            blank_boost = 0.10
            improbable_boost = 0.20

        flw_ids = [f"SYN{mode[0].upper()}_{i:03d}" for i in range(1, n_flws + 1)]
        fields = list(schema.keys())

        # Generate realistic visit dates
        # Start from 6 months ago, spread visits over time
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months back
        date_range_days = (end_date - start_date).days

        # Precompute mode/least-likely for each field
        field_modes = {}
        field_leasts = {}
        has_blank = {}
        for field, (ftype, keys, probs) in schema.items():
            if not keys:
                continue
            # Index of most and least likely
            idx_mode = int(np.argmax(probs))
            idx_min = int(np.argmin(probs))
            field_modes[field] = keys[idx_mode]
            field_leasts[field] = keys[idx_min]
            has_blank[field] = any(str(k) == "__BLANK__" for k in keys)

        visit_id_counter = 1
        
        for flw_idx, flw in enumerate(flw_ids):
            # Generate visit dates for this FLW (roughly 1-2 visits per week)
            flw_visit_dates = []
            for v in range(visits_per_flw):
                # Spread visits somewhat evenly but with realistic variation
                days_offset = (v / visits_per_flw) * date_range_days
                # Add some random variation (±3 days)
                days_offset += random.uniform(-3, 3)
                visit_date = start_date + timedelta(days=max(0, days_offset))
                # Add realistic time component
                hour = random.randint(8, 17)  # Work hours
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                microsecond = random.randint(0, 999999)
                visit_date = visit_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
                flw_visit_dates.append(visit_date)
            
            # Sort dates to be chronological
            flw_visit_dates.sort()
            
            for v, visit_date in enumerate(flw_visit_dates):
                row = {
                    "flw_id": flw,
                    "visit_id": f"VISIT_{visit_id_counter:06d}",
                    "visit_date": visit_date.strftime('%Y-%m-%d %H:%M:%S.%f+00:00'),
                    "flw_name": f"Synthetic Worker {flw_idx + 1}",
                    "opportunity_id": f"OPP_{(flw_idx % 5) + 1:03d}",  # Cycle through 5 opportunities
                    "opportunity_name": f"Test Opportunity {(flw_idx % 5) + 1}"
                }
                visit_id_counter += 1
                for field, (ftype, keys, probs) in schema.items():
                    if ftype == 'categorical':
                        # Build a working probability vector
                        p = probs.copy()
                        # For bad: concentrate on least-likely; for subtle: slightly
                        target_key = field_leasts[field] if mode == 'bad' else field_modes[field]
                        try:
                            idx_target = keys.index(target_key)
                        except ValueError:
                            idx_target = int(np.argmax(p))

                        # Start with baseline; apply boost to target
                        p = np.array(p, dtype=float)
                        p[idx_target] += improbable_boost
                        # Add blank boost if present
                        if has_blank[field]:
                            try:
                                idx_blank = keys.index("__BLANK__")
                                p[idx_blank] += blank_boost
                            except ValueError:
                                pass
                        # Normalize
                        s = p.sum()
                        if s > 0:
                            p = p / s
                        else:
                            p = np.array([1.0 / len(keys)] * len(keys))

                        # Draw value, occasionally force constant behavior
                        if np.random.rand() < p_constant:
                            val = target_key
                        else:
                            val = np.random.choice(keys, p=p)

                        # Map BLANK marker back to actual blank for the CSV
                        row[field] = None if str(val) == "__BLANK__" else val

                    else:  # numeric: sample a bin, then sample uniformly inside it
                        bins = schema[field][1]
                        probs_bins = np.array(schema[field][2], dtype=float)
                        # Apply boost similarly
                        target_bin = field_leasts[field] if mode == 'bad' else field_modes[field]
                        try:
                            idx_target = bins.index(target_bin)
                        except ValueError:
                            idx_target = int(np.argmax(probs_bins))
                        probs_bins[idx_target] += improbable_boost
                        s = probs_bins.sum()
                        if s > 0:
                            probs_bins = probs_bins / s
                        else:
                            probs_bins = np.array([1.0 / len(bins)] * len(bins))

                        if np.random.rand() < p_constant:
                            chosen = target_bin
                        else:
                            chosen = np.random.choice(bins, p=probs_bins)

                        # Sample inside numeric bin
                        edges = numeric_edges[field]
                        try:
                            i = bins.index(chosen)
                            lo, hi = edges[i], edges[i + 1]
                            # If zero width, just use lo
                            if hi <= lo:
                                val = lo
                            else:
                                val = float(np.random.uniform(lo, hi))
                        except Exception:
                            # Fallback: use mid of first bin
                            lo, hi = edges[0], edges[1]
                            val = float((lo + hi) / 2.0)
                        row[field] = val

                rows.append(row)

        return pd.DataFrame(rows)
