import os
import random
import pandas as pd
import numpy as np
import inspect
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """Handles synthetic dataset generation for testing fraud detection"""
    
    def __init__(self, get_parameter_func, log_func, original_df=None):
        self.get_parameter_value = get_parameter_func
        self.log = log_func
        self.original_df = original_df

    def generate_synthetic_datasets(self, baseline_data, output_dir):
        """Generate synthetic datasets for testing"""
        print("DEBUG: generate_synthetic_datasets() start")
        synth_flws = int(self.get_parameter_value('synth_flws', 15))
        synth_visits = int(self.get_parameter_value('synth_visits', 200))
        seed = int(self.get_parameter_value('seed', 42))
        print(f"DEBUG: synth_flws={synth_flws}, synth_visits={synth_visits}, seed={seed}")
        
        random.seed(seed)
        np.random.seed(seed)
        
        fields = baseline_data.get('fields', {})
        if not fields:
            self.log("No fields found in baseline for synthetic generation")
            print("DEBUG: no fields in baseline; aborting synthetic generation")
            return []
        
        schema = {}
        numeric_edges = {}
        
        for field, info in fields.items():
            if info['type'] == 'categorical':
                cats = list(info['categories'].keys())
                probs = [info['categories'][c]['proportion'] for c in cats]
                s = sum(probs)
                probs = [p / s if s > 0 else 0 for p in probs]
                schema[field] = ('categorical', cats, probs)
            elif info['type'] == 'numeric':
                bins = list(info['bins'].keys())
                probs = [info['bins'][b]['proportion'] for b in bins]
                s = sum(probs)
                probs = [p / s if s > 0 else 0 for p in probs]
                edges = [float(b.split('-')[0]) for b in bins]
                edges.append(float(bins[-1].split('-')[1]))
                schema[field] = ('numeric', bins, probs)
                numeric_edges[field] = edges
        
        print(f"DEBUG: schema built for {len(schema)} fields")
        
        # Generate datasets and save to output_dir (analysis subdirectory)
        bad_path = os.path.join(output_dir, "synthetic_test_glaring.csv")
        subtle_path = os.path.join(output_dir, "synthetic_test_subtle.csv")
        duplicator_path = os.path.join(output_dir, "synthetic_test_duplicator.csv")
        single_field_path = os.path.join(output_dir, "synthetic_test_single_field_extremes.csv")
        
        bad_df = self._sample_dataset(schema, numeric_edges, synth_flws, synth_visits, mode='bad')
        subtle_df = self._sample_dataset(schema, numeric_edges, synth_flws, synth_visits, mode='subtle')
        duplicator_df = self._generate_duplicator_dataset()
        single_field_df = self._generate_single_field_extremes_dataset()
        
        print(f"DEBUG: writing synthetic CSVs -> {bad_path}, {subtle_path}, {duplicator_path}, {single_field_path}")
        bad_df.to_csv(bad_path, index=False)
        subtle_df.to_csv(subtle_path, index=False)
        duplicator_df.to_csv(duplicator_path, index=False)
        single_field_df.to_csv(single_field_path, index=False)
        
        self.log(f"Synthetic datasets saved: {bad_path}, {subtle_path}, {duplicator_path}, {single_field_path}")
        return [bad_path, subtle_path, duplicator_path, single_field_path]

    def _sample_dataset(self, schema, numeric_edges, n_flws, visits_per_flw, mode='bad'):
        """Generate a single synthetic dataset"""
        print(f"DEBUG: _sample_dataset(mode={mode}) n_flws={n_flws}, visits_per_flw={visits_per_flw}")
        rows = []
        
        if mode == 'bad':
            p_constant = 0.95
            blank_boost = 0.5
            improbable_boost = 0.8
        else:
            p_constant = 0.70
            blank_boost = 0.10
            improbable_boost = 0.20
        
        flw_ids = [f"SYN{mode[0].upper()}_{i:03d}" for i in range(1, n_flws + 1)]
        fields = list(schema.keys())
        
        # Generate date range for visits
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        date_range_days = (end_date - start_date).days
        
        # Pre-calculate target values and patterns for each field
        field_modes = {}
        field_leasts = {}
        has_blank = {}
        
        for field, (ftype, keys, probs) in schema.items():
            if not keys:
                continue
            idx_mode = int(np.argmax(probs))
            idx_min = int(np.argmin(probs))
            field_modes[field] = keys[idx_mode]
            field_leasts[field] = keys[idx_min]
            has_blank[field] = any(str(k) == "__BLANK__" for k in keys)
        
        visit_id_counter = 1
        
        for flw_idx, flw in enumerate(flw_ids):
            # Generate visit dates for this FLW
            flw_visit_dates = []
            for v in range(visits_per_flw):
                days_offset = (v / visits_per_flw) * date_range_days
                days_offset += random.uniform(-3, 3)  # Add some randomness
                visit_date = start_date + timedelta(days=max(0, days_offset))
                
                # Add time components
                hour = random.randint(8, 17)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                microsecond = random.randint(0, 999999)
                visit_date = visit_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
                
                flw_visit_dates.append(visit_date)
            
            flw_visit_dates.sort()
            
            for v, visit_date in enumerate(flw_visit_dates):
                row = {
                    "flw_id": flw,
                    "visit_id": f"VISIT_{visit_id_counter:06d}",
                    "visit_date": visit_date.strftime('%Y-%m-%d %H:%M:%S.%f+00:00'),
                    "flw_name": f"Synthetic Worker {flw_idx + 1}",
                    "opportunity_id": f"OPP_{(flw_idx % 5) + 1:03d}",
                    "opportunity_name": f"Test Opportunity {(flw_idx % 5) + 1}"
                }
                visit_id_counter += 1
                
                # Generate values for each field based on schema
                for field, (ftype, keys, probs) in schema.items():
                    if ftype == 'categorical':
                        p = probs.copy()
                        target_key = field_leasts[field] if mode == 'bad' else field_modes[field]
                        
                        try:
                            idx_target = keys.index(target_key)
                        except ValueError:
                            idx_target = int(np.argmax(p))
                        
                        p = np.array(p, dtype=float)
                        p[idx_target] += improbable_boost
                        
                        if has_blank[field]:
                            try:
                                idx_blank = keys.index("__BLANK__")
                                p[idx_blank] += blank_boost
                            except ValueError:
                                pass
                        
                        s = p.sum()
                        p = (p / s) if s > 0 else np.array([1.0 / len(keys)] * len(keys))
                        
                        if np.random.rand() < p_constant:
                            val = target_key
                        else:
                            val = np.random.choice(keys, p=p)
                        
                        row[field] = None if str(val) == "__BLANK__" else val
                        
                    else:  # numeric
                        bins = schema[field][1]
                        probs_bins = np.array(schema[field][2], dtype=float)
                        target_bin = field_leasts[field] if mode == 'bad' else field_modes[field]
                        
                        try:
                            idx_target = bins.index(target_bin)
                        except ValueError:
                            idx_target = int(np.argmax(probs_bins))
                        
                        probs_bins[idx_target] += improbable_boost
                        s = probs_bins.sum()
                        probs_bins = (probs_bins / s) if s > 0 else np.array([1.0 / len(bins)] * len(bins))
                        
                        if np.random.rand() < p_constant:
                            chosen = target_bin
                        else:
                            chosen = np.random.choice(bins, p=probs_bins)
                        
                        # Convert bin to actual numeric value
                        edges = numeric_edges[field]
                        try:
                            i = bins.index(chosen)
                            lo, hi = edges[i], edges[i + 1]
                            val = lo if hi <= lo else float(np.random.uniform(lo, hi))
                        except Exception:
                            lo, hi = edges[0], edges[1]
                            val = float((lo + hi) / 2.0)
                        
                        row[field] = val
                
                rows.append(row)
        
        return pd.DataFrame(rows)

    def _generate_duplicator_dataset(self):
        """Generate dataset by creating FLWs with 100 visits each, where each FLW's visits
        are 10 different original rows duplicated 10 times each"""
        print("DEBUG: _generate_duplicator_dataset() start")
        
        # Use actual original data if available, otherwise fall back to sample data
        if self.original_df is not None and len(self.original_df) > 0:
            print("DEBUG: Using actual original dataset for duplicator")
            sample_data = self.original_df.copy()
        else:
            print("DEBUG: No original data available, using artificial sample data")
            sample_data = self._create_sample_realistic_data(num_rows=100)
        
        # Configuration
        num_flws = 20  # Number of fraudulent FLWs to create
        visits_per_flw = 100  # Each FLW gets 100 visits
        templates_per_flw = 10  # Each FLW uses 10 different template rows
        duplicates_per_template = 10  # Each template is duplicated 10 times
        
        all_rows = []
        visit_id_counter = 1
        
        for flw_idx in range(num_flws):
            flw_id = f"SYNDUP_{flw_idx+1:03d}"
            flw_name = f"Synthetic Duplicator {flw_idx+1}"
            
            # Pick 10 random template rows for this FLW from REAL data
            if len(sample_data) >= templates_per_flw:
                template_rows = sample_data.sample(n=templates_per_flw, random_state=flw_idx+100)
            else:
                # If not enough sample data, sample with replacement
                template_rows = sample_data.sample(n=templates_per_flw, random_state=flw_idx+100, replace=True)
            
            # For each template row, create 10 duplicates
            for template_idx, (_, template_row) in enumerate(template_rows.iterrows()):
                for duplicate_num in range(duplicates_per_template):
                    new_row = template_row.copy()
                    new_row['flw_id'] = flw_id
                    new_row['flw_name'] = flw_name
                    new_row['visit_id'] = f"VISIT_{visit_id_counter:06d}"
                    
                    # Generate visit dates spread over 6 months
                    base_date = datetime.now() - timedelta(days=180)
                    days_offset = (template_idx * duplicates_per_template + duplicate_num) * 1.8  # ~1.8 days apart
                    visit_date = base_date + timedelta(days=days_offset)
                    new_row['visit_date'] = visit_date.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')
                    
                    # Assign opportunity info
                    new_row['opportunity_id'] = f"OPP_{(flw_idx % 5) + 1:03d}"
                    new_row['opportunity_name'] = f"Duplicator Test Opportunity {(flw_idx % 5) + 1}"
                    
                    all_rows.append(new_row)
                    visit_id_counter += 1
        
        result_df = pd.DataFrame(all_rows)
        print(f"DEBUG: duplicator dataset created with {len(result_df)} rows ({num_flws} FLWs × {visits_per_flw} visits)")
        return result_df

    def _generate_single_field_extremes_dataset(self):
        """Generate dataset with one FLW per field, each with extreme pattern in that field.
        All FLWs use the same 100 base rows for clean comparison."""
        print("DEBUG: _generate_single_field_extremes_dataset() start")
        
        # Define fields to test with their extreme patterns
        field_extremes = {
            'have_glasses': {'pattern': 'always_yes', 'name': 'Glasses Cheater'},
            'diarrhea_last_month': {'pattern': 'high_yes', 'pct': 0.80, 'name': 'Diarrhea Cheater'},
            'diagnosed_with_mal_past_3_months': {'pattern': 'high_yes', 'pct': 0.60, 'name': 'Malnutrition Cheater'},
            'childs_gender': {'pattern': 'gender_bias', 'pct': 0.95, 'value': 'male_child', 'name': 'Gender Cheater'},
            'va_child_unwell_today': {'pattern': 'high_yes', 'pct': 0.70, 'name': 'Unwell Cheater'},
            'muac_consent': {'pattern': 'never_yes', 'name': 'MUAC Consent Cheater'},
            'childs_age_in_month': {'pattern': 'age_cluster', 'min_age': 12, 'max_age': 18, 'name': 'Age Cheater'},
            'under_treatment_for_mal': {'pattern': 'high_yes', 'pct': 0.50, 'name': 'Treatment Cheater'},
            'va_consent': {'pattern': 'never_yes', 'name': 'VA Consent Cheater'},
            'received_any_vaccine': {'pattern': 'high_yes', 'pct': 0.90, 'name': 'Vaccine Cheater'},
            'muac_colour': {'pattern': 'always_red', 'name': 'MUAC Color Cheater'},
            'received_va_dose_before': {'pattern': 'high_yes', 'pct': 0.80, 'name': 'VA Dose Cheater'},
            'recent_va_dose': {'pattern': 'high_yes', 'pct': 0.75, 'name': 'Recent VA Cheater'},
            'no_of_children': {'pattern': 'always_same', 'value': 7, 'name': 'Children Count Cheater'},
            'soliciter_muac_cm': {'pattern': 'narrow_range', 'min_val': 10.5, 'max_val': 11.0, 'name': 'MUAC Measurement Cheater'}
        }
        
        # Generate exactly 100 base realistic rows (same for all FLWs)
        print("DEBUG: generating 100 base rows for all FLWs")
        base_rows = self._create_sample_realistic_data(num_rows=100)
        
        all_rows = []
        visit_id_counter = 1
        
        for field_idx, (field_name, config) in enumerate(field_extremes.items()):
            flw_id = f"SYNEXT_{field_idx+1:03d}"
            flw_name = f"Synthetic {config['name']}"
            
            print(f"DEBUG: creating FLW {flw_id} for field '{field_name}' with pattern '{config['pattern']}'")
            
            # Use the SAME 100 base rows for every FLW
            for row_idx, (_, row) in enumerate(base_rows.iterrows()):
                new_row = row.copy()
                new_row['flw_id'] = flw_id
                new_row['flw_name'] = flw_name
                new_row['visit_id'] = f"VISIT_{visit_id_counter:06d}"
                
                # Generate visit date (spread over 6 months)
                base_date = datetime.now() - timedelta(days=180)
                visit_date = base_date + timedelta(days=row_idx * 1.8)  # ~1.8 days apart
                new_row['visit_date'] = visit_date.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')
                
                # Keep opportunity info
                new_row['opportunity_id'] = f"OPP_{(field_idx % 5) + 1:03d}"
                new_row['opportunity_name'] = f"Extreme Test Opportunity {(field_idx % 5) + 1}"
                
                # Apply extreme pattern to the target field ONLY
                if field_name in new_row.index:
                    if config['pattern'] == 'always_yes':
                        new_row[field_name] = 'yes'
                    elif config['pattern'] == 'never_yes':
                        new_row[field_name] = 'no'
                    elif config['pattern'] == 'high_yes':
                        # Set to 'yes' with specified probability
                        if random.random() < config['pct']:
                            new_row[field_name] = 'yes'
                        else:
                            new_row[field_name] = 'no'
                    elif config['pattern'] == 'gender_bias':
                        # Set to specified gender with high probability
                        if random.random() < config['pct']:
                            new_row[field_name] = config['value']
                        else:
                            new_row[field_name] = 'female_child' if config['value'] == 'male_child' else 'male_child'
                    elif config['pattern'] == 'age_cluster':
                        # Cluster ages in specific range
                        new_row[field_name] = random.uniform(config['min_age'], config['max_age'])
                    elif config['pattern'] == 'always_red':
                        new_row[field_name] = 'red'
                    elif config['pattern'] == 'always_same':
                        new_row[field_name] = config['value']
                    elif config['pattern'] == 'narrow_range':
                        # Cluster numeric values in narrow range
                        new_row[field_name] = random.uniform(config['min_val'], config['max_val'])
                
                all_rows.append(new_row)
                visit_id_counter += 1
        
        result_df = pd.DataFrame(all_rows)
        print(f"DEBUG: single field extremes dataset created with {len(result_df)} rows, {len(field_extremes)} FLWs")
        return result_df

    def _create_sample_realistic_data(self, num_rows=50):
        """Create sample realistic data when original data isn't available"""
        print(f"DEBUG: creating {num_rows} sample realistic rows")
        
        # Create specified number of sample realistic rows
        sample_rows = []
        
        for i in range(num_rows):
            row = {
                'flw_id': f'REAL_{i:03d}',
                'visit_id': f'VISIT_{i:06d}',
                'visit_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d %H:%M:%S.%f+00:00'),
                'flw_name': f'Real Worker {i}',
                'opportunity_id': f'OPP_{(i % 5) + 1:03d}',
                'opportunity_name': f'Real Opportunity {(i % 5) + 1}',
                
                # Realistic field values based on typical distributions
                'childs_age_in_month': random.uniform(0, 59),
                'childs_gender': random.choice(['male_child', 'female_child']),
                'have_glasses': 'yes' if random.random() < 0.03 else 'no',  # 3% have glasses
                'diarrhea_last_month': 'yes' if random.random() < 0.15 else 'no',  # 15% diarrhea
                'diagnosed_with_mal_past_3_months': 'yes' if random.random() < 0.08 else 'no',  # 8% malnutrition
                'va_child_unwell_today': 'yes' if random.random() < 0.05 else ('no' if random.random() < 0.8 else ''),  # 5% unwell, 20% blank
                'muac_consent': 'yes' if random.random() < 0.75 else 'no',  # 75% consent
                'va_consent': 'yes' if random.random() < 0.78 else 'no',  # 78% consent
                'under_treatment_for_mal': 'yes' if random.random() < 0.03 else 'no',  # 3% under treatment
                'muac_colour': random.choice(['green', 'yellow', 'red']) if random.random() < 0.2 else '',  # 20% measured
                'soliciter_muac_cm': random.uniform(9.0, 15.0) if random.random() < 0.18 else None,  # 18% measured
                'no_of_children': random.randint(1, 8),
                'received_va_dose_before': 'yes' if random.random() < 0.20 else 'no',  # 20% received before
                'recent_va_dose': 'yes' if random.random() < 0.15 else 'no',  # 15% recent dose
                'received_any_vaccine': 'yes' if random.random() < 0.25 else 'no',  # 25% vaccinated
                'child_name': f'Child_{i}' if random.random() < 0.98 else '',  # 2% blank
                'household_phone': f'123-456-{i:04d}' if random.random() < 0.55 else '',  # 45% blank
                'household_name': f'Household_{i}' if random.random() < 0.97 else ''  # 3% blank
            }
            sample_rows.append(row)
        
        return pd.DataFrame(sample_rows)