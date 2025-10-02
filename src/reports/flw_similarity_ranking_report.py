import os
import json
import re
import math
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from .base_report import BaseReport
from .fraud_detection_core import FraudDetectionCore

try:
    from .fraud_html_reporter import FraudHTMLReporter
    HTML_REPORTER_AVAILABLE = True
except ImportError:
    HTML_REPORTER_AVAILABLE = False
    print("WARNING: fraud_html_reporter not available")

try:
    from .descriptive_analyzer import run_standard_analysis
    DESCRIPTIVE_ANALYZER_AVAILABLE = True
except ImportError:
    DESCRIPTIVE_ANALYZER_AVAILABLE = False
    print("WARNING: descriptive_analyzer not available - experimental analysis will be disabled")

try:
    from .synthetic_data_generator import SyntheticDataGenerator
    SYNTHETIC_GENERATOR_AVAILABLE = True
except ImportError:
    SYNTHETIC_GENERATOR_AVAILABLE = False
    print("WARNING: synthetic_data_generator not available")

class UnifiedFLWAnomalyReport(BaseReport):
    """Unified report for FLW fraud detection using composite scoring - VERSION 3.1"""

    @staticmethod
    def setup_parameters(parent_frame):
        """Set up GUI parameters for this report type"""
        print("DEBUG: setup_parameters() called for UnifiedFLWAnomalyReport v3.1")
        row = 0
        
        # Baseline generation toggle
        ttk.Label(parent_frame, text="Generate Baseline:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.generate_baseline_var = tk.BooleanVar()
        ttk.Checkbutton(parent_frame, text="Treat current data as baseline", variable=parent_frame.generate_baseline_var).grid(row=row, column=1, sticky="w")
        row += 1

        # Optional tag to control output filenames
        ttk.Label(parent_frame, text="Optional tag for output filename:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.tag_var = tk.StringVar(value="")
        ttk.Entry(parent_frame, textvariable=parent_frame.tag_var, width=24).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Baseline-specific parameters
        ttk.Label(parent_frame, text="--- Baseline Generation Parameters ---", font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 5))
        row += 1
        
        # Minimum responses
        ttk.Label(parent_frame, text="Min responses to include field:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.min_responses_var = tk.StringVar(value="50")
        ttk.Entry(parent_frame, textvariable=parent_frame.min_responses_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Default bin size
        ttk.Label(parent_frame, text="Default numeric bin size:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.bin_size_var = tk.StringVar(value="1")
        ttk.Entry(parent_frame, textvariable=parent_frame.bin_size_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Synthetic data generation
        ttk.Label(parent_frame, text="Generate Synthetic Data:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.gen_synth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, text="Create test datasets", variable=parent_frame.gen_synth_var).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Synthetic data parameters
        ttk.Label(parent_frame, text="# FLWs (per synthetic set):").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.synth_flws_var = tk.StringVar(value="15")
        ttk.Entry(parent_frame, textvariable=parent_frame.synth_flws_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(parent_frame, text="# visits per FLW:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.synth_visits_var = tk.StringVar(value="200")
        ttk.Entry(parent_frame, textvariable=parent_frame.synth_visits_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(parent_frame, text="Random seed:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.seed_var = tk.StringVar(value="42")
        ttk.Entry(parent_frame, textvariable=parent_frame.seed_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Experimental analysis
        ttk.Label(parent_frame, text="Run Experimental Analysis:").grid(row=row, column=0, sticky="w", padx=(0, 5))
        parent_frame.experimental_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent_frame, text="Generate aggregation function analysis", variable=parent_frame.experimental_analysis_var).grid(row=row, column=1, sticky="w")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize core components
        self.fraud_core = FraudDetectionCore(self.df, self.get_parameter_value, self.log)
        
        if SYNTHETIC_GENERATOR_AVAILABLE:
            self.synthetic_generator = SyntheticDataGenerator(self.get_parameter_value, self.log, self.df)
        else:
            self.synthetic_generator = None

    def _sanitize_tag(self, tag):
        """Sanitize tag for use in filenames"""
        if tag is None:
            return ""
        tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(tag).strip())
        return tag.strip("._-")

    def _create_analysis_directory(self):
        """Create dated analysis subdirectory and return path"""
        date_str = datetime.utcnow().strftime('%Y%m%d')
        analysis_dir = os.path.join(self.output_dir, f"analysis_{date_str}")
        
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            print(f"DEBUG: Created analysis directory: {analysis_dir}")
        else:
            print(f"DEBUG: Analysis directory already exists: {analysis_dir}")
        
        return analysis_dir

    def _calculate_percent_female(self):
        """Calculate percentage of records where childs_gender is 'female'"""
        if 'childs_gender' not in self.df.columns:
            return 0.000  # Default if no childs_gender column found
        
        # Get non-null gender values
        gender_values = self.df['childs_gender'].dropna()
        
        if len(gender_values) == 0:
            return 0.000
        
        # Count 'female' values
        female_count = (gender_values == 'female_child').sum()
        
        pct_female = (female_count / len(gender_values)) * 100
        return round(pct_female, 3)

    def _get_synthetic_data_indicator(self):
        """Check if this appears to be synthetic/fake data and return appropriate indicator"""
        if 'flw_name' not in self.df.columns:
            return None
        
        flw_names = self.df['flw_name'].dropna().astype(str)
        
        # Check for "fake" first, then "synthetic"
        if flw_names.str.contains('Fake', case=False, na=False).any():
            return "_fake"
        elif flw_names.str.contains('Synthetic', case=False, na=False).any():
            return "_synthetic"
        else:
            return None

    def _is_synthetic_data(self):
        """Check if this appears to be synthetic data based on flw_name (for backwards compatibility)"""
        return self._get_synthetic_data_indicator() is not None

    def generate(self):
        """Main generation method - coordinates all components"""
        print("DEBUG: generate() entered")
        
        total_rows = len(self.df)
        pct_female = self._calculate_percent_female()
        output_files = []
        
        # Get parameters
        generate_baseline = self.get_parameter_value("generate_baseline", False)
        tag_raw = self.get_parameter_value("tag", "").strip()
        tag = self._sanitize_tag(tag_raw)
        
        print(f"DEBUG: output_dir={self.output_dir}")
        print(f"DEBUG: generate_baseline={generate_baseline}")
        print(f"DEBUG: raw tag='{tag_raw}' -> sanitized tag='{tag}'")

        # Create analysis subdirectory
        analysis_dir = self._create_analysis_directory()

        if generate_baseline:
            print("DEBUG: Path = BASELINE generation")
            
            # Generate baseline data using core module
            baseline_data = self.fraud_core.generate_baseline_data()
            print(f"DEBUG: Baseline fields included: {len(baseline_data.get('fields', {}))}")
            
            # Calculate fraud scores
            results_df = self.fraud_core.calculate_fraud_scores(baseline_data, is_baseline=True)
            
            # Save baseline stats to MAIN directory (not analysis subdir)
            json_path = os.path.join(self.output_dir, "baseline_stats.json")
            print(f"DEBUG: Writing baseline JSON -> {json_path}")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(baseline_data, f, indent=2)
            output_files.append(json_path)
            
            # Save flattened baseline stats CSV to MAIN directory
            csv_path = self.fraud_core.save_baseline_csv(baseline_data, self.output_dir)
            output_files.append(csv_path)
            
            # Generate synthetic datasets to ANALYSIS directory
            if self.get_parameter_value("gen_synth", True) and self.synthetic_generator:
                print("DEBUG: Generating synthetic datasets...")
                synth_files = self.synthetic_generator.generate_synthetic_datasets(baseline_data, analysis_dir)
                output_files.extend(synth_files)
            else:
                print("DEBUG: Skipping synthetic datasets")
                
            filename_suffix = "fraud_baseline"
        else:
            print("DEBUG: Path = FRAUD RANKINGS (using existing baseline)")
            
            # Load existing baseline from MAIN directory
            baseline_data = self.fraud_core.load_baseline_data(self.output_dir)
            print(f"DEBUG: Loaded baseline with {len(baseline_data.get('fields', {}))} fields")
            
            # Calculate fraud scores
            results_df = self.fraud_core.calculate_fraud_scores(baseline_data, is_baseline=False)
            filename_suffix = "fraud_rankings"
        
        # Decide filename for fraud rankings (goes to ANALYSIS directory)
        if tag:
            rankings_filename = f"flw_fraud_{filename_suffix}_{tag}.csv"
            print(f"DEBUG: Using tag-based filename: {rankings_filename}")
        else:
            # Use data characteristics instead of timestamp
            data_signature = f"{total_rows}rows_{pct_female:.3f}pctF"
            
            # Add synthetic/fake indicator if detected
            synthetic_indicator = self._get_synthetic_data_indicator()
            if synthetic_indicator:
                data_signature += synthetic_indicator
            
            # Add date at the end
            date_str = datetime.utcnow().strftime('%Y%m%d')
            data_signature += f"_{date_str}"
            
            rankings_filename = f"flw_fraud_{filename_suffix}_{data_signature}.csv"
            print(f"DEBUG: Using data-based filename: {rankings_filename}")

        # Save fraud rankings to ANALYSIS directory
        rankings_path = os.path.join(analysis_dir, rankings_filename)
        print(f"DEBUG: Final rankings CSV path: {rankings_path}")
        print("DEBUG: Writing fraud rankings CSV...")
        results_df.to_csv(rankings_path, index=False)
        print("DEBUG: Fraud rankings CSV written.")
        output_files.append(rankings_path)

        # Generate HTML dashboard
        if HTML_REPORTER_AVAILABLE:
            try:
                print("DEBUG: Generating HTML dashboard...")
                reporter = FraudHTMLReporter(results_df, baseline_data=baseline_data)

                # Create dashboard filename
                if tag:
                    dashboard_filename = f"fraud_dashboard_{tag}.html"
                else:
                    dashboard_filename = f"fraud_dashboard_{filename_suffix}.html"

                dashboard_path = reporter.generate_dashboard(
                    output_path=os.path.join(analysis_dir, dashboard_filename),
                    top_n=50,
                    title=f"FLW Fraud Detection Dashboard - {filename_suffix.title()}"
                )
                output_files.append(dashboard_path)
                self.log(f"Interactive fraud dashboard saved to {dashboard_filename}")
                print(f"DEBUG: HTML dashboard saved to: {dashboard_path}")

                # Generate investigation pages for top 25 FLWs
                try:
                    print("DEBUG: Generating investigation pages...")
                    investigation_files = reporter.generate_investigation_pages(analysis_dir, top_n=25)
                    output_files.extend(investigation_files)
                    self.log(f"Generated {len(investigation_files)} investigation pages")
                    print(f"DEBUG: Investigation pages complete: {len(investigation_files)} files")
                
                except Exception as e:
                    self.log(f"Warning: Could not generate investigation pages - {e}")
                    print(f"DEBUG: Investigation page generation failed: {e}")
        
                # Generate field analysis pages  
                try:

                    print("DEBUG: Generating field analysis pages...")
                    from .field_analysis_generator import FieldAnalysisPageGenerator, add_field_analysis_to_dashboard

                    # 1) generate the pages
                    gen = FieldAnalysisPageGenerator(results_df, baseline_data)
                    field_analysis_files = gen.generate_all_field_pages(analysis_dir)

                    # 2) (optional) patch your main dashboard HTML to link to the index
                    #    pass a *path string* or Path � NOT a DataFrame
                    dashboard_path = os.path.join(analysis_dir, "fraud_dashboard.html")  # adjust if your dashboard file has a different name
                    patched = add_field_analysis_to_dashboard(dashboard_path)  # returns a list of files changed; okay if empty

                    # 3) collect outputs
                    output_files.extend(field_analysis_files)
                    output_files.extend(patched)

                    self.log(f"Generated {len(field_analysis_files)} field analysis pages")
                    print(f"DEBUG: Field analysis pages complete: {len(field_analysis_files)} files")

                except Exception as e:
                    self.log(f"Warning: Could not generate field analysis pages - {e}")
                    print(f"DEBUG: Field analysis page generation failed: {e}")

            except Exception as e:
                self.log(f"Warning: Could not generate HTML dashboard - {e}")
                print(f"DEBUG: HTML dashboard generation failed: {e}")
        else:
            print("DEBUG: HTML dashboard skipped - reporter not available")
        
        # Add experimental analysis to ANALYSIS directory
        if self.get_parameter_value("experimental_analysis", False):
            if not DESCRIPTIVE_ANALYZER_AVAILABLE:
                self.log("WARNING: Experimental analysis requested but descriptive_analyzer not available")
                print("DEBUG: Experimental analysis skipped - descriptive_analyzer not available")
            else:
                print("DEBUG: Running experimental aggregation analysis...")
                try:
                    # Use the results_df we just calculated (has fraud scores)
                    analysis_results = run_standard_analysis(
                        df=self.df, 
                        existing_scores_df=results_df,
                        output_dir=analysis_dir  # Changed to analysis_dir
                    )
                    excel_path = os.path.join(analysis_dir, "aggregation_analysis.xlsx")
                    output_files.append(excel_path)
                    self.log("Experimental analysis saved to aggregation_analysis.xlsx")
                except Exception as e:
                    self.log(f"Error in experimental analysis: {e}")
                    print(f"DEBUG: Experimental analysis failed: {e}")
        
        self.log(f"FLW fraud {filename_suffix} saved to {rankings_path}")
        print(f"DEBUG: generate() complete. Files: {output_files}")
        return output_files
