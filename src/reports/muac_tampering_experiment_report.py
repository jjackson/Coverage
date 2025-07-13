"""
MUAC Tampering Experiment Report - Demonstrates difficulty of faking authentic MUAC data
Standalone report class that integrates with your existing framework

Save this file as: src/reports/muac_tampering_experiment_report.py
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime
from .base_report import BaseReport
from .utils.excel_exporter import ExcelExporter
from .flw_muac_analyzer_enhanced import EnhancedFLWMUACAnalyzer

class MUACTamperingExperimentReport(BaseReport):
    """MUAC Tampering Experiment Report - Shows how data manipulation breaks authentic patterns"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for MUAC Tampering Experiment report"""
        
        # Tampering parameters
        ttk.Label(parent_frame, text="Tamper percentage (% of data to modify):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tamper_pct_var = tk.StringVar(value="20")
        ttk.Entry(parent_frame, textvariable=tamper_pct_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(parent_frame, text="Noise range (±cm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        noise_range_var = tk.StringVar(value="2")
        ttk.Entry(parent_frame, textvariable=noise_range_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(parent_frame, text="Random seed (for reproducibility):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        seed_var = tk.StringVar(value="42")
        ttk.Entry(parent_frame, textvariable=seed_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Experiment types
        ttk.Label(parent_frame, text="Experiment types:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        experiments_frame = ttk.Frame(parent_frame)
        experiments_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        random_noise_var = tk.BooleanVar(value=True)
        digit_preference_var = tk.BooleanVar(value=True)
        excessive_rounding_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(experiments_frame, text="Random Noise", variable=random_noise_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(experiments_frame, text="Digit Preference", variable=digit_preference_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(experiments_frame, text="Excessive Rounding", variable=excessive_rounding_var).grid(row=0, column=2, sticky=tk.W)
        
        # Analysis scope
        ttk.Label(parent_frame, text="Analysis scope:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        
        scope_frame = ttk.Frame(parent_frame)
        scope_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        analyze_flw_var = tk.BooleanVar(value=True)
        analyze_opps_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(scope_frame, text="FLW-level analysis", variable=analyze_flw_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(scope_frame, text="Opportunity-level analysis", variable=analyze_opps_var).grid(row=0, column=1, sticky=tk.W)
        
        # Output options
        ttk.Label(parent_frame, text="Output options:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        
        output_frame = ttk.Frame(parent_frame)
        output_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        export_csv_var = tk.BooleanVar(value=True)
        create_viz_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(output_frame, text="Export detailed CSV", variable=export_csv_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(output_frame, text="Create visualizations", variable=create_viz_var).grid(row=0, column=1, sticky=tk.W)
        
        # Store variables for access
        parent_frame.tamper_pct_var = tamper_pct_var
        parent_frame.noise_range_var = noise_range_var
        parent_frame.seed_var = seed_var
        parent_frame.random_noise_var = random_noise_var
        parent_frame.digit_preference_var = digit_preference_var
        parent_frame.excessive_rounding_var = excessive_rounding_var
        parent_frame.analyze_flw_var = analyze_flw_var
        parent_frame.analyze_opps_var = analyze_opps_var
        parent_frame.export_csv_var = export_csv_var
        parent_frame.create_viz_var = create_viz_var
    
    def generate(self):
        """Generate MUAC tampering experiment reports"""
        output_files = []
        excel_data = {}
        
        # Get parameters
        tamper_pct = float(self.get_parameter_value('tamper_pct', '20'))
        noise_range = float(self.get_parameter_value('noise_range', '2'))
        seed = int(self.get_parameter_value('seed', '42'))