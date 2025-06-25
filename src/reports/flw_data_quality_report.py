"""
FLW Data Quality Assessment Report - Main Orchestrator
Coordinates all analysis modules

Save this file as: src/reports/flw_data_quality_report.py
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime
from .base_report import BaseReport
from .utils.excel_exporter import ExcelExporter

# Import analysis modules
from .flw_assessment_framework import AssessmentEngine
from .flw_age_analyzer import FLWAgeAnalyzer
from .flw_muac_analyzer import FLWMUACAnalyzer


class FLWDataQualityReport(BaseReport):
    """FLW Data Quality Assessment Report for identifying concerning patterns and data quality issues"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for FLW Data Quality Assessment report"""
        
        # Visit threshold
        ttk.Label(parent_frame, text="Visit threshold (min visits):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        visit_threshold_var = tk.StringVar(value="300")
        ttk.Entry(parent_frame, textvariable=visit_threshold_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Data quality checks selection
        ttk.Label(parent_frame, text="Data quality checks:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Checkboxes for each assessment type
        assessments_frame = ttk.Frame(parent_frame)
        assessments_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        gender_var = tk.BooleanVar(value=True)
        muac_var = tk.BooleanVar(value=True)
        age_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(assessments_frame, text="Gender Balance", variable=gender_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(assessments_frame, text="MUAC Data Quality", variable=muac_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(assessments_frame, text="Age Distribution", variable=age_var).grid(row=0, column=2, sticky=tk.W)
        
        # Distribution analysis options
        ttk.Label(parent_frame, text="Distribution analysis:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        distribution_frame = ttk.Frame(parent_frame)
        distribution_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        flw_ages_var = tk.BooleanVar(value=True)
        opps_ages_var = tk.BooleanVar(value=True)
        flw_muac_var = tk.BooleanVar(value=True)
        opps_muac_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(distribution_frame, text="FLW Ages", variable=flw_ages_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="Opps Ages", variable=opps_ages_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="FLW MUAC", variable=flw_muac_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(distribution_frame, text="Opps MUAC", variable=opps_muac_var).grid(row=1, column=1, sticky=tk.W)
        
        # Store variables for access
        parent_frame.visit_threshold_var = visit_threshold_var
        parent_frame.gender_var = gender_var
        parent_frame.muac_var = muac_var
        parent_frame.age_var = age_var
        parent_frame.flw_ages_var = flw_ages_var
        parent_frame.opps_ages_var = opps_ages_var
        parent_frame.flw_muac_var = flw_muac_var
        parent_frame.opps_muac_var = opps_muac_var
        
    def generate(self):
        """Generate FLW data quality assessment reports"""
        output_files = []
        excel_data = {}
        
        # Get parameters
        visit_threshold = int(self.get_parameter_value('visit_threshold', '300'))
        
        # Get selected assessment types
        selected_assessments = []
        if self.get_parameter_value('gender', True):
            selected_assessments.append('gender_ratio')
        if self.get_parameter_value('muac', True):
            selected_assessments.append('low_red_muac')
        if self.get_parameter_value('age', True):
            selected_assessments.append('low_young_child')
        
        # Get distribution analysis options
        include_flw_ages = self.get_parameter_value('flw_ages', True)
        include_opps_ages = self.get_parameter_value('opps_ages', True)
        include_flw_muac = self.get_parameter_value('flw_muac', True)
        include_opps_muac = self.get_parameter_value('opps_muac', True)
        
        if not selected_assessments and not any([include_flw_ages, include_opps_ages, include_flw_muac, include_opps_muac]):
            self.log("Error: No data quality checks or distribution analysis selected")
            return output_files
        
        self.log(f"Starting FLW data quality assessment with threshold: {visit_threshold} visits")
        if selected_assessments:
            self.log(f"Selected quality checks: {', '.join(selected_assessments)}")
        
        distribution_types = []
        if include_flw_ages: distribution_types.append("FLW Ages")
        if include_opps_ages: distribution_types.append("Opps Ages") 
        if include_flw_muac: distribution_types.append("FLW MUAC")
        if include_opps_muac: distribution_types.append("Opps MUAC")
        
        if distribution_types:
            self.log(f"Distribution analysis: {', '.join(distribution_types)}")
        
        try:
            # Run main quality assessments if any selected
            if selected_assessments:
                # Create assessment engine
                engine = AssessmentEngine(visit_threshold=visit_threshold)
                
                # Filter to selected assessments only
                engine.assessments = {k: v for k, v in engine.assessments.items() if k in selected_assessments}
                
                self.log("Running data quality assessment pipeline...")
                assessment_results, population_stats = engine.run_assessments(self.df)
                
                if len(assessment_results) > 0:
                    # Create opportunity summary
                    self.log("Creating opportunity summary...")
                    opportunity_summary = engine.create_opportunity_summary(assessment_results, self.df)
                    
                    # Create assessment summary statistics
                    self.log("Creating assessment summary...")
                    assessment_summary = self._create_assessment_summary(assessment_results, population_stats, visit_threshold)
                    
                    # Add to Excel data
                    excel_data['FLW Results'] = assessment_results
                    excel_data['Opportunity Summary'] = opportunity_summary
                    excel_data['Assessment Summary'] = assessment_summary
                    
                    # Create flagged FLWs summary (only strong negatives)
                    flagged_flws = assessment_results[assessment_results['has_any_strong_negative']]
                    if len(flagged_flws) > 0:
                        excel_data['Quality Issues Found'] = flagged_flws
                        self.log(f"Found {len(flagged_flws)} FLWs with data quality issues")
                    
                    # Create insufficient data summary
                    insufficient_data_flws = assessment_results[assessment_results['has_insufficient_data']]
                    if len(insufficient_data_flws) > 0:
                        excel_data['Insufficient Data'] = insufficient_data_flws
                        self.log(f"Found {len(insufficient_data_flws)} FLWs with insufficient data for assessment")
                else:
                    self.log("No FLW/opportunity pairs met the minimum visit threshold")
            
            # Generate age distribution analysis
            if include_flw_ages or include_opps_ages:
                self.log("Creating age distribution analysis...")
                age_analyzer = FLWAgeAnalyzer(self.df, self.log)
                
                if include_flw_ages:
                    flw_ages_df = age_analyzer.create_flw_ages_analysis()
                    if flw_ages_df is not None and len(flw_ages_df) > 0:
                        excel_data['FLW Ages'] = flw_ages_df
                
                if include_opps_ages:
                    opps_ages_df = age_analyzer.create_opps_ages_analysis()
                    if opps_ages_df is not None and len(opps_ages_df) > 0:
                        excel_data['Opps Ages'] = opps_ages_df
            
            # Generate MUAC distribution analysis
            if include_flw_muac or include_opps_muac:
                self.log("Creating MUAC distribution analysis...")
                muac_analyzer = FLWMUACAnalyzer(self.df, self.log)
                
                if include_flw_muac:
                    flw_muac_df = muac_analyzer.create_flw_muac_analysis()
                    if flw_muac_df is not None and len(flw_muac_df) > 0:
                        excel_data['FLW MUAC'] = flw_muac_df
                
                if include_opps_muac:
                    opps_muac_df = muac_analyzer.create_opps_muac_analysis()
                    if opps_muac_df is not None and len(opps_muac_df) > 0:
                        excel_data['Opps MUAC'] = opps_muac_df
            
            # Export to Excel
            if excel_data:
                self.log("Creating Excel file with all results...")
                excel_exporter = ExcelExporter(self.log)
                excel_file = excel_exporter.export_to_excel(
                    excel_data,
                    os.path.join(self.output_dir, f"flw_data_quality_assessment_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx")
                )
                if excel_file:
                    output_files.append(excel_file)
                    self.log(f"Created Excel file: {os.path.basename(excel_file)}")
            
            # Log summary statistics if assessments were run
            if selected_assessments and 'assessment_results' in locals() and len(assessment_results) > 0:
                self._log_summary_stats(assessment_results, opportunity_summary if 'opportunity_summary' in locals() else pd.DataFrame())
            
        except Exception as e:
            self.log(f"Error during data quality assessment: {str(e)}")
            raise
        
        return output_files
    
    def _create_assessment_summary(self, assessment_results, population_stats, visit_threshold):
        """Create summary statistics for the assessment run"""
        
        if len(assessment_results) == 0:
            return pd.DataFrame()
        
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Metric': 'Visit Threshold Used',
            'Value': visit_threshold,
            'Description': 'Minimum visits required for assessment'
        })
        
        summary_data.append({
            'Metric': 'Total FLW/Opportunity Pairs Assessed',
            'Value': len(assessment_results),
            'Description': 'Number of FLW/opportunity combinations evaluated'
        })
        
        summary_data.append({
            'Metric': 'Pairs with Data Quality Issues',
            'Value': assessment_results['has_any_strong_negative'].sum(),
            'Description': 'Number of pairs flagged with concerning patterns'
        })
        
        summary_data.append({
            'Metric': 'Pairs with Insufficient Data',
            'Value': assessment_results['has_insufficient_data'].sum(),
            'Description': 'Number of pairs with insufficient data for some assessments'
        })
        
        summary_data.append({
            'Metric': 'Percentage with Issues',
            'Value': f"{(assessment_results['has_any_strong_negative'].sum() / len(assessment_results) * 100):.1f}%",
            'Description': 'Percentage of assessed pairs with quality flags'
        })
        
        summary_data.append({
            'Metric': 'Percentage with Insufficient Data',
            'Value': f"{(assessment_results['has_insufficient_data'].sum() / len(assessment_results) * 100):.1f}%",
            'Description': 'Percentage of assessed pairs with insufficient data'
        })
        
        # Assessment-specific statistics
        assessment_cols = [col for col in assessment_results.columns if col.endswith('_result')]
        
        for col in assessment_cols:
            assessment_name = col.replace('_result', '')
            strong_negative_count = (assessment_results[col] == 'strong_negative').sum()
            insufficient_data_count = (assessment_results[col] == 'insufficient_data').sum()
            total_with_data = assessment_results[col].notna().sum()
            
            if total_with_data > 0:
                summary_data.append({
                    'Metric': f'{assessment_name} - Issues Found',
                    'Value': strong_negative_count,
                    'Description': f'FLW/opp pairs flagged for {assessment_name} quality issues'
                })
                
                summary_data.append({
                    'Metric': f'{assessment_name} - Insufficient Data',
                    'Value': insufficient_data_count,
                    'Description': f'FLW/opp pairs with insufficient {assessment_name} data'
                })
                
                summary_data.append({
                    'Metric': f'{assessment_name} - Issue Rate',
                    'Value': f"{(strong_negative_count / (total_with_data - insufficient_data_count) * 100):.1f}%" if (total_with_data - insufficient_data_count) > 0 else "0.0%",
                    'Description': f'Percentage with {assessment_name} quality issues (excluding insufficient data)'
                })
        
        # Population statistics if available
        if population_stats and 'gender_ratio' in population_stats:
            gender_stats = population_stats['gender_ratio']
            summary_data.append({
                'Metric': 'Population Female Ratio',
                'Value': f"{gender_stats['female_ratio']:.3f}",
                'Description': 'Overall population female child ratio'
            })
            
            summary_data.append({
                'Metric': 'Population CI Range',
                'Value': f"{gender_stats['ci_lower']:.3f} - {gender_stats['ci_upper']:.3f}",
                'Description': '99% confidence interval for population gender ratio'
            })
        
        return pd.DataFrame(summary_data)
    
    def _log_summary_stats(self, assessment_results, opportunity_summary):
        """Log summary statistics to the GUI"""
        
        if len(assessment_results) == 0:
            return
        
        self.log("=== Data Quality Assessment Summary ===")
        self.log(f"Total FLW/opportunity pairs assessed: {len(assessment_results)}")
        
        flagged_count = assessment_results['has_any_strong_negative'].sum()
        flagged_pct = (flagged_count / len(assessment_results) * 100)
        self.log(f"Pairs with quality issues: {flagged_count} ({flagged_pct:.1f}%)")
        
        insufficient_count = assessment_results['has_insufficient_data'].sum()
        insufficient_pct = (insufficient_count / len(assessment_results) * 100)
        self.log(f"Pairs with insufficient data: {insufficient_count} ({insufficient_pct:.1f}%)")
        
        # Assessment-specific counts
        assessment_cols = [col for col in assessment_results.columns if col.endswith('_result')]
        for col in assessment_cols:
            assessment_name = col.replace('_result', '').replace('_', ' ').title()
            issues = (assessment_results[col] == 'strong_negative').sum()
            insufficient = (assessment_results[col] == 'insufficient_data').sum()
            if issues > 0 or insufficient > 0:
                self.log(f"  - {assessment_name}: {issues} issues, {insufficient} insufficient data")
        
        if len(opportunity_summary) > 0:
            self.log(f"Opportunities analyzed: {len(opportunity_summary)}")
            worst_opp = opportunity_summary.iloc[0] if len(opportunity_summary) > 0 else None
            if worst_opp is not None and worst_opp['pct_flws_with_strong_negative'] > 0:
                self.log(f"Highest issue rate: {worst_opp['opportunity_name']} ({worst_opp['pct_flws_with_strong_negative']:.1f}%)")
