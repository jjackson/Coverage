"""
FLW Data Quality Assessment Report - Updated Version
Complete version with all methods included

Save this file as: src/reports/flw_data_quality_report.py
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime
from .base_report import BaseReport
from .utils.excel_exporter import ExcelExporter

# Import the assessment framework
from .flw_assessment_framework import AssessmentEngine


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
        
        # Store variables for access
        parent_frame.visit_threshold_var = visit_threshold_var
        parent_frame.gender_var = gender_var
        parent_frame.muac_var = muac_var
        parent_frame.age_var = age_var
        
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
        
        if not selected_assessments:
            self.log("Error: No data quality checks selected")
            return output_files
        
        self.log(f"Starting FLW data quality assessment with threshold: {visit_threshold} visits")
        self.log(f"Selected quality checks: {', '.join(selected_assessments)}")
        
        # Create assessment engine
        engine = AssessmentEngine(visit_threshold=visit_threshold)
        
        # Filter to selected assessments only
        engine.assessments = {k: v for k, v in engine.assessments.items() if k in selected_assessments}
        
        try:
            # Run assessments
            self.log("Running data quality assessment pipeline...")
            assessment_results, population_stats = engine.run_assessments(self.df)
            
            if len(assessment_results) == 0:
                self.log("No FLW/opportunity pairs met the minimum visit threshold")
                return output_files
            
            # Create opportunity summary
            self.log("Creating opportunity summary...")
            opportunity_summary = engine.create_opportunity_summary(assessment_results, self.df)
            
            # Create assessment summary statistics
            self.log("Creating assessment summary...")
            assessment_summary = self._create_assessment_summary(assessment_results, population_stats, visit_threshold)
            
            # Prepare Excel data with updated tab names
            excel_data['FLW Results'] = assessment_results  # Renamed from 'Data Quality Results'
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
            
            # Save main CSV
            main_csv = self.save_csv(assessment_results, "flw_data_quality_results")
            output_files.append(main_csv)
            self.log(f"Created: {os.path.basename(main_csv)}")
            
            # Save opportunity summary CSV
            if len(opportunity_summary) > 0:
                opp_csv = self.save_csv(opportunity_summary, "opportunity_quality_summary")
                output_files.append(opp_csv)
                self.log(f"Created: {os.path.basename(opp_csv)}")
            
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
            
            # Log summary statistics
            self._log_summary_stats(assessment_results, opportunity_summary)
            
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
