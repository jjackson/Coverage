"""
FLW Analysis Report - Main orchestrator
Coordinates various analyzers and handles Excel export
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime
from .base_report import BaseReport
from .analyzers.basic_analyzer import BasicAnalyzer
from .analyzers.work_window_analyzer import WorkWindowAnalyzer
from .analyzers.timeline_analyzer import TimelineAnalyzer
from .analyzers.opportunity_analyzer import OpportunityAnalyzer
from .analyzers.statistics_analyzer import StatisticsAnalyzer
from .analyzers.location_analyzer import LocationAnalyzer  
from .analyzers.time_analyzer import TimeAnalyzer  
from .utils.data_cleaner import DataCleaner
from .utils.excel_exporter import ExcelExporter


class FLWAnalysisReport(BaseReport):
    """Comprehensive FLW analysis report combining basic metrics and work window analysis"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for FLW Analysis report"""
        
        # Minimum visits filter
        ttk.Label(parent_frame, text="Minimum visits to include:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        min_visits_var = tk.StringVar(value="1")
        ttk.Entry(parent_frame, textvariable=min_visits_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Maximum days to include in timeline
        ttk.Label(parent_frame, text="Max days in timeline (0=all):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        max_timeline_days_var = tk.StringVar(value="30")
        ttk.Entry(parent_frame, textvariable=max_timeline_days_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Store variables for access
        parent_frame.min_visits_var = min_visits_var
        parent_frame.max_timeline_days_var = max_timeline_days_var
        
    def generate(self):
        """Generate comprehensive FLW analysis reports"""
        output_files = []
        excel_data = {}  # Store data for Excel export
        
        # Get parameters
        min_visits = int(self.get_parameter_value('min_visits', '1'))
        max_timeline_days = int(self.get_parameter_value('max_timeline_days', '30'))
        
        # Internal parameters (can be changed later if needed)
        include_work_windows = True
        include_daily_timeline = True
        generate_opp_summary = True
        separate_files = False  # Keep internal for future flexibility
        
        # Auto-detect required columns
        flw_id_patterns = ['flw_id', 'flw id', 'worker_id', 'worker id', 'field_worker_id']
        flw_id_col = self.auto_detect_column(flw_id_patterns, required=True)
        self.log(f"Using '{flw_id_col}' as FLW ID column")
        
        # Clean and prepare data
        self.log("Cleaning and preparing visit data...")
        data_cleaner = DataCleaner(self.df, self.log)
        cleaned_df = data_cleaner.clean_visits_data()
        
        # Split FLWs by opportunity (creates flw_123_1, flw_123_2 etc.)
        self.log("Splitting FLWs by opportunity...")
        cleaned_df = data_cleaner.split_flws_by_opportunity(cleaned_df, flw_id_col)
        
        # Generate basic analysis
        self.log("Generating basic FLW analysis...")
        basic_analyzer = BasicAnalyzer(cleaned_df, flw_id_col, self.log, self.auto_detect_column)
        basic_results = basic_analyzer.analyze(min_visits)

        # Generate work window analysis
        self.log("Generating work window analysis...")
        work_window_analyzer = WorkWindowAnalyzer(cleaned_df, flw_id_col, self.log)
        work_window_results = work_window_analyzer.analyze()

        # Generate time analysis
        self.log("Generating time analysis...")
        time_analyzer = TimeAnalyzer(cleaned_df, flw_id_col, self.log)
        time_results = time_analyzer.analyze()

        # Create combined FLW analysis
        self.log("Combining basic, work window, and time analysis...")
        flw_analysis = basic_results.copy()

        # Merge work window results
        if work_window_results is not None and len(work_window_results) > 0:
            flw_analysis = flw_analysis.merge(work_window_results, on=flw_id_col, how='outer')

        # Merge time results
        if time_results is not None and len(time_results) > 0:
            flw_analysis = flw_analysis.merge(time_results, on=flw_id_col, how='outer')
        else:
            self.log("Time analysis skipped - no timing data available")
        
        # Sort by last visit date (most recent first) and reorder columns logically
        if len(flw_analysis) > 0:
            # Define logical column order
            column_order = [
                # 1. Identity & Basic Info
                flw_id_col,
                'opportunity_name',
                'flw_name',
                
                # 2. Visit Volume & Dates
                'total_visits',
                'total_days_with_visits',
                'first_visit_date',
                'last_visit_date',
                'median_visits_per_day',
                
                # 3. Case Metrics
                'number_cases',
                'percent_cases_with_2plus_visits',
                'percent_active_days_with_2plus_cases',
                
                # 4. Location Metrics
                'median_distance_traveled_per_multi_visit_day',
                'avg_bounding_box_area_multi_visit_cases',
                
                # 5. Quality/Approval
                'pct_approved',
                
                # 6. Work Window & Efficiency
                'avg_visits_per_day',
                'avg_start_time',
                'avg_end_time',
                'avg_work_window_minutes',
                'visits_per_hour',
                'avg_minutes_per_visit',
                
                # 7. Visit Duration & Timing
                'percent_visits_with_timing_issues',
                'avg_visit_duration_minutes',
                'median_visit_duration_minutes',
                'min_visit_duration_minutes',
                'max_visit_duration_minutes',
                'avg_minutes_between_consecutive_visits'
            ]
            
            # Reorder columns - only include columns that actually exist
            existing_columns = list(flw_analysis.columns)
            ordered_columns = [col for col in column_order if col in existing_columns]
            
            # Add any remaining columns that weren't in our predefined order
            remaining_columns = [col for col in existing_columns if col not in ordered_columns]
            final_column_order = ordered_columns + remaining_columns
            
            flw_analysis = flw_analysis[final_column_order]
            
            # Sort by last visit date (most recent first)
            if 'last_visit_date' in flw_analysis.columns:
                flw_analysis = flw_analysis.sort_values('last_visit_date', ascending=False)
        
        excel_data['FLW Analysis'] = flw_analysis
        
        # Generate opportunity summary
        self.log("Generating opportunity summary...")
        opp_analyzer = OpportunityAnalyzer(cleaned_df, flw_id_col, self.log, self.auto_detect_column)
        opp_summary = opp_analyzer.analyze()
        if opp_summary is not None:
            excel_data['Opportunity Summary'] = opp_summary
        
        # Generate daily timeline analysis
        self.log("Generating daily timeline analysis...")
        timeline_analyzer = TimelineAnalyzer(cleaned_df, flw_id_col, self.log, self.auto_detect_column)
        timeline_results = timeline_analyzer.analyze(max_timeline_days)
        
        for name, data in timeline_results.items():
            excel_data[name] = data
        
        # Generate statistics
        self.log("Generating summary statistics...")
        stats_analyzer = StatisticsAnalyzer(flw_analysis, self.log)
        stats_results = stats_analyzer.analyze()
        if stats_results is not None:
            excel_data['Statistics'] = stats_results
        
        # Generate case location analysis (NEW SECTION)
        self.log("Generating case location analysis...")
        location_analyzer = LocationAnalyzer(cleaned_df, flw_id_col, self.log, self.auto_detect_column)
        case_locations = location_analyzer.analyze_case_locations()
        if case_locations is not None:
            excel_data['Case Locations'] = case_locations
        else:
            self.log("Case location analysis skipped - no GPS data or case_id column found")
        
        # Always save the main FLW analysis CSV
        flw_file = self.save_csv(flw_analysis, "flw_analysis")
        output_files.append(flw_file)
        self.log(f"Created: {os.path.basename(flw_file)}")
        
        # Export to Excel with all tabs
        if excel_data:
            self.log("Creating Excel file with all analysis tabs...")
            excel_exporter = ExcelExporter(self.log)
            excel_file = excel_exporter.export_to_excel(
                excel_data, 
                os.path.join(self.output_dir, f"flw_analysis_complete_{datetime.now().strftime('%Y-%m-%d')}.xlsx")
            )
            if excel_file:
                output_files.append(excel_file)
                self.log(f"Created Excel file: {os.path.basename(excel_file)}")
            else:
                self.log("Excel export failed - CSV files still available")
        
        return output_files
