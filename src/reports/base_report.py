"""
Base Report Class

Base class for all report generators in the Coverage Analysis tool.
"""

import tkinter as tk
from tkinter import ttk


class BaseReport:
    """Base class for all reports"""
    
    def __init__(self, df, output_dir, log_callback, params_frame=None):
        """
        Initialize the report
        
        Args:
            df: pandas DataFrame with input data
            output_dir: Directory to save output files
            log_callback: Function to call for logging messages
            params_frame: GUI frame containing parameter widgets (optional)
        """
        self.df = df
        self.output_dir = output_dir
        self.log = log_callback
        self.params_frame = params_frame
        
    def generate(self):
        """
        Generate the report and return list of created files
        
        Returns:
            list: Paths to created output files
        """
        raise NotImplementedError("Subclasses must implement the generate() method")
        
    @staticmethod
    def setup_parameters(parent_frame):
        """
        Set up GUI parameters for this report type
        
        Args:
            parent_frame: tkinter Frame to add parameter widgets to
        """
        ttk.Label(parent_frame, text="No additional parameters required").grid(row=0, column=0)
        
    def get_parameter_value(self, param_name, default_value=None):
        """
        Helper method to get parameter values from the GUI
        
        Args:
            param_name: Name of the parameter variable (should be stored as param_name_var)
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        if self.params_frame and hasattr(self.params_frame, f"{param_name}_var"):
            try:
                var = getattr(self.params_frame, f"{param_name}_var")
                return var.get()
            except:
                pass
        return default_value
        
    def auto_detect_column(self, patterns, required=True):
        """
        Auto-detect a column based on common naming patterns
        
        Args:
            patterns: List of strings to search for in column names (case insensitive)
            required: If True, raises error if no column found
            
        Returns:
            Column name or None if not found and not required
        """
        for pattern in patterns:
            matching_cols = [col for col in self.df.columns if pattern.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
                
        if required:
            raise ValueError(f"Could not find column matching patterns: {patterns}")
        return None
        
    def save_csv(self, df, filename_prefix, include_timestamp=True):
        """
        Helper method to save a DataFrame as CSV with consistent naming
        
        Args:
            df: DataFrame to save
            filename_prefix: Prefix for the filename
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            Full path to saved file
        """
        import os
        from datetime import datetime
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
        else:
            filename = f"{filename_prefix}.csv"
            
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath