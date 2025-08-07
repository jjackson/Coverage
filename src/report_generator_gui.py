"""
Report Generator GUI

Main GUI application for generating CSV reports from input data.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import threading
import json

# Import available reports
from .reports import AVAILABLE_REPORTS

# Config functions
def get_config_path():
    """Get path to config file in data directory"""
    script_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "config.json"

def save_settings(download_file, report_type):
    """Save the last used settings"""
    config = {
        "last_download_file": download_file,
        "last_report_type": report_type
    }
    try:
        with open(get_config_path(), 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass  # Fail silently

def load_settings():
    """Load the last used settings"""
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("last_download_file", ""), config.get("last_report_type", "")
    except Exception:
        pass  # Fail silently
    
    return "", ""  # Return empty defaults

class ReportGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coverage Analysis - Report Generator")
        self.root.geometry("600x700")  # Increased height for larger output window
        
        # Variables
        self.input_file_path = tk.StringVar()
        self.output_directory = tk.StringVar(value=str(Path.home() / "Downloads"))
        self.selected_report = tk.StringVar()
        
        # Available reports - automatically loaded from reports package
        self.reports = AVAILABLE_REPORTS
        
        self.setup_ui()
        self.load_last_settings()  # Load settings after UI is setup
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Input file selection
        ttk.Label(main_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=1, columnspan=2, sticky="we", pady=5)
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.input_file_path, width=40)
        self.file_entry.grid(row=0, column=0, sticky="we")
        ttk.Button(file_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=1, padx=(5, 0))
        
        file_frame.columnconfigure(0, weight=1)
        
        # Report type selection
        ttk.Label(main_frame, text="Report Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.report_dropdown = ttk.Combobox(main_frame, textvariable=self.selected_report, 
                                          values=list(self.reports.keys()), state="readonly", width=30)
        self.report_dropdown.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.report_dropdown.bind("<<ComboboxSelected>>", self.on_report_selected)
        
        # Parameters frame (dynamic)
        self.params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        self.params_frame.grid(row=2, column=0, columnspan=3, sticky="we", pady=10)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=1, columnspan=2, sticky="we", pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_directory, width=40).grid(row=0, column=0, sticky="we")
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_directory).grid(row=0, column=1, padx=(5, 0))
        
        output_frame.columnconfigure(0, weight=1)
        
        # Generate button
        self.generate_button = ttk.Button(main_frame, text="Generate Report", command=self.generate_report)
        self.generate_button.grid(row=4, column=0, columnspan=3, pady=15)
        
        # Progress area
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=5)
        
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=30, width=70)  # Adjusted height for better fit
        self.progress_text.grid(row=0, column=0, sticky="nsew")
        
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(0, weight=1)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def load_last_settings(self):
        """Load the last used settings and populate the form"""
        last_file, last_type = load_settings()
        
        # Set the file path if it exists
        if last_file and os.path.exists(last_file):
            self.input_file_path.set(last_file)
            
        # Set the report type if it's valid
        if last_type and last_type in self.reports:
            self.selected_report.set(last_type)
            # Trigger the parameter setup for this report type
            self.on_report_selected()
        
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_path.set(filename)
            
    def browse_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory.set(directory)
            
    def on_report_selected(self, event=None):
        # Clear existing parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        # Get selected report class and setup its parameters
        report_name = self.selected_report.get()
        if report_name in self.reports:
            report_class = self.reports[report_name]
            if hasattr(report_class, 'setup_parameters'):
                report_class.setup_parameters(self.params_frame)
                
    def log_message(self, message):
        """Add a message to the progress text area"""
        self.progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()
        
    def generate_report(self):
        # Save current settings first
        save_settings(self.input_file_path.get(), self.selected_report.get())
        
        # Validate inputs
        if not self.input_file_path.get():
            messagebox.showerror("Error", "Please select an input file")
            return
            
        if not self.selected_report.get():
            messagebox.showerror("Error", "Please select a report type")
            return
            
        if not os.path.exists(self.input_file_path.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
            
        # Run report generation in separate thread to prevent GUI freezing
        self.generate_button.config(state="disabled")
        self.progress_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_report_generation)
        thread.daemon = True
        thread.start()
        
    def run_report_generation(self):
        try:
            self.log_message("Starting report generation...")
            
            # Load input data
            self.log_message(f"Loading data from {os.path.basename(self.input_file_path.get())}...")
            df = pd.read_csv(self.input_file_path.get())
            self.log_message(f"Loaded {len(df):,} records")
            
            # Get report class and generate report
            report_class = self.reports[self.selected_report.get()]
            report_instance = report_class(df, self.output_directory.get(), self.log_message, self.params_frame)
            
            # Generate the report
            output_files = report_instance.generate()
            
            self.log_message(f"? Report complete! Generated {len(output_files)} files:")
            for file in output_files:
                self.log_message(f"  - {os.path.basename(file)}")
                
        except Exception as e:
            self.log_message(f"? Error: {str(e)}")
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")
        finally:
            # Re-enable the button
            self.root.after(0, lambda: self.generate_button.config(state="normal"))


def main():
    """Entry point for the GUI application"""
    root = tk.Tk()
    app = ReportGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
