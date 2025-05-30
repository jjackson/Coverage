#!/usr/bin/env python3
"""
Report Generator Entry Point

Simple GUI tool for generating CSV reports from input data files.
"""

import sys
import os

# Add the src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Add the SUV directory to the Python path for SUV reports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'suv'))

from src.report_generator_gui import main

if __name__ == "__main__":
    main()

