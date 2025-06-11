#!/usr/bin/env python
"""
Script to launch the organization dashboard from the index.html link
"""

import sys
import os
import pickle
from src.flw_summary_dashboard import create_flw_dashboard

if __name__ == "__main__":
    try:
        # Load coverage data from file
        with open('coverage_data.pkl', 'rb') as f:
            coverage_data_objects = pickle.load(f)
        
        if coverage_data_objects:
            print(f"\n🚀 Launching organization dashboard for {len(coverage_data_objects)} opportunities...")
            create_flw_dashboard(coverage_data_objects)
        else:
            print("❌ No coverage data found. Please run coverage analysis first.")
    except FileNotFoundError:
        print("❌ Coverage data file not found. Please run coverage analysis first.")
    except Exception as e:
        print(f"❌ Error loading coverage data: {str(e)}")
