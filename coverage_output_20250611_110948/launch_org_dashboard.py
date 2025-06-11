#!/usr/bin/env python
"""
Script to launch the organization dashboard from the index.html link
"""

import sys
import os
from src.coverage_master import main
from src.flw_summary_dashboard import create_flw_dashboard

if __name__ == "__main__":
    # Get coverage data by calling main()
    coverage_data_objects = main()
    
    if coverage_data_objects:
        print(f"\n🚀 Launching organization dashboard for {len(coverage_data_objects)} opportunities...")
        create_flw_dashboard(coverage_data_objects)
    else:
        print("❌ No valid opportunities found.")
