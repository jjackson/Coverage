#!/usr/bin/env python
"""
Script to launch the organization dashboard from the index.html link
"""

import sys
import os
from src.coverage_master import coverage_data_objects
from src.flw_summary_dashboard import create_flw_dashboard

if __name__ == "__main__":
    if coverage_data_objects:
        print(f"\n🚀 Launching organization dashboard for {len(coverage_data_objects)} opportunities...")
        create_flw_dashboard(coverage_data_objects)
    else:
        print("❌ No coverage data found. Please run coverage analysis first.")
