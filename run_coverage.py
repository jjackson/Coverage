#!/usr/bin/env python
"""
Coverage Analysis Master Script

This script runs the coverage analysis pipeline.
It loads delivery unit data and service delivery data, and generates
a map and statistics for coverage analysis.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function
from src.coverage_master import main
if __name__ == "__main__":
    # Run the main function
    main()


