# Coverage Analysis Tools

This project contains tools for analyzing delivery coverage data and generating visualizations and statistics.

## Project Structure

The project is organized into the following directories :

### Root Level Entry Points

- **run_coverage.py** - Main entry point for generating Coverage HTML reports and visualizations
- **run_report_generator.py** - Launches a GUI interface for generating additional specialized reports

### Source Code Organization

- **src/** - Contains all the source code organized into modules:
  - `coverage_master.py` - The main command that coordinates the coverage analysis workflow
  - `create_delivery_map.py` - Creates an interactive map of delivery units
  - `create_statistics.py` - Generates statistical analysis and visualizations
  - `create_flw_views.py` - Generates Field-Level Worker performance analysis
  - `water_visualization/` - 🆕 Water points mapping from CommCare survey data
  - `opportunity_comparison_statistics.py` - Generates comparison statistics between opportunities
  - `report_generator_gui.py` - GUI interface for report generation
  
  - **models/** - Data models for coverage analysis:
    - `coverage_data.py` - Main coverage data model
    - `delivery_unit.py` - Delivery unit data model
    - `flw.py` - Field-Level Worker data model
    - `service_delivery_point.py` - Service delivery point data model
    - `service_area.py` - Service area data model
  
  - **utils/** - Utility functions and data handling:
    - `data_loader.py` - Data loading utility functions
    - `superset_export.py` - Superset data export utilities
  
  - **reports/** - Specialized report generation modules:
    - `base_report.py` - Base class for report generation
    - `microplan_review.py` - Microplan review report generator
    - `flw_visit_count.py` - FLW visit count analysis report

### Documentation

- **[Coverage Data Guide](COVERAGE_DATA_GUIDE.md)** - Comprehensive guide explaining the data flow into the CoverageData object and data dictionary for all models

### Supporting Directories

- **tests/** - Contains unit tests for the application
- **data/** - Directory for input data files

### Installing Python requirements for the project
Python projects require certain libraries to be installed in order to successfully run the project.
In this step, we will first  install the required libraries to run the project. 
In order to install those requirements, run the following command from project's root directory: 
``` 
python install_requirements.py
```
This command will automatically check the project's library requirements.
It will then download and install the required libraries for the project. 
This will also check if Python version you have installed is >= "3.12.0"
This command will automatically update the requirements.txt to store the versions of those libraries for referencing back in case of any issues faced. 

## Usage

### Setup a .env file

You will need to create a .env file that includes:  
COMMCARE_API_KEY
COMMCARE_USERNAME
USE_API=TRUE or False
OPPORTUNITY_DOMAIN_MAPPING={"ZEGCAWIS | CHC Givewell Scale Up": "ccc-chc-zegcawis-2024-25", "COWACDI | CHC Givewell Scale Up": "ccc-chc-cowacdi-2024-25", "Name of Opp in Connect": "Name of project space in HQ"}
SUPERSET_URL
SUPERSET_USERNAME
SUPERSET_PASSWORD


### Using the Entry Point Script

The simplest way to use these tools is through the main entry point script:

```
python run_coverage.py
```

This will:
1. Load environment variables from .env file (including API credentials and opportunity-domain mappings)
2. Check if USE_API is set to True in environment variables:
   - **API Mode**: Load service delivery data from Superset, then fetch delivery unit data from CommCare API for each opportunity found in the CSV
   - **Local File Mode**: Prompt you to select input Excel and CSV files (if there is only one excel and/or one csv, it will use those without asking)
3. Create coverage data objects for each project/opportunity (Local File Mode will only let you select on xls and csv, API mode must be used for multiple oppurtunities)
4. Create a timestamped output directory (unless custom directory specified)
5. For each project, generate outputs in separate subdirectories:
   - Generate the delivery map
   - Generate the statistics report  
   - Generate the FLW analysis report
6. Create a main index HTML dashboard page linking to all project reports
7. Automatically open the dashboard in your default web browser

## Water Points Mapping

🆕 **New Feature**: Interactive water points visualization from CommCare survey data.

### Quick Start
```bash
cd src/water_visualization
python launch_water_map.py
```

This will:
1. Automatically detect water survey projects in the `water_data/` directory
2. Process Excel files and link them to corresponding image folders
3. Generate an interactive Leaflet map with all water points
4. Copy images to the output directory for offline viewing
5. Automatically open the map in your default browser

### Features
- **Interactive Markers**: Color-coded by water point type (piped water, boreholes, wells)
- **Rich Popups**: Location hierarchy, water point characteristics, survey metadata
- **Image Gallery**: Click photo placeholders to view full-size images in lightbox
- **Responsive Design**: Works on desktop and mobile devices
- **Offline Ready**: Single HTML file with all images included

### Data Structure
Place your CommCare water survey data in the `water_data/` directory:
```
water_data/
├── PROJECT1 CCC Waterbody Survey - August 7.xlsx
├── PROJECT1 Pics-Aug 7/
│   └── photo1-username-form_uuid.jpg
├── PROJECT2 CCC Waterbody Survey - August 7.xlsx
└── PROJECT2 Pics - Aug 7/
    └── photo1-username-form_uuid.jpg
```

The system automatically matches Excel files to image directories by project name.

## Input Files

The tools expect input files:

1. **Excel File** (Delivery Unit Data Exported from CommCareHQ if using local mode) - Contains delivery unit data with the following columns like:
   - du_id
   - service_area or service_area_number
   - buildings
   - delivery_count
   - delivery_target
   - du_status
   - owner_name (used as FLW identifier)
   - WKT (geographic boundaries)
   - surface_area

2. **CSV File** (Service Delivery Data exported from Connect Superset) - Contains service delivery points with columns like:
   - lattitude, longitude (coordinates)
   - flw_id, flw_name (field worker identifiers)
   - service_date or date (when available)

When in API mode, if the Superset query contains multiple oppurtunities, Delivery Unit data for each oppurtunityw will be retrieved.
