# Coverage Analysis Tools

This project contains tools for analyzing delivery coverage data and generating visualizations and statistics.

## Project Structure

The project is organized into the following directories:

- **src/** - Contains all the source code:
  - `coverage_master.py` - The main command that coordinates the workflow
  - `create_delivery_map.py` - Creates an interactive map of delivery units
  - `create_statistics.py` - Generates statistical analysis and visualizations
  - `create_flw_views.py` - Generates Field-Level Worker performance analysis
  - `models.py` - Data models for coverage analysis
  - `utils.py` - General utility functions
  - `utils_data_loader.py` - Data loading utility functions
- **tests/** - Contains unit tests for the application
- **data/** - Directory for input data files

## Requirements

This requires numerous Python packages, You can install them with:

```
pip install -r requirements.txt
```

## Usage

### Setup a .env file

You will need to create a .env file that includes:  
COMMCARE_API_KEY
COMMCARE_USERNAME
USE_API=True or False
OPPORTUNITY_DOMAIN_MAPPING={"ZEGCAWIS | CHC Givewell Scale Up": "ccc-chc-zegcawis-2024-25", "COWACDI | CHC Givewell Scale Up": "ccc-chc-cowacdi-2024-25", "Name of Opp in Connect": "Name of project space in HQ"}

### Using the Entry Point Script

The simplest way to use these tools is through the main entry point script:

```
python run_coverage.py
```

This will:
1. Load environment variables from .env file (including API credentials and opportunity-domain mappings)
2. Check if USE_API is set to True in environment variables:
   - **API Mode**: Load service delivery data from CSV, then fetch delivery unit data from CommCare API for each opportunity found in the CSV
   - **Local File Mode**: Prompt you to select input Excel and CSV files (if there is only one excel and/or one csv, it will use those without asking)
3. Create coverage data objects for each project/opportunity (Local File Mode will only let you select on xls and csv, API mode must be used for multiple oppurtunities)
4. Create a timestamped output directory (unless custom directory specified)
5. For each project, generate outputs in separate subdirectories:
   - Generate the delivery map
   - Generate the statistics report  
   - Generate the FLW analysis report
6. Create a main index HTML dashboard page linking to all project reports
7. Automatically open the dashboard in your default web browser

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

When in API mode, if the CSV contains multiple oppurtunities, Delivery Unit data for each oppurtunityw will be retrieved.