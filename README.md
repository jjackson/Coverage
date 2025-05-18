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

The following Python packages are required:

```
pandas>=1.3.0
geopandas>=0.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
numpy>=1.20.0
shapely>=1.8.0
```

You can install them with:

```
pip install -r requirements.txt
```

## Usage

### Using the Entry Point Script

The simplest way to use these tools is through the main entry point script:

```
python run_coverage.py
```

This will:
1. Prompt you to select input Excel and CSV files. If there is only one excel and/or one csv, it will use those without asking you to select the file.
2. Create a timestamped output directory
3. Generate the delivery map
4. Generate the statistics report
5. Generate the FLW analysis report
6. Create an index HTML page linking to all reports

#### Command-line options:

```
python run_coverage.py --excel-file [EXCEL_FILE] --csv-file [CSV_FILE] --output-dir [OUTPUT_DIR]
```

- `--excel-file`: Specify the Excel file containing delivery unit data
- `--csv-file`: Specify the CSV file containing service delivery data
- `--output-dir`: Custom output directory name (optional)

### Using Individual Tools

You can also run each tool separately:

#### Delivery Map Generator

```
python src/create_delivery_map.py --excel [EXCEL_FILE] --csv [CSV_FILE]
```

#### Statistics Generator

```
python src/create_statistics.py --excel [EXCEL_FILE] --csv [CSV_FILE]
```

#### FLW Analysis Generator

```
python src/create_flw_views.py --excel [EXCEL_FILE] --csv [CSV_FILE]
```

## Input Files

The tools expect two input files:

1. **Excel File** (Delivery Unit Data Exported from CommCareHQ) - Contains delivery unit data with the following columns:
   - du_id
   - service_area or service_area_number
   - buildings
   - delivery_count
   - delivery_target
   - du_status
   - owner_name (used as FLW identifier)
   - WKT (geographic boundaries)
   - surface_area

2. **CSV File** (Service Delivery Data exported from Connect Superset) - Contains service delivery points with columns:
   - lattitude, longitude (coordinates)
   - flw_id, flw_name (field worker identifiers)
   - service_date or date (when available)

## Output

The tools generate the following outputs in the specified directory:

1. **nigeria_delivery_units_map.html** - Interactive map showing delivery units and service points
2. **coverage_statistics.html** - Statistical report with visualizations
3. **flw_analysis.html** - Field-Level Worker performance analysis
4. **index.html** - Dashboard page linking to all reports

## Development

To run the tests:

```
python -m pytest
``` 