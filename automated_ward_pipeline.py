#!/usr/bin/env python3 
"""
Automated Ward Analysis Pipeline - With Proxy Ward Support

Runs: Superset Query → Ward Extractor/Proxy Generator → Grid3 Analysis
- Date-stamped directories (not time-stamped)
- Cached Superset data (download once per day)
- Direct report instantiation (no GUI dependency)
- Multiple analysis configurations
- Support for proxy wards (opportunity-based boundaries)

Usage:
    python automated_ward_pipeline.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import geopandas as gpd  
import requests
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed - using system environment variables only")

# Add src to path for imports
sys.path.append('src')

from src.utils.data_loader import export_superset_query_with_pagination
from src.reports.WardBoundaryExtractor import WardBoundaryExtractor
from src.reports.OpportunityBoundaryGenerator import OpportunityBoundaryGenerator
from src.reports.Grid3WardAnalysis import Grid3WardAnalysis

# Configuration: Define your analysis runs
ANALYSIS_CONFIGS = [


    {
        "name": "GW8",
        "superset_query_id": 187,
        "grid3_file": "data/grid3/NGA_population_v3_0_gridded.tif",
        "shapefile_dir": "data/shape",
        "min_visits": 500,
        "buffer_distance": 0,
        "expected_visits_per_pop": 0.18,
        "description": "GW8 visits in Nigeria",
	"buildings_dir": "data/buildings/"
    },
    {
        "name": "Solina first pass", 
        "superset_query_id": 205,
        "grid3_file": "data/grid3/NGA_population_v3_0_gridded.tif",
        "shapefile_dir": "data/shape",
        "min_visits": 500,
        "buffer_distance": 0,
        "expected_visits_per_pop": 0.18,
        "description": "Solina visits in Nigeria",
	"buildings_dir": "data/buildings/"
    },
    {
        "name": "Bauchi",
        "superset_query_id": "local_file",
        "local_file": r"C:\Users\Neal Lesh\Coverage\data\bauchi\bauchi_standardized_visits.csv",
        "grid3_file": "data/grid3/NGA_population_v3_0_gridded.tif",
        "shapefile_dir": "data/shape",
        "min_visits": 300,
        "buffer_distance": 0,
        "expected_visits_per_pop": 0.106,
        "description": "Bauchi visits"
    }


]


class AutomatedWardPipeline:
    """Automated pipeline for Ward Boundary Extraction and Grid3 Analysis"""
    
    def __init__(self, base_output_dir=r"C:\Users\Neal Lesh\Coverage\automated_pipeline_output"):
        """Initialize pipeline with base output directory"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create today's directory
        today = datetime.now().strftime("%Y_%m_%d")
        self.today_dir = self.base_output_dir / today
        self.today_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.superset_data_dir = self.today_dir / "superset_data"
        self.superset_data_dir.mkdir(exist_ok=True)
        
        # Load Superset credentials
        self.superset_url = os.getenv('SUPERSET_URL')
        self.superset_username = os.getenv('SUPERSET_USERNAME') 
        self.superset_password = os.getenv('SUPERSET_PASSWORD')
        
        self._validate_credentials()
        
        print(f"📁 Working directory: {self.today_dir}")
        
    def _validate_credentials(self):
        """Validate Superset credentials"""
        missing = []
        if not self.superset_url:
            missing.append('SUPERSET_URL')
        if not self.superset_username:
            missing.append('SUPERSET_USERNAME')
        if not self.superset_password:
            missing.append('SUPERSET_PASSWORD')
            
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
            
        print(f"✓ Superset credentials loaded: {self.superset_url}")

    def run_pipeline(self, analysis_configs):
        """Run the complete pipeline for all configurations"""
        print(f"\n🚀 Starting Automated Ward Pipeline")
        print(f"📊 Processing {len(analysis_configs)} analysis configurations")
        print("=" * 60)
        
        # Phase 1: Download all Superset data
        superset_files = self._download_all_superset_data(analysis_configs)
        
        # Phase 2: Run analyses
        results = []
        for i, config in enumerate(analysis_configs, 1):
            print(f"\n📋 Analysis {i}/{len(analysis_configs)}: {config['name']}")
            print("-" * 40)
            
            try:
                csv_file = superset_files.get(config['name'])
                if not csv_file:
                    raise ValueError(f"No data file found for {config['name']}")
                
                result = self._run_single_analysis(config, csv_file)
                results.append({
                    'config': config,
                    'status': 'success',
                    'output_dir': result['output_dir'],
                    'files': result['files']
                })
                print(f"✅ Success: {len(result['files'])} files generated")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                results.append({
                    'config': config,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self._print_summary(results)
        return results
 
    def _download_all_superset_data(self, configs):
        """Download data from Superset for all configurations (cache-aware)"""
        print("\n📥 Phase 1: Downloading Superset Data")
        print("-" * 40)
        
        superset_files = {}
        
        for config in configs:
            name = config['name']
            query_id = config['superset_query_id']
            
            # Check for local file case
            if query_id in [None, "local_file"] and 'local_file' in config:
                local_file_path = Path(config['local_file'])
                target_file = self.superset_data_dir / f"{name.lower()}_data.csv"
                
                if target_file.exists():
                    print(f"  ✓ {name}: Using cached local file ({target_file.name})")
                else:
                    print(f"  📋 {name}: Copying local file...")
                    import shutil
                    shutil.copy2(local_file_path, target_file)
                    print(f"  ✓ {name}: Copied local file")
                
                superset_files[name] = target_file
                continue
            
            # Regular Superset processing
            csv_file = self.superset_data_dir / f"{name.lower()}_data.csv"
            
            if csv_file.exists():
                print(f"  ✓ {name}: Using cached data ({csv_file.name})")
                superset_files[name] = csv_file
            else:
                print(f"  📥 {name}: Downloading from query {query_id}...")
                try:
                    # Get SQL from saved query
                    sql_query = self._get_sql_from_saved_query(query_id)
                    
                    # Download data
                    downloaded_file = export_superset_query_with_pagination(
                        superset_url=self.superset_url,
                        sql_query=sql_query,
                        username=self.superset_username,
                        password=self.superset_password,
                        output_filename=str(csv_file.with_suffix(''))  # Remove .csv as function adds it
                    )
                    
                    # Verify and get info
                    if os.path.exists(downloaded_file):
                        df = pd.read_csv(downloaded_file)
                        print(f"  ✓ {name}: Downloaded {len(df):,} rows")
                        superset_files[name] = Path(downloaded_file)
                    else:
                        print(f"  ✗ {name}: Download failed")
                        
                except Exception as e:
                    print(f"  ✗ {name}: Error downloading - {str(e)}")
        
        return superset_files
    
    def _get_sql_from_saved_query(self, query_id):
        """Get SQL query from Superset saved query ID"""
        try:
            session = requests.Session()
            
            # Login
            auth_url = f'{self.superset_url}/api/v1/security/login'
            auth_data = {
                'username': self.superset_username,
                'password': self.superset_password,
                'provider': 'db',
                'refresh': True
            }
            
            response = session.post(auth_url, json=auth_data, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Authentication failed: {response.text}")
            
            auth_data = response.json()
            access_token = auth_data.get('access_token')
            if not access_token:
                raise RuntimeError("No access token received")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get CSRF token
            csrf_url = f'{self.superset_url}/api/v1/security/csrf_token/'
            csrf_response = session.get(csrf_url, headers=headers, timeout=30)
            if csrf_response.status_code == 200:
                csrf_data = csrf_response.json()
                csrf_token = csrf_data.get('result')
                if csrf_token:
                    headers['x-csrftoken'] = csrf_token
                    headers['Referer'] = self.superset_url + "/sqllab"
            
            # Get saved query
            saved_query_url = f'{self.superset_url}/api/v1/saved_query/{query_id}'
            response = session.get(saved_query_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get saved query {query_id}: {response.text}")
            
            query_data = response.json()
            result = query_data.get('result', {})
            sql_query = result.get('sql', '')
            
            if not sql_query:
                raise RuntimeError(f"No SQL found in saved query {query_id}")
            
            return sql_query
            
        except Exception as e:
            raise RuntimeError(f"Failed to get SQL from saved query {query_id}: {str(e)}")
    
    def _run_single_analysis(self, config, csv_file):
        """Run Ward Extractor/Proxy Generator → Grid3 Analysis for a single configuration"""
        
        # Create analysis directory
        analysis_dir = self.today_dir / f"{config['name'].lower()}_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        print(f"📂 Analysis directory: {analysis_dir.name}")
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df):,} rows from {csv_file.name}")

        # Step 1: Generate boundaries (real wards, proxy wards, or premade polygons)
        use_proxy = config.get('use_proxy_wards', False)
        use_premade = config.get('use_premade_polygons', False)
        
        if use_premade:
            print("📐 Step 1: Converting premade polygon CSV...")
            boundary_dir = analysis_dir / "premade_boundaries"
            
            polygon_csv = config.get('polygon_csv_file')
            if not polygon_csv or not os.path.exists(polygon_csv):
                raise ValueError(f"Polygon CSV file not found: {polygon_csv}")
            
            actual_boundary_dir = convert_polygon_csv_to_shapefile(
                polygon_csv_path=polygon_csv,
                output_dir=str(boundary_dir),
                data_tag=config['name'].lower()
            )
            
            # For premade polygons, we already have the shapefile - no need to generate
            boundary_files = [str(Path(actual_boundary_dir) / f"affected_wards_{config['name'].lower()}.shp")]
            print(f"  ✓ Boundary conversion complete: {len(boundary_files)} files")
            
        elif use_proxy:
            print("🔷 Step 1: Generating proxy ward boundaries...")
            boundary_dir = analysis_dir / "proxy_wards"
            boundary_dir.mkdir(exist_ok=True)
            
            boundary_generator = OpportunityBoundaryGenerator.create_for_automation(
                df=df,
                output_dir=str(boundary_dir),
                buffer_distance=config.get('proxy_buffer_m', 50),
                max_radius_km=config.get('proxy_max_radius_km', 75),
                min_visits=config.get('min_visits', 500),
                output_format="Shapefile + GeoJSON"
            )
            
            # Add logging
            def log_func(message):
                print(f"    {message}")
            boundary_generator.log = log_func
            
            try:
                boundary_files = boundary_generator.generate()
                print(f"  ✓ Boundary generation complete: {len(boundary_files)} files")
                
                # Find the actual output directory (it creates a timestamped subdirectory)
                actual_boundary_dir = None
                for subdir in boundary_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("proxy_wards_"):
                        actual_boundary_dir = subdir
                        break
                
                if not actual_boundary_dir:
                    raise RuntimeError("Could not find boundary generator output directory")
                    
            except Exception as e:
                raise RuntimeError(f"Boundary generation failed: {str(e)}")
                
        else:
            print("🗺️  Step 1: Extracting ward boundaries...")
            boundary_dir = analysis_dir / "ward_boundaries"
            boundary_dir.mkdir(exist_ok=True)
            
            boundary_generator = WardBoundaryExtractor.create_for_automation(
                df=df,
                output_dir=str(boundary_dir),
                shapefile_dir=config['shapefile_dir'],
                buffer_distance=config.get('buffer_distance', 0),
                min_visits=config.get('min_visits', 100),
                output_format="Shapefile + GeoJSON"
            )
            
            # Add logging
            def log_func(message):
                print(f"    {message}")
            boundary_generator.log = log_func
            
            try:
                boundary_files = boundary_generator.generate()
                print(f"  ✓ Boundary generation complete: {len(boundary_files)} files")
                
                # Find the actual output directory (it creates a timestamped subdirectory)
                actual_boundary_dir = None
                for subdir in boundary_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("ward_extractor_"):
                        actual_boundary_dir = subdir
                        break
                
                if not actual_boundary_dir:
                    raise RuntimeError("Could not find boundary generator output directory")
                    
            except Exception as e:
                raise RuntimeError(f"Boundary generation failed: {str(e)}")

        # Step 2: Grid3 Ward Analysis
        print("Step 2: Running Grid3 ward analysis...")
        grid3_output_dir = analysis_dir  # Output directly to analysis_dir
        
        # Define log_func for Grid3 analyzer
        def log_func(message):
            print(f"    {message}")
        
        grid3_analyzer = Grid3WardAnalysis.create_for_automation(
             df=df,
             output_dir=str(grid3_output_dir),  
             stage1_folder=str(actual_boundary_dir),
             grid3_file=config['grid3_file'],
             include_partial=True,
             expected_visits_per_pop=config.get('expected_visits_per_pop', 0.18),
             buildings_dir=config.get('buildings_dir', None)
        )

        # Add logging
        grid3_analyzer.log = log_func
        
        try:
            analysis_files = grid3_analyzer.generate()
            print(f"  ✓ Grid3 analysis complete: {len(analysis_files)} files")
        except Exception as e:
            raise RuntimeError(f"Grid3 analysis failed: {str(e)}")
        
        all_files = boundary_files + analysis_files
        
        return {
            'output_dir': str(analysis_dir),
            'files': all_files,
            'boundary_files': boundary_files,
            'analysis_files': analysis_files
        }
    
    def _print_summary(self, results):
        """Print pipeline execution summary"""
        print(f"\n📈 Pipeline Summary")
        print("=" * 60)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            print(f"\n📁 Output directories:")
            for result in successful:
                name = result['config']['name']
                output_dir = Path(result['output_dir']).name
                file_count = len(result['files'])
                
                if result['config'].get('use_premade_polygons', False):
                    ward_type = "premade polygons"
                elif result['config'].get('use_proxy_wards', False):
                    ward_type = "proxy wards"
                else:
                    ward_type = "real wards"
                    
                print(f"  • {name}: {output_dir} ({file_count} files, {ward_type})")
        
        if failed:
            print(f"\n💥 Failed analyses:")
            for result in failed:
                name = result['config']['name']
                error = result['error']
                print(f"  • {name}: {error}")


def convert_polygon_csv_to_shapefile(polygon_csv_path, output_dir, data_tag):
    """
    Convert polygon CSV to shapefile format compatible with Grid3WardAnalysis
    
    Expected CSV columns: opp_id, cluster_id, polygon_wkt
    Creates: ward_id, ward_name, state_name, geometry
    """
    print(f"  📐 Converting polygon CSV to shapefile...")
    
    # Read CSV
    df = pd.read_csv(polygon_csv_path)
    
    # Parse WKT geometries
    from shapely import wkt
    df['geometry'] = df['polygon_wkt'].apply(wkt.loads)
    
    # Create ward columns
    df['ward_id'] = df['opp_id'].astype(str) + "_" + df['cluster_id'].astype(str)
    df['ward_name'] = df['ward_id']
    df['state_name'] = "state_proxy"
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    # Keep only necessary columns for Grid3 analysis
    essential_cols = ['ward_id', 'ward_name', 'state_name', 'geometry']
    gdf = gdf[essential_cols]
    
    # Save as shapefile with expected naming
    boundary_dir = Path(output_dir)
    boundary_dir.mkdir(exist_ok=True)
    
    shapefile_path = boundary_dir / f"affected_wards_{data_tag}.shp"
    gdf.to_file(shapefile_path)
    
    print(f"  ✓ Created shapefile: {shapefile_path.name}")
    print(f"  ✓ Converted {len(gdf)} polygons")
    
    return str(boundary_dir)

def main():

    """Main entry point"""
    print("🤖 Automated Ward Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = AutomatedWardPipeline()
        
        # Run the pipeline
        results = pipeline.run_pipeline(ANALYSIS_CONFIGS)
        
        print(f"\n🎉 Pipeline completed!")
        
    except Exception as e:
        print(f"\n💥 Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
