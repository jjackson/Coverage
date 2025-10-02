#!/usr/bin/env python3
"""
Village Settlement Analysis Pipeline

Matches a list of villages against Grid3 settlement data, then runs the same
Grid3 analysis workflow as the ward pipeline but focused on settlement points
using real visit data.

Workflow:
1. Load village list (state, LGA, ward, village name)
2. Download real visit data from Superset
3. Load Grid3 settlement data and match by name/location
4. Create "settlement boundaries" (buffered points)
5. Filter visit data to settlement areas
6. Run Grid3 analysis for matched settlements with real visits

Input:
- CSV with columns: state, lga, ward, village_name
- Superset query for visit data
- Grid3 settlement data (shapefile with point locations)
- Grid3 population raster

Output:
- Matched settlements with Grid3 analysis using real visit data
- Settlement-level population and visit analysis
- Interactive maps of settlements with real visit points
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import warnings
import requests
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("? Loaded environment variables from .env file")
except ImportError:
    print("??  python-dotenv not installed - using system environment variables only")

# Option 1: Use fuzzywuzzy (install with: pip install fuzzywuzzy python-levenshtein)
# from fuzzywuzzy import fuzz, process

# Option 2: Use rapidfuzz (install with: pip install rapidfuzz) - faster alternative
try:
    from rapidfuzz import fuzz, process
except ImportError:
    from fuzzywuzzy import fuzz, process

# Add src to path for imports
sys.path.append('src')

from src.utils.data_loader import export_superset_query_with_pagination
from src.reports.Grid3WardAnalysis import Grid3WardAnalysis

class VillageSettlementMatcher:
    """Matches village names against Grid3 settlement data"""
    
    def __init__(self, log_func=print):
        self.log = log_func
        
    def load_village_list(self, village_file):
        """Load village list from CSV"""
        df = pd.read_csv(village_file)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Required columns
        required_cols = ['state', 'lga', 'ward', 'village_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.log(f"Loaded {len(df)} villages from {village_file}")
        return df
    
    def load_settlement_data(self, settlement_file):
        """Load Grid3 settlement data"""
        if settlement_file.endswith('.shp'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = gpd.read_file(settlement_file)
        elif settlement_file.endswith('.csv'):
            df = pd.read_csv(settlement_file)
            # Handle coordinate columns for Grid3 CSV format
            lat_cols = ['latitude', 'lat', 'y']
            lon_cols = ['longitude', 'lon', 'lng', 'x']
            
            lat_col = next((c for c in lat_cols if c in df.columns), None)
            lon_col = next((c for c in lon_cols if c in df.columns), None)
            
            if not lat_col or not lon_col:
                raise ValueError(f"Settlement CSV must have latitude/longitude columns. Available columns: {list(df.columns)}")
            
            # Clean coordinate data
            df = df.dropna(subset=[lat_col, lon_col])
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        else:
            raise ValueError("Settlement file must be .shp or .csv")
        
        # Ensure WGS84
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        self.log(f"Loaded {len(gdf)} settlements from {settlement_file}")
        self.log(f"Available columns: {list(gdf.columns)}")
        
        return gdf
    
    def match_villages_to_settlements(self, villages_df, settlements_gdf, 
                                    name_threshold=80, location_threshold_km=50):
        """
        Match villages to settlements using fuzzy name matching and location filtering
        """
        
        # Find name column in settlements - update for Grid3 format
        name_candidates = ['set_name', 'name', 'settlement_name', 'place_name', 'village_name', 'set_altnam']
        settlement_name_col = None
        for col in name_candidates:
            if col in settlements_gdf.columns:
                settlement_name_col = col
                self.log(f"Using settlement name column: {col}")
                break
        
        if not settlement_name_col:
            self.log(f"Available columns: {list(settlements_gdf.columns)}")
            raise ValueError(f"No name column found in settlements. Tried: {name_candidates}")
        
        # Find state/LGA columns in settlements for location filtering - update for Grid3 format
        state_col = next((c for c in ['statename', 'state', 'state_name'] if c in settlements_gdf.columns), None)
        lga_col = next((c for c in ['lganame', 'lga', 'lga_name'] if c in settlements_gdf.columns), None)
        ward_col = next((c for c in ['wardname', 'ward', 'ward_name'] if c in settlements_gdf.columns), None)
        
        self.log(f"Location columns found - State: {state_col}, LGA: {lga_col}, Ward: {ward_col}")
        
        matches = []
        
        for idx, village in villages_df.iterrows():
            village_name = str(village['village_name']).strip()
            village_state = str(village['state']).strip().lower()
            village_lga = str(village['lga']).strip().lower()
            village_ward = str(village['ward']).strip().lower() if 'ward' in village else None
            
            self.log(f"\nSearching for: {village_name} in {village_state}/{village_lga}")
            
            # Filter settlements by location if possible
            candidate_settlements = settlements_gdf.copy()
            
            if state_col:
                state_mask = candidate_settlements[state_col].str.lower().str.contains(
                    village_state, na=False, regex=False
                )
                before_count = len(candidate_settlements)
                candidate_settlements = candidate_settlements[state_mask]
                self.log(f"  State filter: {before_count} -> {len(candidate_settlements)} settlements")
            
            if lga_col and len(candidate_settlements) > 0:
                lga_mask = candidate_settlements[lga_col].str.lower().str.contains(
                    village_lga, na=False, regex=False
                )
                before_count = len(candidate_settlements)
                location_filtered = candidate_settlements[lga_mask]
                if len(location_filtered) > 0:
                    candidate_settlements = location_filtered
                    self.log(f"  LGA filter: {before_count} -> {len(candidate_settlements)} settlements")
            
            if ward_col and village_ward and len(candidate_settlements) > 0:
                ward_mask = candidate_settlements[ward_col].str.lower().str.contains(
                    village_ward, na=False, regex=False
                )
                before_count = len(candidate_settlements)
                ward_filtered = candidate_settlements[ward_mask]
                if len(ward_filtered) > 0:
                    candidate_settlements = ward_filtered
                    self.log(f"  Ward filter: {before_count} -> {len(candidate_settlements)} settlements")
            
            if len(candidate_settlements) == 0:
                self.log(f"  No location candidates for {village_name}")
                continue
            
            # Show some sample names for debugging
            sample_names = candidate_settlements[settlement_name_col].dropna().head(5).tolist()
            self.log(f"  Sample settlement names: {sample_names}")
            
            # Fuzzy name matching
            settlement_names = candidate_settlements[settlement_name_col].astype(str).tolist()
            
            if not settlement_names:
                self.log(f"  No settlement names found")
                continue
            
            # Find best match
            best_match = process.extractOne(
                village_name, 
                settlement_names, 
                scorer=fuzz.token_sort_ratio
            )
            
            if best_match:
                match_name, match_score = best_match[0], best_match[1]
                self.log(f"  Best match: '{match_name}' (score: {match_score})")
                
                if match_score >= name_threshold:
                    # Find the settlement row
                    settlement_mask = candidate_settlements[settlement_name_col].astype(str) == match_name
                    matched_settlements = candidate_settlements[settlement_mask]
                    
                    if len(matched_settlements) > 0:
                        settlement = matched_settlements.iloc[0]
                        
                        match_info = {
                            'village_idx': idx,
                            'village_name': village_name,
                            'village_state': village['state'],
                            'village_lga': village['lga'],
                            'village_ward': village['ward'],
                            'settlement_name': match_name,
                            'match_score': match_score,
                            'settlement_latitude': settlement.geometry.y,
                            'settlement_longitude': settlement.geometry.x,
                            'settlement_geometry': settlement.geometry
                        }
                        
                        # Add any additional settlement attributes
                        for col in settlement.index:
                            if col not in ['geometry'] and not col.startswith('settlement_'):
                                match_info[f'settlement_{col}'] = settlement[col]
                        
                        matches.append(match_info)
                        
                        self.log(f"  ? MATCHED: {village_name} -> {match_name} (score: {match_score})")
                    else:
                        self.log(f"  Name matched but no settlement found: {village_name}")
                else:
                    self.log(f"  Score too low: {match_score} < {name_threshold}")
            else:
                self.log(f"  No fuzzy match found for {village_name}")
        
        matches_df = pd.DataFrame(matches)
        self.log(f"\nSuccessfully matched {len(matches_df)} of {len(villages_df)} villages")
        
        return matches_df
    
    def create_settlement_boundaries(self, matches_df, buffer_distance_m=1000):
        """Create buffered boundaries around settlement points"""
        if len(matches_df) == 0:
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame from matches
        geometries = [Point(row['settlement_longitude'], row['settlement_latitude']) 
                     for _, row in matches_df.iterrows()]
        
        gdf = gpd.GeoDataFrame(matches_df, geometry=geometries, crs="EPSG:4326")
        
        # Convert to UTM for accurate buffering (use UTM Zone 32N for Nigeria)
        utm_crs = "EPSG:32632"
        gdf_utm = gdf.to_crs(utm_crs)
        
        # Apply buffer
        gdf_utm['geometry'] = gdf_utm.geometry.buffer(buffer_distance_m)
        
        # Convert back to WGS84
        settlement_boundaries = gdf_utm.to_crs("EPSG:4326")
        
        # Add settlement metadata with ward-compatible column names
        settlement_boundaries['settlement_id'] = (
            settlement_boundaries['village_state'] + "_" + 
            settlement_boundaries['village_lga'] + "_" + 
            settlement_boundaries['village_name'].str.replace(' ', '_')
        )
        
        # Create ward-compatible columns for Grid3Analysis
        settlement_boundaries['ward_id'] = settlement_boundaries['settlement_id']  # Use settlement_id as ward_id
        settlement_boundaries['ward_name'] = settlement_boundaries['village_name']  # Use village name as ward name
        settlement_boundaries['state_name'] = settlement_boundaries['village_state']  # Map to expected column
        
        settlement_boundaries['buffer_distance_m'] = buffer_distance_m
        
        self.log(f"Created {len(settlement_boundaries)} settlement boundaries with {buffer_distance_m}m buffer")
        
        return settlement_boundaries


class VillageSettlementPipeline:
    """Complete pipeline for village settlement analysis using real visit data"""
    
    def __init__(self, base_output_dir=r"C:\Users\Neal Lesh\Coverage\automated_village_pipeline_output"):
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
        
        # Initialize matcher with our log function
        self.matcher = VillageSettlementMatcher(log_func=self.log)
        
        print(f"Working directory: {self.today_dir}")
    
    def log(self, message):
        """Simple logging function"""
        print(f"[village_pipeline] {message}")
    
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
            
        print(f"? Superset credentials loaded: {self.superset_url}")
    
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
    
    def _download_visit_data(self, superset_query_id):
        """Download visit data from Superset"""
        csv_file = self.superset_data_dir / "gw8_village_data.csv"
        
        if csv_file.exists():
            self.log(f"Using cached visit data: {csv_file.name}")
            df = pd.read_csv(csv_file)
        else:
            self.log(f"Downloading visit data from query {superset_query_id}...")
            try:
                # Get SQL from saved query
                sql_query = self._get_sql_from_saved_query(superset_query_id)
                
                # Download data
                downloaded_file = export_superset_query_with_pagination(
                    superset_url=self.superset_url,
                    sql_query=sql_query,
                    username=self.superset_username,
                    password=self.superset_password,
                    output_filename=str(csv_file.with_suffix(''))  # Remove .csv as function adds it
                )
                
                # Load the downloaded data
                if os.path.exists(downloaded_file):
                    df = pd.read_csv(downloaded_file)
                    self.log(f"Downloaded {len(df):,} visit records")
                else:
                    raise RuntimeError("Download failed - no file created")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to download visit data: {str(e)}")
        
        return df
    
    def _prepare_visit_data(self, visit_df):
        """Prepare and validate visit data (similar to Grid3WardAnalysis)"""
        data = visit_df.copy()
        data.columns = data.columns.str.lower().str.strip()

        # Find coordinate columns
        col_map = {
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "lng", "x"],
            "opportunity_id": ["opportunity_id"],
            "opportunity_name": ["opportunity_name", "opportunity"],
            "visit_id": ["visit_id", "visitid", "id"],
            "flw_id": ["flw_id", "user_id", "username", "enumerator_id", "worker_id"],
        }

        def pick_column(candidates):
            for c in candidates:
                if c in data.columns:
                    return c
            return None

        lat_col = pick_column(col_map["latitude"])
        lon_col = pick_column(col_map["longitude"])
        
        if lat_col is None or lon_col is None:
            raise ValueError("Visit data must include latitude/longitude columns")

        # Rename columns
        rename_map = {lat_col: "latitude", lon_col: "longitude"}
        
        for key, candidates in col_map.items():
            if key not in ["latitude", "longitude"]:
                col = pick_column(candidates)
                if col and col in data.columns:
                    rename_map[col] = key
                    
        data = data.rename(columns=rename_map)

        # Clean and validate coordinates
        data = data.dropna(subset=["latitude", "longitude"]).copy()
        valid_coords = (
            data["latitude"].between(-90, 90, inclusive="both") & 
            data["longitude"].between(-180, 180, inclusive="both")
        )
        data = data[valid_coords].reset_index(drop=True)

        # Add default columns if missing
        if "opportunity_id" not in data.columns:
            data["opportunity_id"] = "GW8"
        if "opportunity_name" not in data.columns:
            data["opportunity_name"] = data["opportunity_id"]
        if "visit_id" not in data.columns:
            data["visit_id"] = "visit_" + data.index.astype(str)
        if "flw_id" not in data.columns:
            # Create synthetic FLW IDs based on unique lat/lon combinations
            coords_str = data["latitude"].astype(str) + "_" + data["longitude"].astype(str)
            unique_coords = coords_str.drop_duplicates()
            coord_to_flw = {coord: f"flw_{i}" for i, coord in enumerate(unique_coords)}
            data["flw_id"] = coords_str.map(coord_to_flw)

        return data
    
    def _visits_to_geodataframe(self, visits_df):
        """Convert visits DataFrame to GeoDataFrame"""
        geometry = [Point(xy) for xy in zip(visits_df['longitude'], visits_df['latitude'])]
        visits_gdf = gpd.GeoDataFrame(visits_df, geometry=geometry, crs="EPSG:4326")
        return visits_gdf
    
    def _clean_spatial_join_result(self, joined_gdf):
        """Clean up the result of spatial join to remove duplicate columns"""
        # Keep only essential columns, avoiding duplicates from the join
        essential_cols = ['visit_id', 'opportunity_id', 'opportunity_name', 'flw_id', 
                         'latitude', 'longitude', 'settlement_id', 'village_name', 
                         'village_state', 'village_lga', 'village_ward']
        
        # Find available columns (some might have _left or _right suffixes)
        available_cols = []
        for col in essential_cols:
            if col in joined_gdf.columns:
                available_cols.append(col)
            elif f"{col}_left" in joined_gdf.columns:
                available_cols.append(f"{col}_left")
                joined_gdf[col] = joined_gdf[f"{col}_left"]
        
        # Select only the columns we need
        result_df = joined_gdf[available_cols].copy()
        
        # Drop geometry column for CSV output
        if 'geometry' in result_df.columns:
            result_df = result_df.drop('geometry', axis=1)
        
        return result_df
    
    def _create_minimal_synthetic_visits(self, settlement_boundaries):
        """Create minimal synthetic visits if no real visits found in settlement areas"""
        synthetic_visits = []
        for _, settlement in settlement_boundaries.iterrows():
            # Get centroid of buffered area
            centroid = settlement.geometry.centroid
            
            synthetic_visits.append({
                'visit_id': f"synthetic_{settlement['settlement_id']}",
                'opportunity_id': "GW8",
                'opportunity_name': "GW8",
                'flw_id': f"synthetic_flw_{settlement['settlement_id']}",
                'latitude': centroid.y,
                'longitude': centroid.x,
                'settlement_id': settlement['settlement_id'],
                'village_name': settlement['village_name'],
                'village_state': settlement['village_state'],
                'village_lga': settlement['village_lga'],
                'village_ward': settlement['village_ward']
            })
        
        return pd.DataFrame(synthetic_visits)
    
    def run_village_analysis(self, config):
        """Run complete village analysis for a single configuration using real visit data"""
        
        self.log(f"Starting analysis: {config['name']}")
        
        # Create analysis directory
        analysis_dir = self.today_dir / f"{config['name'].lower()}_village_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Step 1: Download real visit data
        self.log("Step 1: Downloading real visit data from Superset...")
        visit_data_df = self._download_visit_data(config['superset_query_id'])
        
        # Step 2: Load and match villages
        self.log("Step 2: Loading and matching villages to settlements...")
        
        villages_df = self.matcher.load_village_list(config['village_file'])
        settlements_gdf = self.matcher.load_settlement_data(config['settlement_file'])
        
        matches_df = self.matcher.match_villages_to_settlements(
            villages_df, 
            settlements_gdf,
            name_threshold=config.get('name_threshold', 80),
            location_threshold_km=config.get('location_threshold_km', 50)
        )
        
        if len(matches_df) == 0:
            raise ValueError("No village matches found")
        
        # Step 3: Create settlement boundaries
        self.log("Step 3: Creating settlement boundaries...")
        
        settlement_boundaries = self.matcher.create_settlement_boundaries(
            matches_df, 
            buffer_distance_m=config.get('buffer_distance_m', 1000)
        )
        
        # Save settlement boundaries
        boundaries_dir = analysis_dir / "settlement_boundaries"
        boundaries_dir.mkdir(exist_ok=True)
        
        # Create a data tag for consistent naming (similar to ward extractor)
        data_tag = f"{len(settlement_boundaries)}settlements"
        
        # Save with ward-compatible filename so Grid3Analysis can find it
        boundaries_file = boundaries_dir / f"affected_wards_{data_tag}.shp"
        settlement_boundaries.to_file(boundaries_file)
        
        # Also save with descriptive name for reference
        descriptive_file = boundaries_dir / "settlement_boundaries.shp"
        settlement_boundaries.to_file(descriptive_file)
        
        matches_file = boundaries_dir / "village_settlement_matches.csv"
        matches_df.to_csv(matches_file, index=False)
        
        self.log(f"Saved settlement boundaries (ward-compatible): {boundaries_file}")
        self.log(f"Saved settlement boundaries (descriptive): {descriptive_file}")
        self.log(f"Saved matches: {matches_file}")
        
        # Step 4: Filter visit data to settlements within our areas of interest
        self.log("Step 4: Filtering visit data to settlement areas...")
        
        # Convert visit data to GeoDataFrame
        visit_data_clean = self._prepare_visit_data(visit_data_df)
        visit_gdf = self._visits_to_geodataframe(visit_data_clean)
        
        # Spatial join to find visits within settlement boundaries
        visits_in_settlements = gpd.sjoin(
            visit_gdf, 
            settlement_boundaries, 
            how='inner', 
            predicate='intersects'
        )
        
        # Clean up the joined data (remove duplicate columns from join)
        visits_cleaned = self._clean_spatial_join_result(visits_in_settlements)
        
        self.log(f"Filtered {len(visit_data_df):,} total visits to {len(visits_cleaned):,} visits in settlement areas")
        
        if len(visits_cleaned) == 0:
            self.log("Warning: No visits found in settlement areas")
            # Create minimal synthetic data so analysis can still run
            visits_cleaned = self._create_minimal_synthetic_visits(settlement_boundaries)
            self.log(f"Created {len(visits_cleaned)} minimal synthetic visits for analysis")
        
        # Save filtered visit data
        filtered_visits_file = analysis_dir / "visits_in_settlement_areas.csv"
        visits_cleaned.to_csv(filtered_visits_file, index=False)
        self.log(f"Saved filtered visits: {filtered_visits_file}")
        
        # Step 5: Run Grid3 analysis using settlement boundaries and real visits
        self.log("Step 5: Running Grid3 analysis on settlement areas with real visit data...")
        
        grid3_analysis_dir = analysis_dir / "grid3_analysis"
        grid3_analysis_dir.mkdir(exist_ok=True)
        
        # Create Grid3 analyzer with real visit data
        grid3_analyzer = Grid3WardAnalysis.create_for_automation(
            df=visits_cleaned,
            output_dir=str(grid3_analysis_dir),
            stage1_folder=str(boundaries_dir),  # Use our settlement boundaries
            grid3_file=config['grid3_file'],
            include_partial=True,
            expected_visits_per_pop=config.get('expected_visits_per_pop', 0.18)
        )
        
        # Add logging
        def grid3_log_func(message):
            print(f"    [grid3] {message}")
        grid3_analyzer.log = grid3_log_func
        
        try:
            analysis_files = grid3_analyzer.generate()
            self.log(f"Grid3 analysis complete: {len(analysis_files)} files")
        except Exception as e:
            raise RuntimeError(f"Grid3 analysis failed: {str(e)}")
        
        # Step 6: Create summary report
        self.log("Step 6: Creating summary report...")
        
        summary_data = {
            'analysis_name': config['name'],
            'villages_input': len(villages_df),
            'settlements_available': len(settlements_gdf),
            'villages_matched': len(matches_df),
            'match_rate_pct': len(matches_df) / len(villages_df) * 100,
            'buffer_distance_m': config.get('buffer_distance_m', 1000),
            'total_visits_downloaded': len(visit_data_df),
            'visits_in_settlement_areas': len(visits_cleaned),
            'visit_coverage_rate_pct': len(visits_cleaned) / len(visit_data_df) * 100 if len(visit_data_df) > 0 else 0,
            'grid3_analysis_files': len(analysis_files)
        }
        
        summary_file = analysis_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        all_files = [boundaries_file, descriptive_file, matches_file, filtered_visits_file, summary_file] + analysis_files
        
        self.log(f"Analysis complete! Generated {len(all_files)} files")
        
        return {
            'output_dir': str(analysis_dir),
            'files': all_files,
            'summary': summary_data
        }


# Configuration for your village analysis using real GW8 visit data
VILLAGE_ANALYSIS_CONFIG = {
    "name": "IPA_Villages",
    "village_file": r"C:\Users\Neal Lesh\Coverage\data\grid3\settlement\village_names.csv",
    "settlement_file": r"C:\Users\Neal Lesh\Coverage\data\grid3\settlement\settlement_data\grid3_nga_settlementpt.shp",
    "superset_query_id": 193,  # GW8 query from your automated pipeline
    "grid3_file": r"C:\Users\Neal Lesh\Coverage\data\grid3\NGA_population_v3_0_gridded.tif",
    "buffer_distance_m": 500,  # 1km buffer around each settlement
    "name_threshold": 60,  # Lower threshold since "Jigawa" vs "Jigawa Bature" should match
    "location_threshold_km": 50,  # Maximum distance for location filtering
    "expected_visits_per_pop": 0.18,  # 18 visits per 100 people
    "description": "Analysis of Nigerian villages using real GW8 visit data"
}


def main():
    """Main entry point"""
    print("Village Settlement Analysis Pipeline")
    print("=" * 50)
    
    try:
        print("DEBUG: About to create pipeline instance")
        # Initialize pipeline
        pipeline = VillageSettlementPipeline()
        print("DEBUG: Pipeline created successfully")
        
        # Run the analysis
        result = pipeline.run_village_analysis(VILLAGE_ANALYSIS_CONFIG)
        
        print(f"\nPipeline completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        print(f"Files generated: {len(result['files'])}")
        print(f"Match rate: {result['summary']['match_rate_pct']:.1f}%")
        print(f"Visit coverage rate: {result['summary']['visit_coverage_rate_pct']:.1f}%")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
