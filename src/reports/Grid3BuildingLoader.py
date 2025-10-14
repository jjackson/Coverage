# Grid3BuildingLoader.py

"""
Grid3 Building Data Loader

Loads building footprint data from CSV files and matches to ward boundaries.
Handles file discovery, ward matching, and data consolidation.

Expected filename format:
    {state}_state_{lga}_lga_{ward}_ward_{subarea}_buildings_data.csv
    Example: Sokoto_state_Sokoto_lga_Kware_ward_Durbawa_buildings_data.csv

Building data columns used:
    - latitude, longitude: Building location
    - area_in_meters: Building footprint area
    - confidence: Confidence score (0-1)
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np


class Grid3BuildingLoader:
    """Loads and matches building data to ward boundaries"""
    
    def __init__(self, log_func=None):
        """
        Initialize loader
        
        Args:
            log_func: Optional logging function
        """
        self.log = log_func if log_func else lambda x: print(x)
    
    def load_buildings_for_wards(self, buildings_dir, ward_boundaries_gdf):
        """
        Load building data for all wards with available data
        
        Args:
            buildings_dir: Directory containing building CSV files
            ward_boundaries_gdf: GeoDataFrame with ward boundaries
            
        Returns:
            DataFrame with columns: ward_id, latitude, longitude, area_in_meters, confidence
            Returns None if no building data found
        """
        if not buildings_dir or not os.path.exists(buildings_dir):
            self.log("[buildings] No buildings directory specified or not found")
            return None
        
        buildings_path = Path(buildings_dir)
        
        # Find all building CSV files
        building_files = list(buildings_path.glob("*_buildings_data.csv"))
        
        if not building_files:
            self.log(f"[buildings] No building files found in {buildings_dir}")
            return None
        
        self.log(f"[buildings] Found {len(building_files)} building data files")
        
        # Parse filenames and match to wards
        file_ward_map = self._match_files_to_wards(building_files, ward_boundaries_gdf)
        
        if not file_ward_map:
            self.log("[buildings] No building files could be matched to wards")
            return None
        
        # Load and consolidate building data
        all_buildings = []
        
        for file_path, ward_info in file_ward_map.items():
            try:
                buildings_df = self._load_building_file(file_path, ward_info)
                if buildings_df is not None and len(buildings_df) > 0:
                    all_buildings.append(buildings_df)
                    self.log(f"[buildings] Loaded {len(buildings_df):,} buildings for ward {ward_info['ward_id']}")
            except Exception as e:
                self.log(f"[buildings] Error loading {file_path.name}: {str(e)}")
                continue
        
        if not all_buildings:
            self.log("[buildings] No building data successfully loaded")
            return None
        
        # Concatenate all building data
        consolidated_df = pd.concat(all_buildings, ignore_index=True)
        
        self.log(f"[buildings] Total buildings loaded: {len(consolidated_df):,} across {len(file_ward_map)} wards")
        
        return consolidated_df
    
    def _match_files_to_wards(self, building_files, ward_boundaries_gdf):
        """
        Match building files to ward IDs based on filename parsing
        
        Args:
            building_files: List of Path objects to building CSV files
            ward_boundaries_gdf: GeoDataFrame with ward boundaries
            
        Returns:
            Dict mapping file_path -> ward_info dict
        """
        file_ward_map = {}
        unmatched_files = []
        
        for file_path in building_files:
            parsed = self._parse_building_filename(file_path.name)
            
            if not parsed:
                unmatched_files.append(file_path.name)
                continue
            
            # Find matching ward in boundaries
            ward_match = self._find_matching_ward(parsed, ward_boundaries_gdf)
            
            if ward_match:
                file_ward_map[file_path] = ward_match
            else:
                unmatched_files.append(file_path.name)
        
        self.log(f"[buildings] Matched {len(file_ward_map)} files to wards")
        
        if unmatched_files:
            self.log(f"[buildings] Could not match {len(unmatched_files)} files:")
            for fname in unmatched_files[:5]:  # Show first 5
                self.log(f"[buildings]   - {fname}")
            if len(unmatched_files) > 5:
                self.log(f"[buildings]   ... and {len(unmatched_files) - 5} more")
        
        return file_ward_map
    
    def _parse_building_filename(self, filename):
        """
        Parse building filename to extract state, lga, and ward
        
        Format: {state}_state_{lga}_lga_{ward}_ward_{subarea}_buildings_data.csv
        
        Args:
            filename: String filename
            
        Returns:
            Dict with state, lga, ward keys, or None if parsing fails
        """
        # Pattern: capture state, lga, and ward from standard format
        pattern = r'^(.+?)_state_(.+?)_lga_(.+?)_ward_(.+?)_buildings_data\.csv$'
        
        match = re.match(pattern, filename)
        
        if not match:
            return None
        
        state = match.group(2).replace('_', ' ').title()
        lga = match.group(3).replace('_', ' ').title()
        ward = match.group(4).replace('_', ' ').title()
        
        return {
            'state': state,
            'lga': lga,
            'ward': ward,
            'filename': filename
        }
    
    def _find_matching_ward(self, parsed_info, ward_boundaries_gdf):
        """
        Find ward in boundaries that matches parsed filename info
        
        Args:
            parsed_info: Dict with state, lga, ward
            ward_boundaries_gdf: GeoDataFrame with ward boundaries
            
        Returns:
            Dict with ward_id, ward_name, state_name, or None if no match
        """
        # Try exact match on state and ward
        matches = ward_boundaries_gdf[
            (ward_boundaries_gdf['state_name'].str.lower() == parsed_info['state'].lower()) &
            (ward_boundaries_gdf['ward_name'].str.lower() == parsed_info['ward'].lower())
        ]
        
        if len(matches) == 1:
            match = matches.iloc[0]
            return {
                'ward_id': match['ward_id'],
                'ward_name': match['ward_name'],
                'state_name': match['state_name']
            }
        elif len(matches) > 1:
            # Multiple wards with same name - this shouldn't happen with state match
            self.log(f"[buildings] Warning: Multiple wards match {parsed_info['state']}/{parsed_info['ward']}")
            match = matches.iloc[0]
            return {
                'ward_id': match['ward_id'],
                'ward_name': match['ward_name'],
                'state_name': match['state_name']
            }
        
        # No match found
        return None
    
    def _load_building_file(self, file_path, ward_info):
        """
        Load building data from CSV file
        
        Args:
            file_path: Path to building CSV file
            ward_info: Dict with ward_id, ward_name, state_name
            
        Returns:
            DataFrame with ward_id, latitude, longitude, area_in_meters, confidence
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_cols = ['latitude', 'longitude', 'area_in_meters', 'confidence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.log(f"[buildings] Warning: {file_path.name} missing columns: {missing_cols}")
                return None
            
            # Select and clean columns
            buildings_df = df[required_cols].copy()
            
            # Add ward info
            buildings_df['ward_id'] = ward_info['ward_id']
            buildings_df['ward_name'] = ward_info['ward_name']
            buildings_df['state_name'] = ward_info['state_name']
            
            # Clean data
            buildings_df = buildings_df.dropna(subset=['latitude', 'longitude']).copy()
            
            # Validate coordinates
            valid_coords = (
                buildings_df['latitude'].between(-90, 90, inclusive='both') &
                buildings_df['longitude'].between(-180, 180, inclusive='both')
            )
            buildings_df = buildings_df[valid_coords].copy()
            
            # Ensure numeric types
            buildings_df['area_in_meters'] = pd.to_numeric(buildings_df['area_in_meters'], errors='coerce')
            buildings_df['confidence'] = pd.to_numeric(buildings_df['confidence'], errors='coerce')
            
            # Fill NaN values
            buildings_df['area_in_meters'] = buildings_df['area_in_meters'].fillna(0)
            buildings_df['confidence'] = buildings_df['confidence'].fillna(0)
            
            return buildings_df
            
        except Exception as e:
            self.log(f"[buildings] Error reading {file_path.name}: {str(e)}")
            return None


def load_buildings_for_analysis(buildings_dir, ward_boundaries_gdf, log_func=None):
    """
    Convenience function to load buildings for ward analysis
    
    Args:
        buildings_dir: Directory containing building CSV files
        ward_boundaries_gdf: GeoDataFrame with ward boundaries
        log_func: Optional logging function
        
    Returns:
        DataFrame with building data, or None if unavailable
    """
    loader = Grid3BuildingLoader(log_func)
    return loader.load_buildings_for_wards(buildings_dir, ward_boundaries_gdf)
