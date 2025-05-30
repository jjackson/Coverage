"""Data cleaning utilities for FLW analysis"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataCleaner:
    def __init__(self, df, log_func):
        self.df = df.copy()
        self.log = log_func
    
    def clean_visits_data(self):
        """Clean and prepare visits data"""
        initial_rows = len(self.df)
        
        # Handle latitude/longitude columns
        self._fix_coordinate_columns()
        
        # Remove rows with invalid coordinates
        self._clean_coordinates()
        
        # Parse and clean visit dates
        self._clean_visit_dates()
        
        final_rows = len(self.df)
        self.log(f"Data cleaning complete: {initial_rows} ? {final_rows} rows ({initial_rows - final_rows} removed)")
        
        return self.df
    
    def _fix_coordinate_columns(self):
        """Fix common misspellings in coordinate columns"""
        if 'lattitude' in self.df.columns:
            self.df['latitude'] = self.df['lattitude']
            self.df = self.df.drop('lattitude', axis=1)
            self.log("Renamed 'lattitude' column to 'latitude'")
    
    def _clean_coordinates(self):
        """Remove rows with invalid coordinates"""
        coord_cols = []
        for lat_col in ['latitude', 'lattitude']:
            if lat_col in self.df.columns and 'longitude' in self.df.columns:
                coord_cols = [lat_col, 'longitude']
                break
        
        if len(coord_cols) == 2:
            before_count = len(self.df)
            self.df = self.df.dropna(subset=coord_cols)
            self.df = self.df[
                (self.df[coord_cols[0]] >= -90) & (self.df[coord_cols[0]] <= 90) & 
                (self.df[coord_cols[1]] >= -180) & (self.df[coord_cols[1]] <= 180)
            ]
            removed = before_count - len(self.df)
            if removed > 0:
                self.log(f"Removed {removed} rows with invalid coordinates")

    def _clean_visit_dates(self):
        """Parse and clean visit date column"""
        if 'visit_date' not in self.df.columns:
            return
        
        # Parse visit dates
        try:
            self.df['visit_date'] = pd.to_datetime(self.df['visit_date'], errors='coerce', utc=True)
        except:
            self.df['visit_date'] = pd.to_datetime(self.df['visit_date'], errors='coerce')
        
        # Remove invalid dates
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['visit_date'])
        removed = before_count - len(self.df)
        if removed > 0:
            self.log(f"Removed {removed} rows with invalid dates")
        
        # Create helper columns
        self.df['visit_day'] = self.df['visit_date'].dt.date
        self.df['visit_date_time'] = self.df['visit_date']

    def split_flws_by_opportunity(self, df, flw_id_col):
        """
        Split FLWs who worked on multiple opportunities into separate entities
        
        Args:
            df: Cleaned dataframe 
            flw_id_col: Name of the FLW ID column
            
        Returns:
            DataFrame with FLWs split by opportunity (flw_123_1, flw_123_2, etc.)
        """
        
        if 'opportunity_name' not in df.columns:
            self.log("No opportunity_name column found - skipping FLW split")
            return df
        
        # Find FLWs that worked on multiple opportunities
        flw_opp_counts = df.groupby(flw_id_col)['opportunity_name'].nunique()
        multi_opp_flws = flw_opp_counts[flw_opp_counts > 1].index
        
        if len(multi_opp_flws) == 0:
            self.log("No FLWs found working on multiple opportunities")
            return df
        
        self.log(f"Found {len(multi_opp_flws)} FLWs working on multiple opportunities")
        
        # Process the split
        split_rows = []
        
        for _, row in df.iterrows():
            flw_id = row[flw_id_col]
            
            if pd.isna(flw_id) or flw_id not in multi_opp_flws:
                # Single opportunity FLW - keep as is
                split_rows.append(row)
            else:
                # Multi-opportunity FLW - needs splitting
                # Get all opportunities this FLW worked on
                flw_opportunities = df[df[flw_id_col] == flw_id]['opportunity_name'].unique()
                flw_opportunities = [opp for opp in flw_opportunities if pd.notna(opp)]
                
                # Find which opportunity this specific row belongs to
                current_opp = row['opportunity_name']
                
                if pd.isna(current_opp):
                    # If opportunity is missing for this row, skip it
                    continue
                
                # Find the index of this opportunity for this FLW
                try:
                    opp_index = list(flw_opportunities).index(current_opp) + 1
                except ValueError:
                    # Fallback to 1 if we can't find the opportunity
                    opp_index = 1
                
                # Create new FLW ID with suffix
                new_flw_id = f"{flw_id}_{opp_index}"
                
                # Create modified row
                new_row = row.copy()
                new_row[flw_id_col] = new_flw_id
                split_rows.append(new_row)
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(split_rows)
        
        # Log the results
        original_flw_count = df[flw_id_col].nunique()
        new_flw_count = result_df[flw_id_col].nunique()
        total_split = new_flw_count - original_flw_count
        
        self.log(f"FLW split complete: {original_flw_count} ? {new_flw_count} FLW entities ({total_split} new entities created)")
        
        return result_df
