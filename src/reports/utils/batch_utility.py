"""
Shared Batching Utility for FLW Analysis
Provides consistent batching logic across different analysis types

Save this file as: src/reports/utils/batch_utility.py
"""

import pandas as pd


class BatchUtility:
    """Utility class for creating consistent batches across FLW analysis types"""
    
    def __init__(self, batch_size=300, min_size=100):
        self.batch_size = batch_size
        self.min_size = min_size
    
    def create_flw_batches(self, flw_visits, include_all_batch=False):
        """
        Create batches for a single FLW/opportunity pair working backwards from most recent visits
        
        Args:
            flw_visits: DataFrame with visits for one FLW/opp pair, must have 'visit_date' column
            include_all_batch: If True, add an "all" batch with all visits
            
        Returns:
            List of tuples: [(batch_df, batch_number, start_date, end_date, visit_count), ...]
            If include_all_batch=True, "all" batch has batch_number="all"
        """
        if len(flw_visits) < self.min_size:
            return []
        
        # Sort chronologically (oldest first)
        flw_visits_sorted = flw_visits.sort_values('visit_date').copy()
        
        # Group by date to respect day boundaries
        flw_visits_sorted['visit_day'] = flw_visits_sorted['visit_date'].dt.date
        daily_groups = flw_visits_sorted.groupby('visit_day')
        
        # Get list of (date, day_data) sorted chronologically
        daily_data = [(date, group) for date, group in daily_groups]
        daily_data.sort(key=lambda x: x[0])  # Sort by date
        
        # Work backwards to create batches
        batches = []
        remaining_days = list(reversed(daily_data))  # Start from most recent
        
        while remaining_days:
            # Try to build a batch working backwards
            batch_days = []
            batch_visits = 0
            
            # Add days until we hit batch_size or run out of days
            for i, (date, day_data) in enumerate(remaining_days):
                day_visit_count = len(day_data)
                
                # Check if adding this day would exceed batch_size
                if batch_visits + day_visit_count > self.batch_size and batch_visits > 0:
                    break
                
                batch_days.append((date, day_data))
                batch_visits += day_visit_count
                
                # If we've reached or exceeded min_size, we can stop here if needed
                if batch_visits >= self.batch_size:
                    break
            
            # Check if this batch meets minimum size
            if batch_visits < self.min_size:
                break
            
            # Create batch DataFrame (reverse to get chronological order within batch)
            batch_days.reverse()
            batch_df = pd.concat([day_data for _, day_data in batch_days], ignore_index=True)
            
            # Calculate batch metadata
            start_date = batch_df['visit_date'].min()
            end_date = batch_df['visit_date'].max()
            
            batches.append((batch_df, start_date, end_date, len(batch_df)))
            
            # Remove used days from remaining_days
            used_days = len(batch_days)
            remaining_days = remaining_days[used_days:]
        
        # Reverse batches to get chronological order (oldest = batch 1)
        batches.reverse()
        
        # Add batch numbers
        numbered_batches = []
        for i, (batch_df, start_date, end_date, visit_count) in enumerate(batches):
            batch_number = i + 1
            numbered_batches.append((batch_df, batch_number, start_date, end_date, visit_count))
        
        # Add "all" batch if requested and we have enough total visits
        if include_all_batch and len(flw_visits_sorted) >= self.min_size:
            all_start_date = flw_visits_sorted['visit_date'].min()
            all_end_date = flw_visits_sorted['visit_date'].max()
            numbered_batches.append((
                flw_visits_sorted, 
                "all", 
                all_start_date, 
                all_end_date, 
                len(flw_visits_sorted)
            ))
        
        return numbered_batches
    
    def create_all_flw_batches(self, df, flw_id_col, opportunity_col='opportunity_name', include_all_batch=False):
        """
        Create batches for all FLW/opportunity pairs in a dataset
        
        Args:
            df: DataFrame with visit data
            flw_id_col: Column name for FLW ID
            opportunity_col: Column name for opportunity 
            include_all_batch: If True, add "all" batch for each FLW/opp pair
            
        Returns:
            List of dicts with batch metadata and data
        """
        if flw_id_col not in df.columns:
            raise ValueError(f"FLW ID column '{flw_id_col}' not found in data")
        
        if 'visit_date' not in df.columns:
            raise ValueError("visit_date column not found in data")
        
        # Get FLW/opportunity pairs with sufficient visits
        if opportunity_col in df.columns:
            flw_opp_visits = df.groupby([flw_id_col, opportunity_col]).size().reset_index(name='visit_count')
            eligible_pairs = flw_opp_visits[flw_opp_visits['visit_count'] >= self.min_size]
            group_cols = [flw_id_col, opportunity_col]
        else:
            # No opportunity column - just group by FLW
            flw_visits = df.groupby([flw_id_col]).size().reset_index(name='visit_count')
            eligible_pairs = flw_visits[flw_visits['visit_count'] >= self.min_size]
            group_cols = [flw_id_col]
        
        all_batch_records = []
        
        for _, pair in eligible_pairs.iterrows():
            # Get filter conditions
            if opportunity_col in df.columns:
                flw_id = pair[flw_id_col]
                opp_name = pair[opportunity_col]
                flw_visits = df[
                    (df[flw_id_col] == flw_id) & 
                    (df[opportunity_col] == opp_name)
                ]
            else:
                flw_id = pair[flw_id_col]
                opp_name = None
                flw_visits = df[df[flw_id_col] == flw_id]
            
            # Create batches for this FLW/opp pair
            batches = self.create_flw_batches(flw_visits, include_all_batch)
            
            # Convert to records with metadata
            for batch_df, batch_number, start_date, end_date, visit_count in batches:
                record = {
                    'flw_id': flw_id,
                    'batch_number': batch_number,
                    'batch_start_date': start_date.date() if hasattr(start_date, 'date') else start_date,
                    'batch_end_date': end_date.date() if hasattr(end_date, 'date') else end_date,
                    'visits_in_batch': visit_count,
                    'batch_data': batch_df
                }
                
                if opp_name is not None:
                    record['opportunity_name'] = opp_name
                
                all_batch_records.append(record)
        
        return all_batch_records
