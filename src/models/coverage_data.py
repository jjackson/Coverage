from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import numpy as np
import time

if TYPE_CHECKING:
    from .service_area import ServiceArea
    from .delivery_unit import DeliveryUnit
    from .service_delivery_point import ServiceDeliveryPoint
    from .flw import FLW


class CoverageData:
    """Data container for coverage analysis"""
    
    def __init__(self):
        self.service_areas: Dict[str, 'ServiceArea'] = {}
        self.delivery_units: Dict[str, 'DeliveryUnit'] = {}
        self.service_points: List['ServiceDeliveryPoint'] = []
        self.flws: Dict[str, 'FLW'] = {}
        self.delivery_units_df: Optional[pd.DataFrame] = None
        
        # Project and opportunity identification
        self.project_space: Optional[str] = None
        self.opportunity_name: Optional[str] = None
        
        # Mapping between FLW CommCare ID and FLW Name
        self.flw_commcare_id_to_name_map: Dict[str, str] = {}
        
        # Additional cached properties
        self.unique_service_area_ids: List[str] = []
        self.unique_flw_names: List[str] = []
        self.unique_status_values: List[str] = []
        self.delivery_status_counts: Dict[str, int] = {}
        
        # Pre-computed aggregated data
        self.flw_service_area_stats: Dict[str, Dict[str, Any]] = {}
        self.service_area_building_density: Dict[str, float] = {}
        self.service_area_progress: Dict[str, Dict[str, Any]] = {}
        self.travel_distances: Dict[str, float] = {}  # Service Area ID -> distance in km
    
    @property
    def total_delivery_units(self) -> int:
        """Get the total number of delivery units"""
        return len(self.delivery_units)
    
    @property
    def total_service_areas(self) -> int:
        """Get the total number of service areas"""
        return len(self.service_areas)
    
    @property
    def total_flws(self) -> int:
        """Get the total number of FLWs"""
        return len(self.flws)
    
    @property
    def total_buildings(self) -> int:
        """Get the total number of buildings across all delivery units"""
        return sum(du.buildings for du in self.delivery_units.values())
    
    @property
    def total_completed_dus(self) -> int:
        """Get the total number of completed delivery units"""
        return sum(1 for du in self.delivery_units.values() if du.status == 'completed')
    
    @property
    def total_visited_dus(self) -> int:
        """Get the total number of visited but not completed delivery units"""
        return sum(1 for du in self.delivery_units.values() if du.status == 'visited')
    
    @property
    def total_unvisited_dus(self) -> int:
        """Get the total number of unvisited delivery units"""
        return sum(1 for du in self.delivery_units.values() if du.status == None)
    
    @property
    def completion_percentage(self) -> float:
        """Get the overall completion percentage"""
        if self.total_delivery_units == 0:
            return 0.0
        return (self.total_completed_dus / self.total_delivery_units) * 100
    
    def _compute_metadata_from_delivery_unit_data(self):
        """Precompute metadata to avoid redundant processing"""
        # Unique service area IDs (sorted for consistency)
        self.unique_service_area_ids = sorted(list(self.service_areas.keys()), key=lambda x: x.zfill(10))
        
        # Unique FLW names
        self.unique_flw_names = sorted(list(self.flws.keys()))
        
        # Unique status values
        status_values = set()
        for du in self.delivery_units.values():
            status_values.add(du.status)
        self.unique_status_values = sorted(list(status_values))
        
        # Count delivery units by status
        self.delivery_status_counts = {}
        for status in self.unique_status_values:
            self.delivery_status_counts[status] = sum(1 for du in self.delivery_units.values() if du.status == status)
            
        # Pre-compute FLW status counts
        for flw in self.flws.values():
            flw.status_counts = {}
            for status in self.unique_status_values:
                flw.status_counts[status] = 0
        
        # Calculate status counts per FLW
        for du in self.delivery_units.values():
            if du.flw_commcare_id in self.flws:
                self.flws[du.flw_commcare_id].status_counts[du.status] = self.flws[du.flw_commcare_id].status_counts.get(du.status, 0) + 1
        
        # Pre-compute service area progress
        self._compute_service_area_progress()
        
        # Pre-compute building density
        self._compute_building_density()
        
        # Pre-compute FLW service area statistics
        self._compute_flw_service_area_stats()
        
        # Pre-compute FLW active dates from delivery units
        self._compute_additional_flw_active_dates_from_delivery_units()

        # Pre-compute travel distances
        self.calculate_travel_distances()

    def _compute_metadata_from_service_delivery_data(self):
        """Precompute metadata to avoid redundant processing, must be called after service delivery data is loaded"""
        self._compute_du_completion_dates()

    def _compute_service_area_progress(self):
        """Pre-compute service area progress statistics"""
        self.service_area_progress = {}
        
        for sa_id, service_area in self.service_areas.items():
            self.service_area_progress[sa_id] = {
                'total_dus': service_area.total_units,
                'completed_dus': service_area.completed_units,
                'percentage': service_area.completion_percentage
            }
    
    def _compute_building_density(self):
        """Pre-compute building density for each service area"""
        self.service_area_building_density = {}
        
        for sa_id, service_area in self.service_areas.items():
            self.service_area_building_density[sa_id] = service_area.building_density
    
    def _compute_additional_flw_active_dates_from_delivery_units(self):
        """Pre-compute FLW active dates from delivery units"""
        
        for du in self.delivery_units.values():
            # Update Active Dates for FLW if a DU was checked in but no service delivery was made that day
            if(du.checked_in_date):
                flw = self.flws[du.flw_commcare_id]
                # Convert checked_in_date from string to datetime object
                checked_in_date = pd.to_datetime(du.checked_in_date).date()
                
                # Skip if the date conversion resulted in NaT (Not a Time)
                if pd.isna(checked_in_date) == False:
                    if checked_in_date not in flw.dates_active:
                        flw.dates_active.append(checked_in_date)
                            
                    # Update first and last du checkin dates
                    if (flw.first_du_checkin is None or 
                        checked_in_date < flw.first_du_checkin):
                        flw.first_du_checkin = checked_in_date
                                
                    if (flw.last_du_checkin is None or 
                        checked_in_date > flw.last_du_checkin):
                        flw.last_du_checkin = checked_in_date

    def _compute_du_completion_dates(self):
        """Pre-compute DU completion dates"""
        
        for du in self.delivery_units.values():
            if du.status == 'completed':
                if du.delivery_count == 0:
                    # this will be None or NaN if the DU was closed manually
                    if pd.isna(du.checked_in_date) == False:
                        du.computed_du_completion_date = pd.to_datetime(du.checked_in_date).to_pydatetime()
                        print(f"DU {du.id}: Using checked_in_date -> {du.computed_du_completion_date}")
                    else:
                        print(f"DU {du.id}: DU is marked as completed but has no deliveries and no checked_in_date")
                else:
                    if(len(du.service_points) != du.delivery_count):
                        print(f"DU {du.id}: Service points count {len(du.service_points)} does not match delivery count {du.delivery_count}")
                    #loop through the service_points and find the earliest date of a service delivery point
                    for sp in du.service_points:
                        sp_date: datetime = pd.to_datetime(sp.visit_date).to_pydatetime()
                        if du.computed_du_completion_date is None or sp_date < du.computed_du_completion_date:
                            du.computed_du_completion_date = sp_date  
                    print(f"DU {du.id}: From {len(du.service_points)} service points -> {du.computed_du_completion_date}")
   
    def _compute_flw_service_area_stats(self):
        """Pre-compute FLW statistics per service area"""
        self.flw_service_area_stats = {}
        
        # Create a mapping of service_area -> flw -> stats
        temp_stats = {}
        
        for du in self.delivery_units.values():
            sa_id = du.service_area_id
            flw_commcare_id = du.flw_commcare_id
            
            if sa_id not in temp_stats:
                temp_stats[sa_id] = {}
            
            if flw_commcare_id not in temp_stats[sa_id]:
                temp_stats[sa_id][flw_commcare_id] = {
                    'total_dus': 0,
                    'completed_dus': 0,
                    'buildings': 0,
                    'surface_area': 0
                }
            
            # Update stats
            temp_stats[sa_id][flw_commcare_id]['total_dus'] += 1
            if du.status == 'completed':
                temp_stats[sa_id][flw_commcare_id]['completed_dus'] += 1
            temp_stats[sa_id][flw_commcare_id]['buildings'] += du.buildings
            temp_stats[sa_id][flw_commcare_id]['surface_area'] += du.surface_area


        # Calculate percentages and restructure for easier access
        for sa_id, flws in temp_stats.items():
            for flw_commcare_id, stats in flws.items():
                if flw_commcare_id not in self.flw_service_area_stats:
                    self.flw_service_area_stats[flw_commcare_id] = {}
                
                # Calculate completion percentage
                total = stats['total_dus']
                completed = stats['completed_dus']
                percentage = (completed / total * 100) if total > 0 else 0
                
                # Calculate building density (buildings per sq km)
                surface_area_sqkm = stats['surface_area'] / 1000000
                density = (stats['buildings'] / surface_area_sqkm) if surface_area_sqkm > 0 else 0
                
                # Store stats
                self.flw_service_area_stats[flw_commcare_id][sa_id] = {
                    'total_dus': total,
                    'completed_dus': completed,
                    'percentage': percentage,
                    'buildings': stats['buildings'],
                    'surface_area': stats['surface_area'],
                    'density': density
                }
    
    def get_status_counts_by_flw(self) -> pd.DataFrame:
        """
        Get a DataFrame with status counts per FLW
        
        Returns:
            DataFrame with columns flw_name, du_status, count
        """
        rows = []
        for flw_name, flw in self.flws.items():
            for status, count in flw.status_counts.items():
                rows.append({
                    'flw_name': flw_name,
                    'du_status': status,
                    'count': count
                })
        
        return pd.DataFrame(rows)
    
    def get_flw_completion_data(self) -> pd.DataFrame:
        """
        Get a DataFrame with FLW completion rates and assigned service areas
        
        Returns:
            DataFrame with columns flw_name, completed_units, assigned_units, completion_rate, service_areas
        """
        rows = []
        for flw_name, flw in self.flws.items():
            rows.append({
                'flw_name': flw_name,
                'completed_units': flw.completed_units,
                'assigned_units': flw.assigned_units,
                'completion_rate': flw.completion_rate,
                'service_areas': flw.get_service_areas_str()
            })
        
        return pd.DataFrame(rows)
    
    def get_service_area_progress(self) -> pd.DataFrame:
        """
        Get a DataFrame with service area progress 
        
        Returns:
            DataFrame with columns service_area_id, completed_dus, total_dus, percentage
        """
        rows = []
        for sa_id, stats in self.service_area_progress.items():
            rows.append({
                'service_area_id': sa_id,
                'completed_dus': stats['completed_dus'],
                'total_dus': stats['total_dus'],
                'percentage': stats['percentage']
            })
        
        return pd.DataFrame(rows)
    
    def get_flw_service_area_progress(self) -> pd.DataFrame:
        """
        Get a DataFrame with FLW progress per service area
        
        Returns:
            DataFrame with columns flw_name, service_area_id, completed_dus, total_dus, percentage
        """
        rows = []
        for flw_name, sa_stats in self.flw_service_area_stats.items():
            for sa_id, stats in sa_stats.items():
                rows.append({
                    'flw_name': flw_name,
                    'service_area_id': sa_id,
                    'completed_dus': stats['completed_dus'],
                    'total_dus': stats['total_dus'],
                    'percentage': stats['percentage']
                })
        
        return pd.DataFrame(rows)
    
    def get_building_density(self) -> pd.DataFrame:
        """
        Get a DataFrame with building density per service area
        
        Returns:
            DataFrame with columns service_area_id, #Buildings, Surface Area (sq. meters), density
        """
        rows = []
        for sa_id, service_area in self.service_areas.items():
            rows.append({
                'service_area_id': sa_id,
                '#Buildings': service_area.total_buildings,
                'Surface Area (sq. meters)': service_area.total_surface_area,
                'density': self.service_area_building_density[sa_id]
            })
        
        return pd.DataFrame(rows)
    
    def get_flw_building_density(self) -> pd.DataFrame:
        """
        Get a DataFrame with building density per FLW and service area
        
        Returns:
            DataFrame with columns flw_name, service_area_id, #Buildings, Surface Area (sq. meters), density
        """
        rows = []
        for flw_name, sa_stats in self.flw_service_area_stats.items():
            for sa_id, stats in sa_stats.items():
                rows.append({
                    'flw_name': flw_name,
                    'service_area_id': sa_id,
                    '#Buildings': stats['buildings'],
                    'Surface Area (sq. meters)': stats['surface_area'],
                    'density': stats['density']
                })
        
        return pd.DataFrame(rows)

    def calculate_travel_distances(self):
        """
        Calculate travel distances between centroids in each service area
        using the Traveling Salesman Problem (TSP) approach with nearest neighbor algorithm
        """
        # Skip if already calculated
        if self.travel_distances:
            return
            
        self.travel_distances = {}
        
        # Process each service area
        for sa_id, service_area in self.service_areas.items():
            # Get valid centroids
            centroids = []
            for du in service_area.delivery_units:
                if du.centroid is None:
                    continue
                    
                # Parse centroid data
                try:
                    if isinstance(du.centroid, str):
                        lat, lon = map(float, du.centroid.split())
                        centroids.append((lat, lon))
                    elif isinstance(du.centroid, (list, tuple)) and len(du.centroid) == 2:
                        centroids.append(tuple(du.centroid))
                except Exception:
                    continue
            
            # Skip if insufficient points
            if len(centroids) <= 1:
                self.travel_distances[sa_id] = 0
                continue
            
            try:
                # Calculate distance matrix
                n = len(centroids)
                dist_matrix = np.zeros((n, n))
                
                # Build distance matrix (upper triangle)
                for i in range(n):
                    for j in range(i+1, n):
                        try:
                            dist = geodesic(centroids[i], centroids[j]).kilometers
                            dist_matrix[i, j] = dist
                            dist_matrix[j, i] = dist
                        except Exception:
                            dist_matrix[i, j] = 0
                            dist_matrix[j, i] = 0
                
                # Apply nearest neighbor TSP algorithm
                current = 0  # Start at first point
                unvisited = set(range(1, n))
                total_distance = 0
                
                # Visit each point, always choosing the nearest
                while unvisited:
                    nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
                    total_distance += dist_matrix[current, nearest]
                    current = nearest
                    unvisited.remove(nearest)
                
                # Return to start to complete the circuit
                total_distance += dist_matrix[current, 0]
                
                # Store result
                self.travel_distances[sa_id] = total_distance
                
            except Exception:
                self.travel_distances[sa_id] = 0
    
    def get_travel_distances_by_flw(self) -> pd.DataFrame:
        """
        Get a DataFrame with travel distances by FLW
        
        Returns:
            DataFrame with columns flw_name, total_distance
        """
        # Calculate travel distances if not already done
        if not self.travel_distances:
            self.calculate_travel_distances()
        
        # Map FLWs to their service areas and calculate total distance
        flw_distances = {}
        for flw_name, flw in self.flws.items():
            total_distance = 0
            for sa_id in flw.service_areas:
                if sa_id in self.travel_distances:
                    total_distance += self.travel_distances[sa_id]
            flw_distances[flw_name] = total_distance
        
        # Convert to DataFrame
        rows = [{'flw_name': flw, 'total_distance': distance} for flw, distance in flw_distances.items()]
        return pd.DataFrame(rows)

    @classmethod
    def load_delivery_units_from_df(cls, delivery_units_df: pd.DataFrame) -> 'CoverageData':
        """
        Load coverage data from CommCare dataframe
        
        Args:
            delivery_units_df: DataFrame containing delivery units data from CommCare
        """
        # Import here to avoid circular imports
        from .delivery_unit import DeliveryUnit
        from .service_area import ServiceArea
        from .flw import FLW
        
        data = cls()
        
        # Store the processed dataframe
        data.delivery_units_df = delivery_units_df
        
        print(f"Processing {len(delivery_units_df)} delivery units from CommCare")
        
        # Clean and prepare the dataframe
        delivery_units_df = cls.clean_du_dataframe(delivery_units_df)
        
        created_count = 0
        
        # Create delivery units and service areas
        for idx, row in delivery_units_df.iterrows():
            row_dict = row.to_dict()
            
            # Skip entries without WKT
            # if pd.isna(row_dict.get('WKT')) or str(row_dict.get('WKT', '')).strip() == '':
                # continue
            
            # Create delivery unit
            du = DeliveryUnit.from_dict(row_dict)
            data.delivery_units[du.du_name] = du
            # print(f"DU id: {du.id}" + " name: " + du.du_name)
            
            created_count += 1
                
            # Create or update service area
            sa_id = du.service_area_id
            if sa_id not in data.service_areas:
                data.service_areas[sa_id] = ServiceArea(id=sa_id)
            data.service_areas[sa_id].delivery_units.append(du)
                
            # Create or update FLW
            if du.flw_commcare_id not in data.flws:
                data.flws[du.flw_commcare_id] = FLW(id=du.flw_commcare_id, name=du.flw_commcare_id)
                
            # Update FLW's assigned units and service areas
            flw = data.flws[du.flw_commcare_id]
            flw.assigned_units += 1
            if du.status == 'completed':
                flw.completed_units += 1
            if sa_id not in flw.service_areas:
                flw.service_areas.append(sa_id)
            # Add this delivery unit to the FLW's delivery_units list
            flw.delivery_units.append(du)
        
        print(f"Successfully created {created_count} delivery units across {len(data.service_areas)} service areas")
        
        # Precompute metadata
        data._compute_metadata_from_delivery_unit_data()
        
        return data
    
    @classmethod
    def clean_du_dataframe(cls, delivery_units_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare delivery units dataframe
        
        Args:
            delivery_units_df: DataFrame containing delivery units data that were loaded from a CommCare export or API call.
            This will fix mismatched column headings and make sure data is typed correctly.
            This will chnage the value of the service_area_id to be a combination of the oa_id and sa_id
            This will replace "---", the null value in a CommCare export, with None
            This will manipulate the dataframe in place and return it.
        
        Returns:
            Cleaned and prepared DataFrame
        """
        # Map column names
        delivery_units_df.rename(columns={
            'caseid': 'case_id',
            'oa': 'oa_id',
            'service_area_number': 'service_area_id',
            'owner_id': 'flw_commcare_id',  # Ensure owner_id is mapped to flw_commcare_id
            'number' : 'du_number',
            'name' : 'du_name', # xlsx column name
            'case_name': 'du_name', # API column name
            '#Buildings' : 'buildings', #API column name
            'Surface Area (sq. meters)' : 'surface_area', #API column name
            'last_modified': 'last_modified_date',
        }, inplace=True)

        # Replace all "---" values with None throughout the entire dataframe,"---" is the null value in a CommCare export
        delivery_units_df = delivery_units_df.replace("---", None)

        # Check for several known columns
        required_columns = ['case_id', 'du_name', 'service_area', 'flw_commcare_id', 'WKT']
        missing_columns = [col for col in required_columns if col not in delivery_units_df.columns]
        if missing_columns:
            found_columns = list(delivery_units_df.columns)
            error_msg = "Required column(s) not found in CommCare data: " + ", ".join(missing_columns) + "\nColumns found: " + ", ".join(found_columns)
            raise ValueError(error_msg)
        
        # Drop rows with missing service area IDs (value = None), count them first for debugging
        dropped_data_count = len(delivery_units_df)
        delivery_units_df.drop(delivery_units_df[(delivery_units_df['service_area_id'].isna())].index, inplace=True)
        dropped_data_count = dropped_data_count - len(delivery_units_df)

        # Create new service_area_id that combines OA and SA
        delivery_units_df['service_area_id'] = delivery_units_df['oa_id'].astype(str) + '-' + delivery_units_df['service_area_id'].astype(str)

        # Print distinct service area counts for debugging
        distinct_service_areas = delivery_units_df['service_area_id'].nunique()
        print(f"Found {distinct_service_areas} distinct service areas in the data. Dropping {dropped_data_count} rows with missing service area ID (likely test data).")
    

        
        return delivery_units_df
    
    def load_service_delivery_dataframe_from_csv(self, service_delivery_csv: str) -> None:
        """
        Load service delivery points from a CSV file
            
        Args:
            service_delivery_csv: Path to service delivery GPS coordinates CSV
        """
        if not service_delivery_csv or not pd.io.common.file_exists(service_delivery_csv):
            print(f"Service delivery CSV file not found: {service_delivery_csv}")
            return
            
        service_df = pd.read_csv(service_delivery_csv)
        return service_df
    
    def load_service_delivery_from_datafame(self, service_df: pd.DataFrame) -> None:
        """
        Load service delivery points from a dataframe

        Args:
            service_df: DataFrame containing service delivery GPS coordinates
        """
        # Import here to avoid circular imports
        from .service_delivery_point import ServiceDeliveryPoint
        
        print(f"Loading service delivery data: {len(service_df)} points in DataFrame")
        
        # Create service delivery points
        for _, row in service_df.iterrows():
            row_dict = row.to_dict()
            
            point = ServiceDeliveryPoint.from_dict(row_dict)
            self.service_points.append(point)
            
            # Associate service point with delivery unit based on du_name
            du = self.delivery_units.get(point.du_name)
            if du:
                du.service_points.append(point)
            else:
                raise ValueError(f"Delivery unit not found for service point: {point.du_name}")
            
            # Add to the FLW CommCare ID to Name mapping if both values are present
            if point.flw_commcare_id and point.flw_name:
                self.flw_commcare_id_to_name_map[point.flw_commcare_id] = point.flw_name
                
                # If this FLW exists in our FLW list, update its name too
                if point.flw_commcare_id in self.flws:
                    self.flws[point.flw_commcare_id].name = point.flw_name
                    # Add the service point to this FLW's service_points list
                    self.flws[point.flw_commcare_id].service_points.append(point)
            
            # Update FLW's active dates if visit_date is present
            if point.visit_date and (point.flw_id in self.flws or point.flw_commcare_id in self.flws):
                visit_date = pd.to_datetime(point.visit_date).date()
                flw_id = point.flw_commcare_id if point.flw_commcare_id in self.flws else point.flw_id
                
                if flw_id in self.flws:
                    if visit_date not in self.flws[flw_id].dates_active:
                        self.flws[flw_id].dates_active.append(visit_date)
                    
                    # Update first and last service delivery dates
                    if (self.flws[flw_id].first_service_delivery_date is None or 
                        visit_date < self.flws[flw_id].first_service_delivery_date):
                        self.flws[flw_id].first_service_delivery_date = visit_date
                        
                    if (self.flws[flw_id].last_service_delivery_date is None or 
                        visit_date > self.flws[flw_id].last_service_delivery_date):
                        self.flws[flw_id].last_service_delivery_date = visit_date
                
        self._compute_metadata_from_service_delivery_data()

        # Set the opportunity name
        self.opportunity_name = service_df.iloc[0]['opportunity_name']

    def create_delivery_units_geodataframe(self) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from delivery units for mapping"""
        delivery_units_data = []
        valid_geoms = 0
        invalid_geoms = 0
        
        print(f"Creating GeoDataFrame from {len(self.delivery_units)} delivery units")
        
        for du_id, du in self.delivery_units.items():
            try:
                # Try to create a geometry object
                geometry = du.geometry
                valid_geoms += 1
                
                # Create a dictionary with all relevant properties
                # Convert Timestamp to string to ensure JSON serialization works
                last_modified_str = None
                if du.last_modified_date is not None:
                    try:
                        last_modified_str = du.last_modified_date.isoformat()
                    except:
                        # If conversion fails, use string representation
                        last_modified_str = str(du.last_modified_date)
                
                du_dict = {
                    'du_id': du.id,
                    'name': du.du_name,
                    'service_area_id': du.service_area_id,
                    'flw_commcare_id': du.flw_commcare_id,
                    'du_status': du.status,
                    'WKT': du.wkt,
                    '#Buildings': du.buildings,
                    'Surface Area (sq. meters)': du.surface_area,
                    'delivery_count': du.delivery_count,
                    'delivery_target': du.delivery_target,
                    'du_checkout_remark': du.du_checkout_remark,
                    'checked_out_date': du.checked_out_date,
                    'last_modified_date': last_modified_str,
                    'geometry': geometry  # This is a Shapely geometry object
                }
                delivery_units_data.append(du_dict)
            except Exception as e:
                print(f"Error creating geometry for delivery unit {du_id}: {e}")
                print(f"WKT string: {du.wkt[:100]}..." if len(du.wkt) > 100 else du.wkt)
                invalid_geoms += 1
        
        if not delivery_units_data:
            print("WARNING: No valid delivery units with geometries found")
            # Return an empty GeoDataFrame with the right structure
            return gpd.GeoDataFrame(columns=['du_id', 'name', 'service_area_id', 'flw_commcare_id', 'du_status', 
                                             'WKT', '#Buildings', 'Surface Area (sq. meters)', 
                                             'delivery_count', 'delivery_target', 'du_checkout_remark', 
                                             'checked_out_date', 'last_modified_date', 'geometry'], 
                                   geometry='geometry', crs="EPSG:4326")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(delivery_units_data, geometry='geometry')
        
        # Ensure the CRS is set to WGS84 (standard for web maps)
        gdf.crs = "EPSG:4326"
        
        return gdf
    
    def create_service_points_geodataframe(self) -> Optional[gpd.GeoDataFrame]:
        """Create a GeoDataFrame from service points for mapping"""
        if not self.service_points:
            return None
            
        service_points_data = []
        for point in self.service_points:
            # Create a dictionary with all relevant properties
            point_dict = {
                'visit_id': point.id,
                'flw_id': point.flw_id,
                'flw_name': point.flw_name,
                'flw_commcare_id': point.flw_commcare_id,
                'visit_date': point.visit_date,
                'accuracy_in_m': point.accuracy_in_m,
                'status': point.status,
                'du_name': point.du_name,
                'flagged': point.flagged,
                'flag_reason': point.flag_reason,
                'geometry': point.geometry  # This is a Shapely Point object
            }
            service_points_data.append(point_dict)
        
        service_points_gdf = gpd.GeoDataFrame(service_points_data, geometry='geometry', crs="EPSG:4326")
        return service_points_gdf 

    def create_service_points_dataframe(self) -> Optional[pd.DataFrame]:
        """Create a DataFrame from service points for statistics"""
        if not self.service_points:
            return None
            
        service_df = pd.DataFrame([{
            'visit_id': point.id,
            'flw_id': point.flw_id,
            'flw_name': point.flw_name,
            'flw_commcare_id': point.flw_commcare_id,
            'lattitude': point.latitude,
            'longitude': point.longitude,
            'visit_date': point.visit_date,
            'accuracy_in_m': point.accuracy_in_m,
            'status': point.status,
            'du_name': point.du_name,
            'flagged': point.flagged,
            'flag_reason': point.flag_reason
        } for point in self.service_points])
        
        return service_df 

    def get_completed_du_heatmap_data(self, flw_filter=None, date_start=None, date_end=None):
        """
        Returns a heatmap matrix of completed DUs per FLW per day.
        - flw_filter: optional list of FLW IDs to include
        - date_start, date_end: optional date strings (YYYY-MM-DD) to filter date range
        Returns a dict: { 'flws': [...], 'flw_names': [...], 'dates': [...], 'matrix': [[count,...], ...] }
        """
        # Collect relevant DUs
        du_list = [du for du in self.delivery_units.values() if du.status == 'completed' and du.last_modified_date]
        
        # Convert last_modified_date to date string (YYYY-MM-DD)
        data = []
        for du in du_list:
            try:
                date_str = du.last_modified_date.strftime('%Y-%m-%d')
            except Exception:
                continue
                
            # For each delivery unit, we need to find the corresponding service point to get the CommCare ID
            # First assume we can't find a match
            flw_name = None
            
            # Look for a matching service point with the same flw_id
            for sp in self.service_points:
                if sp.flw_id == du.flw_commcare_id and sp.flw_commcare_id:
                    # If we find a match, use the CommCare ID to look up the name
                    flw_name = self.flw_commcare_id_to_name_map.get(sp.flw_commcare_id)
                    break
            
            # If we couldn't find a match through service points, fall back to direct lookup
            if flw_name is None:
                # This is a fallback in case we can't find the CommCare ID
                flw_name = du.flw_commcare_id
            
            data.append({'flw_id': du.flw_commcare_id, 'flw_name': flw_name, 'date': date_str})
        
        df = pd.DataFrame(data)
        # Determine FLWs and dates to use for axes
        if flw_filter is not None:
            flw_ids = sorted(list(flw_filter))
        else:
            flw_ids = sorted(df['flw_id'].unique().tolist()) if not df.empty else []
        
        # Get display names for the FLWs
        flw_names = []
        for flw_id in flw_ids:
            # Try to find a matching service point to get the CommCare ID
            match_found = False
            for sp in self.service_points:
                if sp.flw_commcare_id == flw_id:
                    flw_names.append(self.flw_commcare_id_to_name_map.get(sp.flw_commcare_id, flw_id))
                    match_found = True
                    break
            
            # If no match found, use the flw_id directly
            if not match_found:
                flw_names.append(flw_id)
        
        # Build date range
        if date_start is not None and date_end is not None:
            try:
                date_range = pd.date_range(start=date_start, end=date_end)
                dates = [d.strftime('%Y-%m-%d') for d in date_range]
            except Exception:
                dates = []
        else:
            dates = sorted(df['date'].unique().tolist()) if not df.empty else []
        
        # If no FLWs or dates, return empty
        if not flw_ids or not dates:
            return {'flws': flw_ids, 'flw_names': flw_names, 'dates': dates, 'matrix': []}
        
        # Filter df to only selected FLWs and dates
        if not df.empty:
            df = df[df['flw_id'].isin(flw_ids) & df['date'].isin(dates)]
        
        # Build matrix: rows=FLWs, cols=dates, fill zeros for missing
        matrix = []
        for flw_id in flw_ids:
            row = []
            for date in dates:
                if not df.empty:
                    count = df[(df['flw_id'] == flw_id) & (df['date'] == date)].shape[0]
                else:
                    count = 0
                row.append(count)
            matrix.append(row)
        
        return {
            'flws': flw_ids,  # FLW IDs for internal use
            'flw_names': flw_names,  # Display names for UI
            'dates': dates,
            'matrix': matrix
        }

    def get_gps_accuracy_heatmap_data(self, flw_filter=None, date_start=None, date_end=None):
        """
        Returns a heatmap matrix of mean GPS accuracy per FLW per day.
        - flw_filter: optional list of FLW IDs to include
        - date_start, date_end: optional date strings (YYYY-MM-DD) to filter date range
        Returns a dict: { 'flws': [...], 'flw_names': [...], 'dates': [...], 'matrix': [[mean_accuracy,...], ...] }
        """
        # Collect service points with GPS accuracy data
        data = []
        for sp in self.service_points:
            if sp.accuracy_in_m is not None and sp.visit_date:
                try:
                    # Parse visit_date and convert to date string
                    visit_date = pd.to_datetime(sp.visit_date)
                    if pd.isna(visit_date):
                        continue
                    date_str = visit_date.strftime('%Y-%m-%d')
                    
                    # Get FLW name for display
                    flw_name = self.flw_commcare_id_to_name_map.get(sp.flw_commcare_id, sp.flw_id)
                    
                    data.append({
                        'flw_id': sp.flw_commcare_id or sp.flw_id,
                        'flw_name': flw_name,
                        'date': date_str,
                        'accuracy_in_m': sp.accuracy_in_m
                    })
                except Exception:
                    continue
        
        df = pd.DataFrame(data)
        
        # Determine FLWs and dates to use for axes
        if flw_filter is not None:
            flw_ids = sorted(list(flw_filter))
        else:
            flw_ids = sorted(df['flw_id'].unique().tolist()) if not df.empty else []
        
        # Get display names for the FLWs
        flw_names = []
        for flw_id in flw_ids:
            # Find the display name for this FLW
            flw_name = self.flw_commcare_id_to_name_map.get(flw_id, flw_id)
            flw_names.append(flw_name)
        
        # Build date range
        if date_start is not None and date_end is not None:
            try:
                date_range = pd.date_range(start=date_start, end=date_end)
                dates = [d.strftime('%Y-%m-%d') for d in date_range]
            except Exception:
                dates = []
        else:
            dates = sorted(df['date'].unique().tolist()) if not df.empty else []
        
        # If no FLWs or dates, return empty
        if not flw_ids or not dates:
            return {'flws': flw_ids, 'flw_names': flw_names, 'dates': dates, 'matrix': []}
        
        # Filter df to only selected FLWs and dates
        if not df.empty:
            df = df[df['flw_id'].isin(flw_ids) & df['date'].isin(dates)]
        
        # Build matrix: rows=FLWs, cols=dates, calculate mean accuracy for each day
        matrix = []
        for flw_id in flw_ids:
            row = []
            for date in dates:
                if not df.empty:
                    day_data = df[(df['flw_id'] == flw_id) & (df['date'] == date)]
                    if len(day_data) > 0:
                        mean_accuracy = day_data['accuracy_in_m'].mean()
                        # Round to 1 decimal place for cleaner display
                        row.append(round(mean_accuracy, 1))
                    else:
                        row.append(None)  # No data for this day
                else:
                    row.append(None)
            matrix.append(row)
        
        return {
            'flws': flw_ids,  # FLW IDs for internal use
            'flw_names': flw_names,  # Display names for UI
            'dates': dates,
            'matrix': matrix
        } 