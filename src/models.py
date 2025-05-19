from shapely import wkt
from shapely.geometry import Polygon, Point
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import numpy as np
import time


@dataclass
class FLW:
    """Field Level Worker (FLW) Model"""
    id: str
    name: str
    service_areas: List[str] = field(default_factory=list)
    assigned_units: int = 0
    completed_units: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as a percentage"""
        if self.assigned_units == 0:
            return 0.0
        return (self.completed_units / self.assigned_units) * 100

    def get_service_areas_str(self) -> str:
        """Returns service areas as a comma-separated string"""
        return ', '.join(str(sa) for sa in sorted(self.service_areas))


@dataclass
class ServiceDeliveryPoint:
    """Service Delivery Point Model"""
    id: str
    latitude: float
    longitude: float
    flw_id: str
    flw_name: str
    flw_commcare_id: Optional[str] = None
    status: Optional[str] = None
    du_name: Optional[str] = None
    flagged: Optional[bool] = None
    flag_reason: Optional[str] = None
    visit_date: Optional[str] = None
    accuracy_in_m: Optional[float] = None
    
    @property
    def geometry(self) -> Point:
        """Get point geometry"""
        return Point(self.longitude, self.latitude)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceDeliveryPoint':
        """Create a ServiceDeliveryPoint from a dictionary"""
        return cls(
            id=str(data.get('visit_id', '')),
            latitude=float(data.get('lattitude', 0.0)),  # Note: using 'lattitude' as in original code
            longitude=float(data.get('longitude', 0.0)),
            flw_id=str(data.get('flw_id', '')),
            flw_name=str(data.get('flw_name', '')),
            flw_commcare_id=data.get('cchq_user_owner_id'),
            status=data.get('status'),
            du_name=data.get('du_name'),
            flagged=bool(data.get('flagged')) if data.get('flagged') is not None else None,
            flag_reason=data.get('flag_reason'),
            visit_date=data.get('visit_date'),
            accuracy_in_m=float(data.get('accuracy_in_m', 0.0)) if data.get('accuracy_in_m') else None
        )


@dataclass
class DeliveryUnit:
    """Delivery Unit Model"""
    id: str
    name: str
    service_area_id: str
    flw_commcare_id: str  # FLW/owner CommCare ID
    status: str  # completed, visited, unvisited represented as "---"
    wkt: str
    buildings: int = 0
    surface_area: float = 0.0
    delivery_count: int = 0
    delivery_target: int = 0
    du_checkout_remark: str = "---"
    checked_out_date: str = "---"
    centroid: Optional[tuple] = None
    last_modified_date: Optional[datetime] = None
    
    @property
    def geometry(self) -> Polygon:
        """Convert WKT to Shapely geometry"""
        try:
            if not self.wkt or self.wkt == '':
                raise ValueError(f"Empty WKT string for delivery unit {self.id}")
            return wkt.loads(self.wkt)
        except Exception as e:
            raise ValueError(f"Invalid WKT for delivery unit {self.id}: {str(e)}")
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.delivery_target == 0:
            return 0.0
        return (self.delivery_count / self.delivery_target) * 100
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeliveryUnit':
        """Create a DeliveryUnit from a dictionary"""
        # Extract required fields with defaults
        du_id = str(data.get('du_id', data.get('caseid', '')))
        
        # Handle potentially missing or NaN values
        name_val = data.get('name', '')
        name = str(name_val) if not pd.isna(name_val) else ''
        
        # Better handling for service_area_id, with multiple fallbacks
        service_area_id = ''
        for key in ['service_area_id', 'service_area', 'service_area_number']:
            if key in data and data.get(key) and str(data.get(key)).strip():
                service_area_id = str(data.get(key))
                break
        
        # If still empty, use a default value
        if not service_area_id:
            service_area_id = "UNKNOWN"
        
        # Look for flw_commcare_id first (as it might have been renamed), then fall back to original column names
        flw_val = data.get('flw_commcare_id', data.get('owner_id', data.get('flw', '')))
        flw = str(flw_val) if not pd.isna(flw_val) else ''
        
        status_val = data.get('du_status', 'unvisited')
        status = str(status_val).lower() if not pd.isna(status_val) else 'unvisited'
        
        wkt_str = str(data.get('WKT', '')) if not pd.isna(data.get('WKT', '')) else ''
        
        # Check if WKT is empty
        if not wkt_str or wkt_str == '':
            raise ValueError(f"Empty WKT for delivery unit {du_id}, name: {name}")
        
        # Extract numeric fields with proper error handling
        try:
            buildings = int(data.get('buildings', data.get('#Buildings', 0)))
        except (ValueError, TypeError):
            buildings = 0
        
        try:
            surface_area = float(data.get('surface_area', data.get('Surface Area (sq. meters)', 0.0)))
        except (ValueError, TypeError):
            surface_area = 0.0
        
        try:
            delivery_count = int(data.get('delivery_count', 0))
        except (ValueError, TypeError):
            delivery_count = 0
        
        try:
            delivery_target = int(data.get('delivery_target', 0))
        except (ValueError, TypeError):
            delivery_target = 0
        
        # Handle string fields
        checkout_remark_val = data.get('du_checkout_remark', '---')
        checkout_remark = str(checkout_remark_val) if not pd.isna(checkout_remark_val) else '---'
        
        checkout_date_val = data.get('checked_out_date', '---')
        checkout_date = str(checkout_date_val) if not pd.isna(checkout_date_val) else '---'
        
        # Parse last_modified_date if available
        last_modified = None
        if 'last_modified_date' in data and data['last_modified_date']:
            try:
                if isinstance(data['last_modified_date'], datetime):
                    last_modified = data['last_modified_date']
                else:
                    # Try parsing the date string
                    last_modified = pd.to_datetime(data['last_modified_date'])
            except Exception:
                # If parsing fails, leave as None
                pass
        
        # Create the DeliveryUnit instance
        return cls(
            id=du_id,
            name=name,
            service_area_id=service_area_id,
            flw_commcare_id=flw,
            status=status,
            wkt=wkt_str,
            buildings=buildings,
            surface_area=surface_area,
            delivery_count=delivery_count,
            delivery_target=delivery_target,
            du_checkout_remark=checkout_remark,
            checked_out_date=checkout_date,
            centroid=data.get('centroid'),
            last_modified_date=last_modified
        )


@dataclass
class ServiceArea:
    """Service Area Model"""
    id: str
    delivery_units: List[DeliveryUnit] = field(default_factory=list)
    
    @property
    def total_buildings(self) -> int:
        """Calculate total buildings in the service area"""
        return sum(du.buildings for du in self.delivery_units)
    
    @property
    def total_surface_area(self) -> float:
        """Calculate total surface area in the service area"""
        return sum(du.surface_area for du in self.delivery_units)
    
    @property
    def total_units(self) -> int:
        """Get total number of delivery units"""
        return len(self.delivery_units)
    
    @property
    def completed_units(self) -> int:
        """Get number of completed delivery units"""
        return sum(1 for du in self.delivery_units if du.status == 'completed')
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_units == 0:
            return 0.0
        return (self.completed_units / self.total_units) * 100
    
    @property
    def assigned_flws(self) -> List[str]:
        """Get list of FLWs assigned to this service area"""
        return list(set(du.flw_commcare_id for du in self.delivery_units))
    
    @property
    def total_deliveries(self) -> int:
        """Get the total number of service deliveries in this service area"""
        return sum(du.delivery_count for du in self.delivery_units)
    
    @property
    def building_density(self) -> float:
        """Calculate building density (buildings per sq km)"""
        if self.total_surface_area == 0:
            return 0.0
        return self.total_buildings / (self.total_surface_area / 1000000)


class CoverageData:
    """Data container for coverage analysis"""
    
    def __init__(self):
        self.service_areas: Dict[str, ServiceArea] = {}
        self.delivery_units: Dict[str, DeliveryUnit] = {}
        self.service_points: List[ServiceDeliveryPoint] = []
        self.flws: Dict[str, FLW] = {}
        self.delivery_units_df: Optional[pd.DataFrame] = None
        
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
        return sum(1 for du in self.delivery_units.values() if du.status == '---')
    
    @property
    def completion_percentage(self) -> float:
        """Get the overall completion percentage"""
        if self.total_delivery_units == 0:
            return 0.0
        return (self.total_completed_dus / self.total_delivery_units) * 100
    
    def _compute_metadata(self):
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
        
        # Pre-compute travel distances
        self.calculate_travel_distances()
    
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
    
    def _compute_flw_service_area_stats(self):
        """Pre-compute FLW statistics per service area"""
        self.flw_service_area_stats = {}
        
        # Create a mapping of service_area -> flw -> stats
        temp_stats = {}
        
        for du in self.delivery_units.values():
            sa_id = du.service_area_id
            flw_name = du.flw_commcare_id
            
            if sa_id not in temp_stats:
                temp_stats[sa_id] = {}
            
            if flw_name not in temp_stats[sa_id]:
                temp_stats[sa_id][flw_name] = {
                    'total_dus': 0,
                    'completed_dus': 0,
                    'buildings': 0,
                    'surface_area': 0
                }
            
            # Update stats
            temp_stats[sa_id][flw_name]['total_dus'] += 1
            if du.status == 'completed':
                temp_stats[sa_id][flw_name]['completed_dus'] += 1
            temp_stats[sa_id][flw_name]['buildings'] += du.buildings
            temp_stats[sa_id][flw_name]['surface_area'] += du.surface_area
        
        # Calculate percentages and restructure for easier access
        for sa_id, flws in temp_stats.items():
            for flw_name, stats in flws.items():
                if flw_name not in self.flw_service_area_stats:
                    self.flw_service_area_stats[flw_name] = {}
                
                # Calculate completion percentage
                total = stats['total_dus']
                completed = stats['completed_dus']
                percentage = (completed / total * 100) if total > 0 else 0
                
                # Calculate building density (buildings per sq km)
                surface_area_sqkm = stats['surface_area'] / 1000000
                density = (stats['buildings'] / surface_area_sqkm) if surface_area_sqkm > 0 else 0
                
                # Store stats
                self.flw_service_area_stats[flw_name][sa_id] = {
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
            DataFrame with columns flw, du_status, count
        """
        rows = []
        for flw_name, flw in self.flws.items():
            for status, count in flw.status_counts.items():
                rows.append({
                    'flw': flw_name,
                    'du_status': status,
                    'count': count
                })
        
        return pd.DataFrame(rows)
    
    def get_flw_completion_data(self) -> pd.DataFrame:
        """
        Get a DataFrame with FLW completion rates and assigned service areas
        
        Returns:
            DataFrame with columns flw, completed_units, assigned_units, completion_rate, service_areas
        """
        rows = []
        for flw_name, flw in self.flws.items():
            rows.append({
                'flw': flw_name,
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
            DataFrame with columns flw, service_area_id, completed_dus, total_dus, percentage
        """
        rows = []
        for flw_name, sa_stats in self.flw_service_area_stats.items():
            for sa_id, stats in sa_stats.items():
                rows.append({
                    'flw': flw_name,
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
            DataFrame with columns flw, service_area_id, #Buildings, Surface Area (sq. meters), density
        """
        rows = []
        for flw_name, sa_stats in self.flw_service_area_stats.items():
            for sa_id, stats in sa_stats.items():
                rows.append({
                    'flw': flw_name,
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
            DataFrame with columns flw, total_distance
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
        rows = [{'flw': flw, 'total_distance': distance} for flw, distance in flw_distances.items()]
        return pd.DataFrame(rows)

    @classmethod
    def load_delivery_units_from_df(cls, delivery_units_df: pd.DataFrame) -> 'CoverageData':
        """
        Load coverage data from CommCare dataframe
        
        Args:
            delivery_units_df: DataFrame containing delivery units data from CommCare
        """
        data = cls()
        
        # Store the processed dataframe
        data.delivery_units_df = delivery_units_df
        
        print(f"Processing {len(delivery_units_df)} delivery units from CommCare")
        
        # Process the dataframe similar to from_excel_and_csv
        # Ensure required columns exist to avoid KeyError
        required_columns = ['caseid', 'name', 'service_area', 'owner_id', 'WKT']
        missing_columns = [col for col in required_columns if col not in delivery_units_df.columns]
        if missing_columns:
            error_msg = "Required column(s) not found in CommCare data: " + ", ".join(missing_columns)
            raise ValueError(error_msg)
        
        # Convert columns to appropriate types
        try:
            delivery_units_df['buildings'] = pd.to_numeric(delivery_units_df['buildings'], errors='coerce').fillna(0).astype(int)
        except KeyError:
            print("Column 'buildings' not found, setting to default 0")
            delivery_units_df['buildings'] = 0
        
        try:
            delivery_units_df['delivery_count'] = pd.to_numeric(delivery_units_df['delivery_count'], errors='coerce').fillna(0).astype(int)
        except KeyError:
            print("Column 'delivery_count' not found, setting to default 0")
            delivery_units_df['delivery_count'] = 0
        
        try:
            delivery_units_df['delivery_target'] = pd.to_numeric(delivery_units_df['delivery_target'], errors='coerce').fillna(0).astype(int)
        except KeyError:
            print("Column 'delivery_target' not found, setting to default 0")
            delivery_units_df['delivery_target'] = 0
        
        try:
            delivery_units_df['surface_area'] = pd.to_numeric(delivery_units_df['surface_area'], errors='coerce').fillna(0)
        except KeyError:
            print("Column 'surface_area' not found, setting to default 0")
            delivery_units_df['surface_area'] = 0
        
        # Map column names
        delivery_units_df.rename(columns={
            'service_area_number': 'service_area_id',
            'buildings': '#Buildings',
            'surface_area': 'Surface Area (sq. meters)',
            'owner_id': 'flw_commcare_id',  # Ensure owner_id is mapped to flw_commcare_id
            'flw': 'flw_commcare_id'  # Also map flw to flw_commcare_id if it exists
        }, inplace=True)
            
        # Ensure service_area_id is string type for existing values
        delivery_units_df['service_area_id'] = delivery_units_df['service_area_id'].fillna('')
        delivery_units_df['service_area_id'] = delivery_units_df['service_area_id'].astype(str)
        
        # Ensure flw_commcare_id exists and is properly formatted
        if 'flw_commcare_id' not in delivery_units_df.columns:
            if 'owner_id' in delivery_units_df.columns:
                delivery_units_df['flw_commcare_id'] = delivery_units_df['owner_id']
            elif 'flw' in delivery_units_df.columns:
                delivery_units_df['flw_commcare_id'] = delivery_units_df['flw']
            else:
                print("WARNING: No FLW identifier column found in data")
                delivery_units_df['flw_commcare_id'] = 'UNKNOWN'
        
        # Ensure flw_commcare_id is string type
        delivery_units_df['flw_commcare_id'] = delivery_units_df['flw_commcare_id'].fillna('').astype(str)
        
        # Count rows with missing service area IDs (likely test data)
        test_data_count = (delivery_units_df['service_area_id'].isna() | (delivery_units_df['service_area_id'] == "---")).sum()
        
        # Filter out rows with missing service area IDs or with value "---"
        initial_row_count = len(delivery_units_df)
        delivery_units_df = delivery_units_df[(delivery_units_df['service_area_id'].notna()) & (delivery_units_df['service_area_id'] != "---")].copy()
        
        # Print distinct service area counts for debugging
        distinct_service_areas = delivery_units_df['service_area_id'].nunique()
        print(f"Found {distinct_service_areas} distinct service areas in the data. Skipped {test_data_count} rows with missing service area ID (likely test data).")
        
        # Make sure all key columns are strings to avoid issues
        str_columns = ['caseid', 'name', 'service_area', 'du_status', 'flw_commcare_id']
        for col in str_columns:
            if col in delivery_units_df.columns:
                # Fixed approach: First convert to object type, then to string
                delivery_units_df[col] = delivery_units_df[col].fillna('').astype(object).astype(str)
                
        # Make sure the status values are lowercase for consistency
        delivery_units_df['du_status'] = delivery_units_df['du_status'].fillna('').astype(object).astype(str).str.lower()
        
        created_count = 0
        
        # Create delivery units and service areas
        for idx, row in delivery_units_df.iterrows():
            row_dict = row.to_dict()
            
            # Skip entries without WKT
            if pd.isna(row_dict.get('WKT')) or str(row_dict.get('WKT', '')).strip() == '':
                continue
            
            # Make sure caseid is used as the id
            row_dict['du_id'] = row_dict.get('caseid', f"row_{idx}")
            
            # Create delivery unit
            try:
                du = DeliveryUnit.from_dict(row_dict)
                data.delivery_units[du.id] = du
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
            
            except Exception as e:
                continue
        
        print(f"Successfully created {created_count} delivery units across {len(data.service_areas)} service areas")
        
        # Precompute metadata
        data._compute_metadata()
        
        return data
    
    def load_service_delivery_from_csv(self, service_delivery_csv: str) -> None:
        """
        Load service delivery points from a CSV file
        
        Args:
            service_delivery_csv: Path to service delivery GPS coordinates CSV
        """
        if not service_delivery_csv or not pd.io.common.file_exists(service_delivery_csv):
            print(f"Service delivery CSV file not found: {service_delivery_csv}")
            return
            
        service_df = pd.read_csv(service_delivery_csv)
        print(f"Loaded service delivery data: {len(service_df)} points")
        
        # Create service delivery points
        for _, row in service_df.iterrows():
            row_dict = row.to_dict()
            try:
                point = ServiceDeliveryPoint.from_dict(row_dict)
                self.service_points.append(point)
                
                # Add to the FLW CommCare ID to Name mapping if both values are present
                if point.flw_commcare_id and point.flw_name:
                    self.flw_commcare_id_to_name_map[point.flw_commcare_id] = point.flw_name
                    
                    # If this FLW exists in our FLW list, update its name too
                    if point.flw_id in self.flws:
                        self.flws[point.flw_id].name = point.flw_name
            except Exception as e:
                print(f"Error creating service delivery point: {e}")
                continue
        
        print(f"DEBUG: Final flw_commcare_id_to_name_map contents: {self.flw_commcare_id_to_name_map}")
        return service_df

    @classmethod
    def from_excel_and_csv(cls, excel_file: str, service_delivery_csv: Optional[str] = None) -> 'CoverageData':
        """
        Load coverage data from Excel and CSV files
        
        Args:
            excel_file: Path to the DU Export Excel file
            service_delivery_csv: Optional path to service delivery GPS coordinates CSV
        """
        data = cls()
        
        # Read the Excel file
        print(f"Loading Excel file: {excel_file}")
        delivery_units_df = pd.read_excel(excel_file, sheet_name="Cases")
        print(f"Excel file loaded, shape: {delivery_units_df.shape}")
        
        # Use the new from_commcare method to process the dataframe
        data = cls.load_delivery_units_from_df(delivery_units_df)
        
        # Load service delivery data if provided
        if service_delivery_csv:
            data.load_service_delivery_from_csv(service_delivery_csv)
        
        return data

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
                    'name': du.name,
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
        
        print(f"DEBUG: flw_commcare_id_to_name_map in heatmap data: {self.flw_commcare_id_to_name_map}")
        
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
            
            print(f"DEBUG: Looking up FLW name for ID {du.flw_commcare_id} -> got {flw_name}")
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
        
        print(f"DEBUG: Final flw_ids: {flw_ids}")
        print(f"DEBUG: Final flw_names: {flw_names}")
        
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