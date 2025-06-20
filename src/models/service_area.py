from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Optional
import numpy as np
from geopy.distance import geodesic
from datetime import datetime

if TYPE_CHECKING:
    from .delivery_unit import DeliveryUnit


@dataclass
class ServiceArea:
    """Service Area Model"""
    id: str
    delivery_units: List['DeliveryUnit'] = field(default_factory=list)
    travel_distance: float = field(default=0.0, init=False)
    
    def calculate_travel_distance(self) -> None:
        """
        Calculate travel distance between centroids in this service area
        using the Traveling Salesman Problem (TSP) approach with nearest neighbor algorithm
        """
        # Get valid centroids
        centroids = []
        for du in self.delivery_units:
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
            self.travel_distance = 0.0
            return
        
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
        
        # Store result
        self.travel_distance = total_distance
    
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
    def is_completed(self) -> bool:
        """Check if the service area is 100% completed"""
        return self.completion_percentage == 100.0
    
    @property
    def is_started(self) -> bool:
        """Check if the service area has at least one completed delivery unit"""
        return self.completed_units > 0
    
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

    @property
    def last_activity_date(self) -> Optional[datetime]:
        """
        Get the last activity date for the service area.
        This is the latest last_activity_date of any delivery unit in the service area.
        """
        dates = [
            du.last_activity_date
            for du in self.delivery_units
            if du.last_activity_date is not None
        ]

        if not dates:
            return None

        return max(dates) 