from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .delivery_unit import DeliveryUnit


@dataclass
class ServiceArea:
    """Service Area Model"""
    id: str
    delivery_units: List['DeliveryUnit'] = field(default_factory=list)
    
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