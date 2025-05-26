from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .service_delivery_point import ServiceDeliveryPoint
    from .delivery_unit import DeliveryUnit


@dataclass
class FLW:
    """Field Level Worker (FLW) Model"""
    id: str
    name: str
    service_areas: List[str] = field(default_factory=list)
    assigned_units: int = 0
    completed_units: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    first_service_delivery_date: Optional[datetime] = None
    last_service_delivery_date: Optional[datetime] = None
    dates_active: List[datetime] = field(default_factory=list)
    service_points: List['ServiceDeliveryPoint'] = field(default_factory=list)
    delivery_units: List['DeliveryUnit'] = field(default_factory=list)
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as a percentage"""
        if self.assigned_units == 0:
            return 0.0
        return (self.completed_units / self.assigned_units) * 100

    def get_service_areas_str(self) -> str:
        """Returns service areas as a comma-separated string"""
        return ', '.join(str(sa) for sa in sorted(self.service_areas))
        
    @property
    def days_active(self) -> int:
        """Return count of unique days the FLW was active"""
        return len(self.dates_active)

    @property
    def delivery_units_completed_per_day(self) -> float:
        """Calculate the average number of delivery units completed per day worked"""
        if self.days_active == 0:
            return 0.0
        return self.completed_units / self.days_active 