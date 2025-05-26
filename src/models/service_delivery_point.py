from dataclasses import dataclass
from typing import Optional, Dict, Any
from shapely.geometry import Point


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