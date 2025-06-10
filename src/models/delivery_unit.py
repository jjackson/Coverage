from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime
from shapely import wkt
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
import pandas as pd

if TYPE_CHECKING:
    from .service_delivery_point import ServiceDeliveryPoint


@dataclass
class DeliveryUnit:
    """Delivery Unit Model"""
    id: str #the case_id from CommCare
    du_name: str #the human alphanumeric Dimagi generates to name the delivery unit
    service_area_id: str #formatted as oa_id-sa_id, sa_id is unique within an oppurtunity area but can be repeated across oppurtunity areas
    flw_commcare_id: str  # FLW/owner CommCare ID
    status: str  # completed, visited, unvisited represented as None
    wkt: str
    buildings: int = 0
    surface_area: float = 0.0
    delivery_count: int = 0
    delivery_target: int = 0
    du_checkout_remark: Optional[str] = None
    checked_out_date: Optional[str] = None
    checked_in_date: Optional[str] = None
    centroid: Optional[tuple] = None
    last_modified_date: Optional[datetime] = None
    computed_du_completion_date: Optional[datetime] = None
    service_points: List['ServiceDeliveryPoint'] = field(default_factory=list)

    @property
    def geometry(self) -> BaseGeometry:
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
        du_id = data.get('case_id')
        du_name = data.get('du_name')
        
        service_area_id = str(data.get('service_area_id', ''))
        
        # Look for flw_commcare_id first (as it might have been renamed), then fall back to original column names
        flw_val = data.get('flw_commcare_id', data.get('owner_id', data.get('flw', '')))
        flw = str(flw_val) if not pd.isna(flw_val) else ''
        
        status_val = data.get('du_status', 'unvisited')
        status = str(status_val).lower() if not pd.isna(status_val) else 'unvisited'
        
        wkt_str = str(data.get('WKT', '')) if not pd.isna(data.get('WKT', '')) else ''
        
        # Check if WKT is empty
        if not wkt_str or wkt_str == '':
            raise ValueError(f"Empty WKT for delivery unit {du_id}, du_name: {du_name}")
        
        # Extract numeric fields with proper error handling
        buildings = int(data.get('buildings', data.get('#Buildings', 0)))
        surface_area = float(data.get('surface_area', data.get('Surface Area (sq. meters)', 0.0)))
        delivery_count = int(data.get('delivery_count', 0))
        delivery_target = int(data.get('delivery_target', 0))
        
        
        # Handle string fields
        checkout_remark = data.get('du_checkout_remark')
        checkout_date = data.get('checked_out_date')
        checkin_date = data.get('checked_in_date')
        
        # Parse last_modified_date
        last_modified = None
        if 'last_modified_date' in data and data['last_modified_date']:
            if isinstance(data['last_modified_date'], datetime):
                last_modified = data['last_modified_date']
            else:
                # Try parsing the date string
                last_modified = pd.to_datetime(data['last_modified_date'])
        
        # Create the DeliveryUnit instance
        return cls(
            id=du_id,
            du_name=du_name,
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
            checked_in_date=checkin_date,
            centroid=data.get('centroid'),
            last_modified_date=last_modified
        ) 