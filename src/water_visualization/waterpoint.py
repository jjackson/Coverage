"""
Water Point Data Model for CommCare Water Survey Visualization
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import os


@dataclass
class WaterPoint:
    """Data model for a water point from CommCare water survey data."""
    
    # Core identifiers
    number: int
    formid: str  # UUID from CommCare
    username: str  # Data collector ID (cowws1, cwws2, etc.)
    
    # Location hierarchy
    state: str
    lga: str  # Local Government Area
    ward: str
    community: str
    
    # Coordinates (parsed from form.gps)
    latitude: float
    longitude: float
    
    # Water point characteristics
    water_point_type: str  # piped_water, borehole_hand_pump, protected_wells, etc.
    is_piped: bool
    has_dispenser: bool
    usage_level: str  # high, medium, low
    
    # Survey metadata
    time_of_visit: datetime
    completed_time: datetime
    started_time: datetime
    received_on: datetime
    
    # CommCare references
    form_link: str
    hq_user: str
    
    # Optional fields with defaults
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    chlorine_dispenser_functional: Optional[bool] = None
    other_treatment: Optional[str] = None
    household_estimate: Optional[int] = None
    notes: Optional[str] = None
    
    # Images (local file paths after processing)
    photo1_path: Optional[str] = None
    photo2_path: Optional[str] = None
    photo3_path: Optional[str] = None
    
    @property
    def project_name(self) -> str:
        """Extract project name from username prefix."""
        if self.username.startswith('cowws'):
            return 'COWACDI'
        elif self.username.startswith('cwws'):
            return 'CWINS'
        return 'UNKNOWN'
    
    @property
    def location_breadcrumb(self) -> str:
        """Generate location hierarchy breadcrumb."""
        return f"{self.state} > {self.lga} > {self.ward} > {self.community}"
    
    @property
    def available_photos(self) -> List[str]:
        """Get list of available photo paths."""
        photos = []
        for photo_path in [self.photo1_path, self.photo2_path, self.photo3_path]:
            if photo_path and os.path.exists(photo_path):
                photos.append(photo_path)
        return photos
    
    @property
    def water_point_type_display(self) -> str:
        """Convert water point type to display-friendly format."""
        type_map = {
            'piped_water': 'Piped Water',
            'borehole_hand_pump': 'Borehole Hand Pump',
            'protected_wells': 'Protected Well',
            'unprotected_wells': 'Unprotected Well',
            'surface_water': 'Surface Water'
        }
        return type_map.get(self.water_point_type, self.water_point_type.replace('_', ' ').title())
    
    @property
    def usage_level_display(self) -> str:
        """Convert usage level to display-friendly format."""
        return self.usage_level.title() if self.usage_level else 'Unknown'
    
    @classmethod
    def from_excel_row(cls, row, project_base_path: str) -> 'WaterPoint':
        """Create WaterPoint instance from Excel row data."""
        
        # Parse GPS coordinates from space-separated format
        gps_parts = str(row['form.gps']).split()
        latitude = float(gps_parts[0]) if len(gps_parts) > 0 else 0.0
        longitude = float(gps_parts[1]) if len(gps_parts) > 1 else 0.0
        altitude = float(gps_parts[2]) if len(gps_parts) > 2 else None
        accuracy = float(gps_parts[3]) if len(gps_parts) > 3 else None
        
        # Parse boolean fields
        is_piped = str(row.get('form.is_piped', '')).lower() == 'yes'
        has_dispenser = str(row.get('form.has_dispenser', '')).lower() == 'yes'
        
        # Parse chlorine dispenser functionality
        chlorine_functional = row.get('form.if_available_is_the_chlorine_dispenser_functional')
        chlorine_dispenser_functional = None
        if chlorine_functional and str(chlorine_functional).strip() not in ['', '---', 'nan']:
            chlorine_dispenser_functional = str(chlorine_functional).lower() == 'yes'
        
        # Parse datetime fields
        def parse_datetime(dt_str):
            if not dt_str or str(dt_str).strip() in ['', 'nan', 'NaT']:
                return datetime.now()
            if isinstance(dt_str, datetime):
                return dt_str
            # Handle various datetime formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S']:
                try:
                    return datetime.strptime(str(dt_str), fmt)
                except ValueError:
                    continue
            return datetime.now()
        
        # Resolve image paths based on filename pattern
        formid = str(row['formid'])
        username = str(row['username'])
        
        def get_image_path(photo_num: int) -> Optional[str]:
            """Get local image path for photo number (1, 2, or 3)."""
            filename = f"photo{photo_num}-{username}-form_{formid}.jpg"
            image_path = os.path.join(project_base_path, filename)
            return image_path if os.path.exists(image_path) else None
        
        return cls(
            number=int(row['number']),
            formid=formid,
            username=username,
            
            # Location
            state=str(row['form.state']),
            lga=str(row['form.lga']),
            ward=str(row['form.ward']),
            community=str(row['form.community']),
            
            # Coordinates
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            accuracy=accuracy,
            
            # Water point characteristics
            water_point_type=str(row['form.water_point_type']),
            is_piped=is_piped,
            has_dispenser=has_dispenser,
            chlorine_dispenser_functional=chlorine_dispenser_functional,
            other_treatment=str(row.get('form.other_treatment', '')) if row.get('form.other_treatment') else None,
            usage_level=str(row['form.usage_level']),
            household_estimate=int(row['form.hh_estimate']) if row.get('form.hh_estimate') and str(row['form.hh_estimate']).strip() not in ['', 'nan'] else None,
            
            # Survey metadata
            time_of_visit=parse_datetime(row['form.time_of_visit']),
            notes=str(row['form.notes']) if row.get('form.notes') and str(row['form.notes']).strip() not in ['', 'nan'] else None,
            completed_time=parse_datetime(row['completed_time']),
            started_time=parse_datetime(row['started_time']),
            received_on=parse_datetime(row['received_on']),
            
            # Images
            photo1_path=get_image_path(1),
            photo2_path=get_image_path(2),
            photo3_path=get_image_path(3),
            
            # CommCare references
            form_link=str(row['form_link']),
            hq_user=str(row['hq_user'])
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'number': self.number,
            'formid': self.formid,
            'username': self.username,
            'project_name': self.project_name,
            
            # Location
            'state': self.state,
            'lga': self.lga,
            'ward': self.ward,
            'community': self.community,
            'location_breadcrumb': self.location_breadcrumb,
            
            # Coordinates
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy': self.accuracy,
            
            # Water point characteristics
            'water_point_type': self.water_point_type,
            'water_point_type_display': self.water_point_type_display,
            'is_piped': self.is_piped,
            'has_dispenser': self.has_dispenser,
            'chlorine_dispenser_functional': self.chlorine_dispenser_functional,
            'other_treatment': self.other_treatment,
            'usage_level': self.usage_level,
            'usage_level_display': self.usage_level_display,
            'household_estimate': self.household_estimate,
            
            # Survey metadata
            'time_of_visit': self.time_of_visit.isoformat(),
            'notes': self.notes,
            'completed_time': self.completed_time.isoformat(),
            'started_time': self.started_time.isoformat(),
            'received_on': self.received_on.isoformat(),
            
            # Images
            'photo_count': len(self.available_photos),
            'has_photos': len(self.available_photos) > 0,
            
            # CommCare references
            'form_link': self.form_link,
            'hq_user': self.hq_user
        }
