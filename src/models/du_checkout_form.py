from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd


@dataclass
class PhotoAttachment:
    """Model for photo attachments in the form"""
    photo_link: Optional[str] = None
    pic_filename: Optional[str] = None
    picture_type: Optional[str] = None


@dataclass
class PrimaryCheck:
    """Model for pri_check section of the form"""
    confirm_no_beneficiary_found: Optional[str] = None
    du_delivery_count: int = 0
    du_delivery_target: int = 0
    finished_working_in_du: Optional[str] = None
    no_beneficiary_found_other_reason: Optional[str] = None
    photo_attachment: Optional[PhotoAttachment] = None
    whether_found_child_under_five: Optional[str] = None


@dataclass
class DuUpdateBlock:
    """Model for du_update_block section of the form"""
    checked_out_date: Optional[str] = None
    du_checkout_remark: Optional[str] = None
    du_status: Optional[str] = None
    label_du_visited: Optional[str] = None


@dataclass
class LocationData:
    """Model for location/GPS data"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    raw_location: Optional[str] = None


@dataclass
class DuCheckoutForm:
    """Model for DU Checkout Form submissions from CommCare"""
    
    # Form identification
    form_id: str
    app_id: Optional[str] = None
    build_id: Optional[str] = None
    domain: Optional[str] = None
    xmlns: Optional[str] = None
    
    # Timing information
    received_on: Optional[datetime] = None
    server_modified_on: Optional[datetime] = None
    indexed_on: Optional[datetime] = None
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    
    # User information
    user_id: Optional[str] = None
    username: Optional[str] = None
    device_id: Optional[str] = None
    submit_ip: Optional[str] = None
    
    # Form data - main fields
    checked_in_du: Optional[str] = None
    confirm_check_out: Optional[str] = None
    du_case_id: Optional[str] = None
    du_name: Optional[str] = None
    please_visit_the_entire_cluster: Optional[str] = None
    update_current_du: Optional[str] = None
    
    # Structured sections
    du_update_block: Optional[DuUpdateBlock] = None
    primary_check: Optional[PrimaryCheck] = None
    location_data: Optional[LocationData] = None
    
    # App metadata
    app_version: Optional[str] = None
    commcare_version: Optional[str] = None
    app_build_version: Optional[int] = None
    
    # Form metadata
    archived: bool = False
    version: Optional[str] = None
    uiversion: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DuCheckoutForm':
        """Create a DuCheckoutForm from a dictionary (API response)"""
        
        # Extract top-level form metadata
        form_id = data.get('id', '')
        app_id = data.get('app_id')
        build_id = data.get('build_id')
        domain = data.get('domain')
        archived = data.get('archived', False)
        version = data.get('version')
        uiversion = data.get('uiversion')
        submit_ip = data.get('submit_ip')
        
        # Extract xmlns from form data
        form_data = data.get('form', {})
        xmlns = form_data.get('@xmlns')
        
        # Parse timing information
        received_on = cls._parse_datetime(data.get('received_on'))
        server_modified_on = cls._parse_datetime(data.get('server_modified_on'))
        indexed_on = cls._parse_datetime(data.get('indexed_on'))
        
        # Extract metadata
        metadata = data.get('metadata', {})
        user_id = metadata.get('userID')
        username = metadata.get('username')
        device_id = metadata.get('deviceID')
        app_version = metadata.get('appVersion')
        commcare_version = metadata.get('commcare_version')
        app_build_version = metadata.get('app_build_version')
        
        # Parse form meta timing
        form_meta = form_data.get('meta', {})
        time_start = cls._parse_datetime(form_meta.get('timeStart'))
        time_end = cls._parse_datetime(form_meta.get('timeEnd'))
        
        # Extract main form fields
        checked_in_du = form_data.get('checked_in_du')
        confirm_check_out = form_data.get('confirm_check_out')
        du_case_id = form_data.get('du_case_id')
        du_name = form_data.get('du_name')
        please_visit_cluster = form_data.get('please_visit_the_entire_cluster')
        update_current_du = form_data.get('update_current_du')
        
        # Parse du_update_block
        du_update_data = form_data.get('du_update_block', {})
        du_update_block = DuUpdateBlock(
            checked_out_date=du_update_data.get('checked_out_date'),
            du_checkout_remark=du_update_data.get('du_checkout_remark'),
            du_status=du_update_data.get('du_status'),
            label_du_visited=du_update_data.get('label_du_visited')
        ) if du_update_data else None
        
        # Parse pri_check section
        pri_check_data = form_data.get('pri_check', {})
        photo_attachment = None
        
        if pri_check_data:
            # Extract photo data
            photo_data = pri_check_data.get('please_take_a_picture_of_a_building_in_this_delivery_unit', {})
            if photo_data:
                photo_attachment = PhotoAttachment(
                    photo_link=photo_data.get('photo_link'),
                    pic_filename=photo_data.get('pic_in_du'),
                    picture_type=photo_data.get('picture_of_building')
                )
            
            primary_check = PrimaryCheck(
                confirm_no_beneficiary_found=pri_check_data.get('confirm_no_beneficiary_found'),
                du_delivery_count=int(pri_check_data.get('du_delivery_count', 0)),
                du_delivery_target=int(pri_check_data.get('du_delivery_target', 0)),
                finished_working_in_du=pri_check_data.get('finished_working_in_du'),
                no_beneficiary_found_other_reason=pri_check_data.get('no_beneficiary_found_other_reason'),
                photo_attachment=photo_attachment,
                whether_found_child_under_five=pri_check_data.get('whether_found_child_under_five')
            )
        else:
            primary_check = None
        
        # Parse location data
        location_data = None
        location_raw = form_meta.get('location', {})
        if location_raw:
            location_text = location_raw.get('#text') if isinstance(location_raw, dict) else str(location_raw)
            if location_text:
                coords = location_text.split()
                if len(coords) >= 2:
                    try:
                        location_data = LocationData(
                            latitude=float(coords[0]),
                            longitude=float(coords[1]),
                            altitude=float(coords[2]) if len(coords) > 2 else None,
                            accuracy=float(coords[3]) if len(coords) > 3 else None,
                            raw_location=location_text
                        )
                    except (ValueError, IndexError):
                        location_data = LocationData(raw_location=location_text)
        
        return cls(
            form_id=form_id,
            app_id=app_id,
            build_id=build_id,
            domain=domain,
            xmlns=xmlns,
            received_on=received_on,
            server_modified_on=server_modified_on,
            indexed_on=indexed_on,
            time_start=time_start,
            time_end=time_end,
            user_id=user_id,
            username=username,
            device_id=device_id,
            submit_ip=submit_ip,
            checked_in_du=checked_in_du,
            confirm_check_out=confirm_check_out,
            du_case_id=du_case_id,
            du_name=du_name,
            please_visit_the_entire_cluster=please_visit_cluster,
            update_current_du=update_current_du,
            du_update_block=du_update_block,
            primary_check=primary_check,
            location_data=location_data,
            app_version=app_version,
            commcare_version=commcare_version,
            app_build_version=app_build_version,
            archived=archived,
            version=version,
            uiversion=uiversion
        )
    
    @staticmethod
    def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export/analysis"""
        return {
            'form_id': self.form_id,
            'app_id': self.app_id,
            'domain': self.domain,
            'received_on': self.received_on,
            'du_case_id': self.du_case_id,
            'du_name': self.du_name,
            'du_status': self.du_update_block.du_status if self.du_update_block else None,
            'checked_out_date': self.du_update_block.checked_out_date if self.du_update_block else None,
            'du_checkout_remark': self.du_update_block.du_checkout_remark if self.du_update_block else None,
            'delivery_count': self.primary_check.du_delivery_count if self.primary_check else 0,
            'delivery_target': self.primary_check.du_delivery_target if self.primary_check else 0,
            'finished_working': self.primary_check.finished_working_in_du if self.primary_check else None,
            'found_child_under_five': self.primary_check.whether_found_child_under_five if self.primary_check else None,
            'user_id': self.user_id,
            'username': self.username,
            'latitude': self.location_data.latitude if self.location_data else None,
            'longitude': self.location_data.longitude if self.location_data else None,
            'time_start': self.time_start,
            'time_end': self.time_end
        } 