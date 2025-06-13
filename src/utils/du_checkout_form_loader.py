import os
import requests
import time
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

from ..models.du_checkout_form import DuCheckoutForm


def load_du_checkout_forms_from_api(domain: str, 
                                   user: str,
                                   api_key: str, 
                                   xmlns: str = None,
                                   app_id: str = None,
                                   base_url: str = "https://www.commcarehq.org") -> Dict[str, List[DuCheckoutForm]]:
    """
    Load DU checkout form submissions from CommCare's Form API v0.5.
    
    Args:
        domain: CommCare project space/domain
        user: Username for authentication
        api_key: API key for authentication
        xmlns: Optional form XML namespace to filter by specific form type
        app_id: Optional application ID to filter forms
        base_url: Base URL for the CommCare instance (default: 'https://www.commcarehq.org')
        
    Returns:
        Dictionary with du_name as keys and list of DuCheckoutForm objects as values
    """
    # Build API endpoint
    endpoint = f"{base_url}/a/{domain}/api/v0.5/form/"
    
    # Set up authentication headers
    headers = {
        'Authorization': f'ApiKey {user}:{api_key}', 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    offset = 0
    # Parameters for the API request
    params = {
        'limit': 100,  # Reasonable batch size
        'offset': offset,
        'order_by': 'indexed_on'  # Ensure consistent ordering for pagination
    }
    
    # Add optional filters if provided
    if xmlns:
        params['xmlns'] = xmlns
    if app_id:
        params['app_id'] = app_id
    
    all_forms = []
    
    # Paginate through all results
    while True:
        params['offset'] = offset
        
        num_retry = 5
        response = None
        
        for attempt in range(num_retry):
            response = requests.get(
                endpoint,
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                break  # Success, exit the retry loop
            else:
                if attempt == num_retry - 1:  # Last attempt failed
                    raise ValueError(f"API request failed with status {response.status_code} after {num_retry} attempts")
                else:
                    time.sleep(2)
        
        try:
            data = response.json()
            forms = data.get('objects', [])
            form_count = len(forms)
            
            if form_count == 0:
                break
            
            all_forms.extend(forms)
            
            # Check if we got fewer forms than the limit, indicating we're done
            if form_count < params['limit']:
                break
            
            offset += params['limit']
                   
        except Exception as e:
            raise ValueError(f"JSON parsing error: {str(e)}")
    
    # Parse forms into DuCheckoutForm objects and create dictionary by du_name
    checkout_forms = {}
    parsing_errors = []
    
    for form_data in all_forms:
        try:
            # Parse into DuCheckoutForm object
            form = DuCheckoutForm.from_dict(form_data)
            
            # Use du_name as key, storing all forms in a list
            du_name = form.du_name
            if du_name:
                if du_name not in checkout_forms:
                    checkout_forms[du_name] = []
                checkout_forms[du_name].append(form)
            
        except Exception as e:
            parsing_errors.append(f"Failed to parse form {form_data.get('id', 'unknown')}: {str(e)}")
    
    # Sort forms within each DU by received_on date (most recent first)
    for du_name in checkout_forms:
        checkout_forms[du_name].sort(
            key=lambda x: x.received_on or datetime.min, 
            reverse=True
        )
    
    # Raise exception if there were parsing errors
    if parsing_errors:
        error_msg = f"Encountered {len(parsing_errors)} parsing errors:\n" + "\n".join(parsing_errors[:5])
        if len(parsing_errors) > 5:
            error_msg += f"\n... and {len(parsing_errors) - 5} more errors"
        raise ValueError(error_msg)
    
    return checkout_forms


def load_du_checkout_forms_from_env(xmlns: str = None, 
                                   app_id: str = None,
                                   domain: str = None) -> Dict[str, List[DuCheckoutForm]]:
    """
    Load DU checkout forms using environment variables for authentication.
    
    Args:
        xmlns: Optional form XML namespace to filter by specific form type
        app_id: Optional application ID to filter forms  
        domain: Optional domain override (uses env var if not provided)
        
    Returns:
        Dictionary with du_name as keys and list of DuCheckoutForm objects as values
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Get CommCare configuration from environment variables
    domain = domain or os.getenv('COMMCARE_DOMAIN')
    user = os.getenv('COMMCARE_USERNAME') 
    api_key = os.getenv('COMMCARE_API_KEY')
    base_url = os.getenv('COMMCARE_BASE_URL', 'https://www.commcarehq.org')
    
    # Validate that all required environment variables are set
    missing_vars = []
    if not domain:
        missing_vars.append('COMMCARE_DOMAIN')
    if not user:
        missing_vars.append('COMMCARE_USERNAME')
    if not api_key:
        missing_vars.append('COMMCARE_API_KEY')
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Add https:// scheme if missing from base_url
    if base_url and not base_url.startswith(('http://', 'https://')):
        base_url = f'https://{base_url}'
    
    return load_du_checkout_forms_from_api(
        domain=domain,
        user=user, 
        api_key=api_key,
        xmlns=xmlns,
        app_id=app_id,
        base_url=base_url
    ) 