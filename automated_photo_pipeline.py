#!/usr/bin/env python3
"""
Automated Photo Download Pipeline

Runs: Superset Query ? Sample Visits ? Download Photos
- Date-stamped directories (download once per day)
- Cached Superset data
- Random sampling of N visits per opp_id
- Downloads all photo types from sampled visits
- Organized by opp_id/photo_type/
- CommCareHQ-compatible filename format

Usage:
    python automated_photo_pipeline.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
from collections import defaultdict
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("? Loaded environment variables from .env file")
except ImportError:
    print("??  python-dotenv not installed - using system environment variables only")

# Add src to path for imports
sys.path.append('src')

from src.utils.data_loader import export_superset_query_with_pagination

# Configuration
SUPERSET_QUERY_ID = 215  # Your CommCare visits query
SAMPLE_SIZE = 25  # Number of visits to sample per opp_id

# Photo type column mappings with json_block and question_id for filename formatting
PHOTO_COLUMNS = {
    'muac': {
        'url_column': 'muac_photo_link',
        'json_block': 'muac_group',
        'question_id': 'muac_photo'
    },
    'ors': {
        'url_column': 'photo_link_ors',
        'json_block': 'ors_group',
        'question_id': 'ors_photo'
    },
    'vaccine': {
        'url_column': 'photo_link_vaccine',
        'json_block': 'vita_group',
        'question_id': 'vaccine_photo'
    }
}


class AutomatedPhotoPipeline:
    """Automated pipeline for downloading CommCare photos"""
    
    def __init__(self, base_output_dir=r"C:\Users\Neal Lesh\Coverage\automated_photo_output"):
        """Initialize pipeline with base output directory"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create today's directory
        today = datetime.now().strftime("%Y_%m_%d")
        self.today_dir = self.base_output_dir / today
        self.today_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.superset_data_dir = self.today_dir / "superset_data"
        self.superset_data_dir.mkdir(exist_ok=True)
        
        self.photos_dir = self.today_dir / "photos"
        self.photos_dir.mkdir(exist_ok=True)
        
        # Load credentials
        self.superset_url = os.getenv('SUPERSET_URL')
        self.superset_username = os.getenv('SUPERSET_USERNAME')
        self.superset_password = os.getenv('SUPERSET_PASSWORD')
        self.commcare_username = os.getenv('COMMCARE_USERNAME')
        self.commcare_api_key = os.getenv('COMMCARE_API_KEY')
        
        self._validate_credentials()
        
        # Stats tracking
        self.stats = {
            'total_visits': 0,
            'total_opps': 0,
            'sampled_visits': 0,
            'photos_attempted': 0,
            'photos_downloaded': 0,
            'photos_failed': 0,
            'errors': [],
            'by_type': {photo_type: {'attempted': 0, 'downloaded': 0, 'failed': 0} 
                       for photo_type in PHOTO_COLUMNS}
        }
        
        # Create flattened review directories
        self.review_dirs = {}
        for photo_type in PHOTO_COLUMNS:
            review_dir = self.today_dir / f"photos_for_review_{photo_type}"
            review_dir.mkdir(exist_ok=True)
            self.review_dirs[photo_type] = review_dir
        
        print(f"?? Working directory: {self.today_dir}")
    
    def _validate_credentials(self):
        """Validate all required credentials"""
        missing = []
        
        if not self.superset_url:
            missing.append('SUPERSET_URL')
        if not self.superset_username:
            missing.append('SUPERSET_USERNAME')
        if not self.superset_password:
            missing.append('SUPERSET_PASSWORD')
        if not self.commcare_username:
            missing.append('COMMCARE_USERNAME')
        if not self.commcare_api_key:
            missing.append('COMMCARE_API_KEY')
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        print(f"? Superset credentials loaded: {self.superset_url}")
        print(f"? CommCare credentials loaded for user: {self.commcare_username}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("\n?? Starting Automated Photo Download Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Download Superset data
            csv_file = self._download_superset_data()
            
            # Step 2: Load and sample visits
            df = pd.read_csv(csv_file)
            self.stats['total_visits'] = len(df)
            print(f"\n?? Loaded {len(df):,} total visits")
            
            sampled_visits = self._sample_visits(df)
            
            # Step 3: Download photos
            self._download_photos(sampled_visits)
            
            # Step 4: Export stats
            self._export_stats()
            
            print(f"\n?? Pipeline completed!")
            self._print_summary()
            
        except Exception as e:
            print(f"\n?? Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _download_superset_data(self):
        """Download data from Superset (cache-aware)"""
        print("\n?? Step 1: Downloading Superset Data")
        print("-" * 40)
        
        csv_file = self.superset_data_dir / "visits_data.csv"
        
        if csv_file.exists():
            print(f"  ? Using cached data ({csv_file.name})")
            return csv_file
        
        print(f"  ?? Downloading from query {SUPERSET_QUERY_ID}...")
        
        try:
            # Get SQL from saved query
            sql_query = self._get_sql_from_saved_query(SUPERSET_QUERY_ID)
            
            # Download data
            downloaded_file = export_superset_query_with_pagination(
                superset_url=self.superset_url,
                sql_query=sql_query,
                username=self.superset_username,
                password=self.superset_password,
                output_filename=str(csv_file.with_suffix(''))
            )
            
            if os.path.exists(downloaded_file):
                df = pd.read_csv(downloaded_file)
                print(f"  ? Downloaded {len(df):,} rows")
                return Path(downloaded_file)
            else:
                raise RuntimeError("Download failed")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download Superset data: {str(e)}")
    
    def _get_sql_from_saved_query(self, query_id):
        """Get SQL query from Superset saved query ID"""
        try:
            session = requests.Session()
            
            # Login
            auth_url = f'{self.superset_url}/api/v1/security/login'
            auth_data = {
                'username': self.superset_username,
                'password': self.superset_password,
                'provider': 'db',
                'refresh': True
            }
            
            response = session.post(auth_url, json=auth_data, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Authentication failed: {response.text}")
            
            auth_data = response.json()
            access_token = auth_data.get('access_token')
            if not access_token:
                raise RuntimeError("No access token received")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get CSRF token
            csrf_url = f'{self.superset_url}/api/v1/security/csrf_token/'
            csrf_response = session.get(csrf_url, headers=headers, timeout=30)
            if csrf_response.status_code == 200:
                csrf_data = csrf_response.json()
                csrf_token = csrf_data.get('result')
                if csrf_token:
                    headers['x-csrftoken'] = csrf_token
                    headers['Referer'] = self.superset_url + "/sqllab"
            
            # Get saved query
            saved_query_url = f'{self.superset_url}/api/v1/saved_query/{query_id}'
            response = session.get(saved_query_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get saved query {query_id}: {response.text}")
            
            query_data = response.json()
            result = query_data.get('result', {})
            sql_query = result.get('sql', '')
            
            if not sql_query:
                raise RuntimeError(f"No SQL found in saved query {query_id}")
            
            return sql_query
            
        except Exception as e:
            raise RuntimeError(f"Failed to get SQL from saved query {query_id}: {str(e)}")
    
    def _sample_visits(self, df):
        """Sample N visits per opp_id that have at least one photo"""
        print("\n?? Step 2: Sampling Visits")
        print("-" * 40)
        
        # Filter to visits that have at least one photo URL
        photo_cols = [PHOTO_COLUMNS[photo_type]['url_column'] for photo_type in PHOTO_COLUMNS]
        has_photo = df[photo_cols].notna().any(axis=1)
        df_with_photos = df[has_photo].copy()
        
        print(f"  ?? Found {len(df_with_photos):,} visits with photos")
        
        # Group by opp_id and sample
        sampled = []
        opps = df_with_photos['opp_id'].unique()
        self.stats['total_opps'] = len(opps)
        
        print(f"  ?? Sampling {SAMPLE_SIZE} visits from each of {len(opps)} opp_ids...")
        
        for opp_id in opps:
            opp_visits = df_with_photos[df_with_photos['opp_id'] == opp_id]
            
            # Sample up to N visits
            n_sample = min(SAMPLE_SIZE, len(opp_visits))
            sampled_opp = opp_visits.sample(n=n_sample, random_state=42)
            sampled.append(sampled_opp)
            
            if len(opp_visits) < SAMPLE_SIZE:
                print(f"    ??  opp_id {opp_id}: only {len(opp_visits)} visits available")
        
        result = pd.concat(sampled, ignore_index=True)
        self.stats['sampled_visits'] = len(result)
        
        print(f"  ? Sampled {len(result):,} total visits")
        
        return result
    
    def _extract_form_uuid_from_url(self, url):
        """Extract form UUID from CommCareHQ photo URL"""
        # URL format: https://.../api/form/attachment/{form_uuid}/{filename}.jpg
        match = re.search(r'/attachment/([a-f0-9\-]{36})/', url)
        if match:
            return match.group(1)
        return None
    
    def _download_photos(self, df):
        """Download all photos from sampled visits"""
        print("\n?? Step 3: Downloading Photos")
        print("-" * 40)
        
        for idx, row in df.iterrows():
            opp_id = row['opp_id']
            visit_id = row.get('visit_id', f'visit_{idx}')
            username = row.get('username', 'unknown')
            
            # Create opp_id directory
            opp_dir = self.photos_dir / str(opp_id)
            opp_dir.mkdir(exist_ok=True)
            
            # Try to download each photo type
            for photo_type, photo_config in PHOTO_COLUMNS.items():
                url_column = photo_config['url_column']
                json_block = photo_config['json_block']
                question_id = photo_config['question_id']
                
                photo_url = row.get(url_column)
                
                if pd.isna(photo_url) or not photo_url:
                    continue
                
                # Count as attempted (we have a URL to try)
                self.stats['photos_attempted'] += 1
                self.stats['by_type'][photo_type]['attempted'] += 1
                
                # Skip incomplete URLs
                if not photo_url.strip().endswith(('.jpg', '.jpeg', '.png')):
                    error_msg = f"Incomplete URL for opp_id={opp_id}, type={photo_type}: {photo_url}"
                    self.stats['errors'].append(error_msg)
                    self.stats['photos_failed'] += 1
                    self.stats['by_type'][photo_type]['failed'] += 1
                    continue
                
                # Extract form UUID from URL
                form_uuid = self._extract_form_uuid_from_url(photo_url)
                if not form_uuid:
                    error_msg = f"Could not extract form UUID from URL: {photo_url}"
                    self.stats['errors'].append(error_msg)
                    self.stats['photos_failed'] += 1
                    self.stats['by_type'][photo_type]['failed'] += 1
                    continue
                
                # Create photo type directory
                photo_type_dir = opp_dir / photo_type
                photo_type_dir.mkdir(exist_ok=True)
                
                # Generate filename in CommCareHQ format
                # Format: {json_block}-{question_id}-{username}-form_{form_uuid}.jpg
                filename = f"{json_block}-{question_id}-{username}-form_{form_uuid}.jpg"
                filepath = photo_type_dir / filename
                
                # Download photo
                success = self._download_single_photo(photo_url, filepath, opp_id, photo_type)
                
                if success:
                    self.stats['photos_downloaded'] += 1
                    self.stats['by_type'][photo_type]['downloaded'] += 1
                    
                    # Also copy to flattened review directory
                    review_filepath = self.review_dirs[photo_type] / filename
                    import shutil
                    shutil.copy2(filepath, review_filepath)
                else:
                    self.stats['photos_failed'] += 1
                    self.stats['by_type'][photo_type]['failed'] += 1
        
        print(f"\n  ? Downloaded {self.stats['photos_downloaded']:,} photos")
        print(f"  ? Failed {self.stats['photos_failed']:,} photos")
    
    def _download_single_photo(self, url, filepath, opp_id, photo_type):
        """Download a single photo from CommCare"""
        try:
            response = requests.get(
                url,
                auth=HTTPBasicAuth(self.commcare_username, self.commcare_api_key),
                timeout=30
            )
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                error_msg = f"HTTP {response.status_code} for opp_id={opp_id}, type={photo_type}: {url}"
                self.stats['errors'].append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Exception for opp_id={opp_id}, type={photo_type}: {str(e)}"
            self.stats['errors'].append(error_msg)
            return False
    
    def _export_stats(self):
        """Export statistics to CSV"""
        print("\n?? Step 4: Exporting Statistics")
        print("-" * 40)
        
        # Summary stats
        summary_data = {
            'metric': [
                'Total Visits in Data',
                'Total Opportunity IDs',
                'Sampled Visits',
                'Photos Attempted',
                'Photos Downloaded',
                'Photos Failed',
                'Success Rate (%)'
            ],
            'value': [
                self.stats['total_visits'],
                self.stats['total_opps'],
                self.stats['sampled_visits'],
                self.stats['photos_attempted'],
                self.stats['photos_downloaded'],
                self.stats['photos_failed'],
                round(100 * self.stats['photos_downloaded'] / max(self.stats['photos_attempted'], 1), 2)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.today_dir / "download_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"  ? Saved summary: {summary_file.name}")
        
        # Stats by photo type
        type_stats_data = {
            'photo_type': [],
            'attempted': [],
            'downloaded': [],
            'failed': [],
            'success_rate': []
        }
        
        for photo_type in PHOTO_COLUMNS:
            stats = self.stats['by_type'][photo_type]
            type_stats_data['photo_type'].append(photo_type)
            type_stats_data['attempted'].append(stats['attempted'])
            type_stats_data['downloaded'].append(stats['downloaded'])
            type_stats_data['failed'].append(stats['failed'])
            success_rate = 100 * stats['downloaded'] / max(stats['attempted'], 1)
            type_stats_data['success_rate'].append(round(success_rate, 2))
        
        type_stats_df = pd.DataFrame(type_stats_data)
        type_stats_file = self.today_dir / "download_summary_by_type.csv"
        type_stats_df.to_csv(type_stats_file, index=False)
        print(f"  ? Saved by-type summary: {type_stats_file.name}")
        
        # Error log
        if self.stats['errors']:
            errors_df = pd.DataFrame({'error': self.stats['errors']})
            errors_file = self.today_dir / "download_errors.csv"
            errors_df.to_csv(errors_file, index=False)
            print(f"  ? Saved errors: {errors_file.name} ({len(self.stats['errors'])} errors)")
    
    def _print_summary(self):
        """Print final summary"""
        print("\n?? Pipeline Summary")
        print("=" * 60)
        print(f"Total Visits: {self.stats['total_visits']:,}")
        print(f"Opportunity IDs: {self.stats['total_opps']:,}")
        print(f"Sampled Visits: {self.stats['sampled_visits']:,}")
        print(f"Photos Downloaded: {self.stats['photos_downloaded']:,}")
        print(f"Photos Failed: {self.stats['photos_failed']:,}")
        
        success_rate = 100 * self.stats['photos_downloaded'] / max(self.stats['photos_attempted'], 1)
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n?? By Photo Type:")
        for photo_type in PHOTO_COLUMNS:
            stats = self.stats['by_type'][photo_type]
            type_success = 100 * stats['downloaded'] / max(stats['attempted'], 1)
            print(f"  {photo_type}: {stats['downloaded']}/{stats['attempted']} ({type_success:.1f}%)")
        
        print(f"\n?? Output: {self.today_dir}")
        print(f"   Organized by opp_id: photos/")
        print(f"   Flattened for review:")
        for photo_type in PHOTO_COLUMNS:
            print(f"    - photos_for_review_{photo_type}/")


def main():
    """Main entry point"""
    print("?? Automated Photo Download Pipeline")
    print("=" * 60)
    
    try:
        pipeline = AutomatedPhotoPipeline()
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"\n?? Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
