#!/usr/bin/env python3
"""
Superset Paginated Data Export Script
This script fetches all data from a Superset saved query by using pagination 
to bypass the 10,000 row limit.
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import time
from typing import List, Dict, Any

def get_superset_session():
    """Create authenticated Superset session."""
    load_dotenv()
    
    superset_url = os.getenv('SUPERSET_URL')
    username = os.getenv('SUPERSET_USERNAME')
    password = os.getenv('SUPERSET_PASSWORD')
    
    # Add https:// if missing
    if superset_url and not superset_url.startswith(('http://', 'https://')):
        superset_url = f'https://{superset_url}'
    
    if not all([superset_url, username, password]):
        raise ValueError("Missing required environment variables: SUPERSET_URL, SUPERSET_USERNAME, SUPERSET_PASSWORD")
    
    session = requests.Session()
    
    # Authenticate
    auth_url = f'{superset_url}/api/v1/security/login'
    auth_payload = {
        'username': username,
        'password': password,
        'provider': 'db',
        'refresh': True
    }
    
    auth_response = session.post(auth_url, json=auth_payload, timeout=30)
    if auth_response.status_code != 200:
        raise Exception(f"Authentication failed: {auth_response.text}")
    
    auth_data = auth_response.json()
    if 'access_token' not in auth_data:
        raise Exception(f"No access token in response: {auth_data}")
    
    access_token = auth_data['access_token']
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Get CSRF token
    csrf_url = f'{superset_url}/api/v1/security/csrf_token/'
    csrf_response = session.get(csrf_url, headers=headers, timeout=30)
    if csrf_response.status_code == 200:
        csrf_data = csrf_response.json()
        csrf_token = csrf_data.get('result')
        if csrf_token:
            headers['X-CSRFToken'] = csrf_token
            headers['Referer'] = superset_url
    
    return session, headers, superset_url

def get_saved_query_details(session, headers, superset_url, query_id):
    """Get saved query SQL and database details."""
    saved_query_url = f'{superset_url}/api/v1/saved_query/{query_id}'
    response = session.get(saved_query_url, headers=headers, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Failed to get saved query: {response.text}")
    
    query_data = response.json()
    result = query_data.get('result', {})
    
    return {
        'sql': result.get('sql', ''),
        'database_id': result.get('database', {}).get('id', ''),
        'label': result.get('label', 'Unknown'),
        'schema': result.get('schema', '')
    }

def execute_paginated_query(session, headers, superset_url, database_id, base_sql, chunk_size=10000):
    """Execute query with pagination to get all results."""
    execute_url = f'{superset_url}/api/v1/sqllab/execute/'
    all_data = []
    all_columns = None
    offset = 0
    total_rows = 0
    chunk_num = 1
    
    print(f"🔄 Starting paginated data fetch (chunk size: {chunk_size:,})")
    print(f"📝 Base SQL preview: {base_sql[:100]}...")
    print()
    
    while True:
        # Add OFFSET and LIMIT to the SQL
        paginated_sql = f"""
{base_sql.rstrip(';')}
OFFSET {offset}
LIMIT {chunk_size}
"""
        
        payload = {
            'database_id': int(database_id),
            'sql': paginated_sql,
            'runAsync': False,
            'queryLimit': chunk_size + 1000,  # Add buffer
        }
        
        print(f"📦 Fetching chunk {chunk_num} (rows {offset + 1:,} to {offset + chunk_size:,})...")
        
        try:
            response = session.post(execute_url, json=payload, headers=headers, timeout=120)
            
            if response.status_code != 200:
                print(f"❌ Chunk {chunk_num} failed: {response.text}")
                break
            
            result = response.json()
            
            if result.get('status') != 'success':
                print(f"❌ Chunk {chunk_num} query failed: {result}")
                break
            
            # Get data and columns
            chunk_data = result.get('data', [])
            columns = result.get('columns', [])
            
            if not chunk_data:
                print(f"✅ No more data found. Stopping at chunk {chunk_num - 1}")
                break
            
            # Store columns from first chunk
            if all_columns is None:
                all_columns = columns
                print(f"📊 Columns found: {len(columns)} columns")
                column_names = [col.get('name', '') for col in columns]
                print(f"   Column names: {', '.join(column_names[:5])}{'...' if len(column_names) > 5 else ''}")
            
            # Add chunk data to overall results
            all_data.extend(chunk_data)
            chunk_rows = len(chunk_data)
            total_rows += chunk_rows
            
            print(f"✅ Chunk {chunk_num}: {chunk_rows:,} rows fetched (Total: {total_rows:,})")
            
            # If we got fewer rows than chunk_size, we've reached the end
            if chunk_rows < chunk_size:
                print(f"🎉 Reached end of data (chunk had {chunk_rows:,} < {chunk_size:,} rows)")
                break
            
            # Prepare for next chunk
            offset += chunk_size
            chunk_num += 1
            
            # Small delay to be nice to the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error fetching chunk {chunk_num}: {e}")
            break
    
    print(f"\n🎯 Pagination complete!")
    print(f"   Total rows fetched: {total_rows:,}")
    print(f"   Total chunks: {chunk_num - 1}")
    
    return all_data, all_columns

def export_to_csv(data, columns, filename):
    """Export data to CSV file."""
    if not data or not columns:
        print("❌ No data to export")
        return False
    
    # Create DataFrame
    column_names = [col.get('name', f'col_{i}') for i, col in enumerate(columns)]
    df = pd.DataFrame(data, columns=column_names)
    
    # Export to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"💾 Data exported to: {filename}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    
    # Show preview
    print(f"\n📋 Data preview:")
    print(df.head().to_string())
    
    return True

def main():
    """Main function to export Superset data with pagination."""
    load_dotenv()
    query_id = os.getenv('SUPERSET_QUERY_ID')
    
    if not query_id:
        print("❌ Missing SUPERSET_QUERY_ID in .env file")
        return False
    
    try:
        print("=== Superset Paginated Data Export ===")
        print(f"Query ID: {query_id}")
        print()
        
        # Set up session
        print("🔐 Authenticating with Superset...")
        session, headers, superset_url = get_superset_session()
        
        # Get query details
        print("📋 Getting saved query details...")
        query_details = get_saved_query_details(session, headers, superset_url, query_id)
        
        print(f"   Query: {query_details['label']}")
        print(f"   Database ID: {query_details['database_id']}")
        print(f"   Schema: {query_details['schema']}")
        print()
        
        # Execute paginated query
        data, columns = execute_paginated_query(
            session, headers, superset_url, 
            query_details['database_id'], 
            query_details['sql']
        )
        
        if data:
            # Export to CSV
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"superset_export_{query_id}_{timestamp}.csv"
            
            success = export_to_csv(data, columns, filename)
            
            if success:
                print(f"\n🎉 Export completed successfully!")
                print(f"   Final row count: {len(data):,}")
                return True
        
        print("❌ No data was exported")
        return False
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False
    
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 