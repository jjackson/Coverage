#!/usr/bin/env python3
"""
Debug script for testing Superset API connection.
This script helps identify issues with Superset API authentication and query execution.
"""

import os
import json
import requests
from dotenv import load_dotenv

def test_superset_connection():
    """Test Superset API connection step by step."""
    
    # Load environment variables
    load_dotenv()
    
    superset_url = os.getenv('SUPERSET_URL')
    username = os.getenv('SUPERSET_USERNAME')
    password = os.getenv('SUPERSET_PASSWORD')
    query_id = os.getenv('SUPERSET_QUERY_ID')
    
    # Add https:// if missing
    if superset_url and not superset_url.startswith(('http://', 'https://')):
        superset_url = f'https://{superset_url}'
    
    print("=== Superset API Debug Test ===")
    print(f"URL: {superset_url}")
    print(f"Username: {username}")
    print(f"Query ID: {query_id}")
    print()
    
    if not all([superset_url, username, password, query_id]):
        print("❌ Missing required environment variables in .env file:")
        missing = []
        if not superset_url: missing.append('SUPERSET_URL')
        if not username: missing.append('SUPERSET_USERNAME') 
        if not password: missing.append('SUPERSET_PASSWORD')
        if not query_id: missing.append('SUPERSET_QUERY_ID')
        for var in missing:
            print(f"   - {var}")
        return False
    
    session = requests.Session()
    
    try:
        # Step 1: Test basic connectivity
        print("1. Testing basic connectivity...")
        ping_url = f'{superset_url}/health'
        try:
            ping_response = session.get(ping_url, timeout=30)
            print(f"   Status: {ping_response.status_code}")
            if ping_response.status_code == 200:
                print("   ✅ Server is reachable")
            else:
                print(f"   ⚠️  Server responded with status {ping_response.status_code}")
        except Exception as e:
            print(f"   ❌ Cannot reach server: {e}")
            return False
        
        # Step 2: Test authentication
        print("\n2. Testing authentication...")
        auth_url = f'{superset_url}/api/v1/security/login'
        auth_payload = {
            'username': username,
            'password': password,
            'provider': 'db',
            'refresh': True
        }
        
        auth_response = session.post(auth_url, json=auth_payload, timeout=30)
        print(f"   Status: {auth_response.status_code}")
        
        if auth_response.status_code == 200:
            auth_data = auth_response.json()
            if 'access_token' in auth_data:
                print("   ✅ Authentication successful")
                access_token = auth_data['access_token']
                headers = {'Authorization': f'Bearer {access_token}'}
            else:
                print(f"   ❌ No access token in response: {auth_data}")
                return False
        else:
            print(f"   ❌ Authentication failed: {auth_response.text}")
            return False
        
        # Step 2b: Get CSRF token
        print("\n2b. Getting CSRF token...")
        csrf_url = f'{superset_url}/api/v1/security/csrf_token/'
        csrf_response = session.get(csrf_url, headers=headers, timeout=30)
        print(f"   Status: {csrf_response.status_code}")
        
        if csrf_response.status_code == 200:
            csrf_data = csrf_response.json()
            csrf_token = csrf_data.get('result')
            if csrf_token:
                headers['X-CSRFToken'] = csrf_token
                # Also add Referer header which is required by Superset
                headers['Referer'] = superset_url
                print("   ✅ CSRF token obtained")
                print(f"   Token preview: {csrf_token[:20]}...")
            else:
                print(f"   ⚠️  No CSRF token in response: {csrf_data}")
        else:
            print(f"   ⚠️  Could not get CSRF token: {csrf_response.text}")
            print("   Continuing without CSRF token...")
        
        # Step 3: Test saved query access
        print(f"\n3. Testing saved query access (ID: {query_id})...")
        saved_query_url = f'{superset_url}/api/v1/saved_query/{query_id}'
        
        saved_query_response = session.get(saved_query_url, headers=headers, timeout=30)
        print(f"   Status: {saved_query_response.status_code}")
        
        if saved_query_response.status_code == 200:
            query_data = saved_query_response.json()
            result = query_data.get('result', {})
            print("   ✅ Saved query found")
            print(f"   Label: {result.get('label', 'N/A')}")
            print(f"   Database ID: {result.get('database', {}).get('id', 'N/A')}")
            print(f"   Schema: {result.get('schema', 'N/A')}")
            print(f"   SQL preview: {result.get('sql', '')[:100]}...")
            
            # Store for next test
            sql_query = result.get('sql', '')
            database_id = result.get('database', {}).get('id', '')
            schema = result.get('schema', '')
            
        elif saved_query_response.status_code == 404:
            print(f"   ❌ Saved query with ID {query_id} not found")
            print("   💡 Try using a Chart ID instead")
            
            # Test if it's a chart
            print(f"\n3b. Testing if ID {query_id} is a chart...")
            chart_url = f'{superset_url}/api/v1/chart/{query_id}'
            chart_response = session.get(chart_url, headers=headers, timeout=30)
            print(f"   Chart Status: {chart_response.status_code}")
            
            if chart_response.status_code == 200:
                chart_data = chart_response.json()
                result = chart_data.get('result', {})
                print("   ✅ Chart found")
                print(f"   Chart Name: {result.get('slice_name', 'N/A')}")
                print(f"   Datasource ID: {result.get('datasource_id', 'N/A')}")
                
                # Try chart data export
                print(f"\n4. Testing chart data export...")
                chart_data_url = f'{superset_url}/api/v1/chart/{query_id}/data'
                
                chart_params = {
                    'format': 'csv',
                    'force': 'true'
                }
                
                chart_data_response = session.get(
                    chart_data_url, 
                    headers=headers, 
                    params=chart_params,
                    timeout=30
                )
                print(f"   Status: {chart_data_response.status_code}")
                
                if chart_data_response.status_code == 200:
                    print("   ✅ Chart data export successful")
                    print(f"   Content-Type: {chart_data_response.headers.get('content-type', 'N/A')}")
                    print(f"   Content-Length: {len(chart_data_response.content)} bytes")
                    return True
                else:
                    print(f"   ❌ Chart data export failed: {chart_data_response.text}")
                    return False
            else:
                print(f"   ❌ ID {query_id} is neither a saved query nor a chart")
                return False
        else:
            print(f"   ❌ Error accessing saved query: {saved_query_response.text}")
            return False
        
        # Step 4: Test SQL execution (if we have a saved query)
        if 'sql_query' in locals() and sql_query and database_id:
            print(f"\n4. Testing SQL execution...")
            execute_url = f'{superset_url}/api/v1/sqllab/execute/'
            
            # Try different approaches to get all rows
            approaches = [
                {
                    'name': 'Basic with minimal parameters',
                    'payload': {
                        'database_id': int(database_id),
                        'sql': sql_query,
                        'runAsync': False,
                        'queryLimit': 1000000,
                    }
                },
                {
                    'name': 'Async execution (proper)',
                    'payload': {
                        'database_id': int(database_id),
                        'sql': sql_query,
                        'runAsync': True,
                        'queryLimit': 1000000,
                    }
                },
                {
                    'name': 'With client_id for tracking',
                    'payload': {
                        'database_id': int(database_id),
                        'sql': sql_query,
                        'runAsync': False,
                        'queryLimit': 1000000,
                        'client_id': f'api_client_{query_id}',
                    }
                }
            ]
            
            for i, approach in enumerate(approaches, 1):
                print(f"\n4.{i} Testing {approach['name']}...")
                print(f"   Payload: {json.dumps(approach['payload'], indent=2)}")
                
                execute_response = session.post(
                    execute_url, 
                    json=approach['payload'],
                    headers=headers,
                    timeout=60  # Longer timeout for larger datasets
                )
                print(f"   Status: {execute_response.status_code}")
                
                # Handle both sync (200) and async (202) responses
                if execute_response.status_code in [200, 202]:
                    execution_result = execute_response.json()
                    print("   ✅ SQL execution successful")
                    print(f"   Status: {execution_result.get('status', 'unknown')}")
                    
                    if approach['payload'].get('runAsync') and execute_response.status_code == 202:
                        # For async, we need to poll for results
                        query_id_result = execution_result.get('query', {}).get('id')
                        if query_id_result:
                            print(f"   Async query ID: {query_id_result}")
                            print("   Polling for results...")
                            
                            # Try different polling endpoints
                            polling_endpoints = [
                                f'{superset_url}/api/v1/sqllab/query/{query_id_result}',
                                f'{superset_url}/api/v1/query/{query_id_result}',
                                f'{superset_url}/api/v1/sqllab/results/{query_id_result}',
                            ]
                            
                            # Poll for results (simplified for testing)
                            import time
                            for attempt in range(15):  # Try 15 times (30 seconds total)
                                time.sleep(2)
                                
                                for endpoint_idx, status_url in enumerate(polling_endpoints):
                                    status_response = session.get(status_url, headers=headers, timeout=30)
                                    
                                    if status_response.status_code == 200:
                                        status_data = status_response.json()
                                        query_status = status_data.get('result', {}).get('status', 'unknown')
                                        
                                        if query_status != 'unknown':
                                            print(f"   Attempt {attempt + 1}, Endpoint {endpoint_idx + 1}: Status = {query_status}")
                                            
                                            if query_status == 'success':
                                                execution_result = status_data.get('result', {})
                                                print(f"   ✅ Async query completed successfully!")
                                                break
                                            elif query_status in ['failed', 'error']:
                                                print(f"   ❌ Async query failed: {status_data}")
                                                break
                                    elif status_response.status_code == 404:
                                        continue  # Try next endpoint
                                    else:
                                        print(f"   ⚠️  Endpoint {endpoint_idx + 1} returned {status_response.status_code}")
                                else:
                                    continue  # Continue outer loop if no endpoint worked
                                break  # Break outer loop if we found a working endpoint
                            else:
                                print("   ⚠️  Async query timeout - no working polling endpoint found")
                                continue
                    
                    if 'data' in execution_result:
                        data_rows = execution_result['data']
                        columns = execution_result.get('columns', [])
                        print(f"   Rows returned: {len(data_rows)}")
                        print(f"   Columns: {len(columns)}")
                        
                        if len(data_rows) > 10000:
                            print(f"   🎉 SUCCESS! Got {len(data_rows)} rows (more than 10k limit)")
                            if columns:
                                column_names = [col.get('name', '') for col in columns]
                                print(f"   Column names: {', '.join(column_names[:5])}{'...' if len(column_names) > 5 else ''}")
                            return True
                        else:
                            print(f"   ⚠️  Still limited to {len(data_rows)} rows")
                        
                        if columns:
                            column_names = [col.get('name', '') for col in columns]
                            print(f"   Column names: {', '.join(column_names[:5])}{'...' if len(column_names) > 5 else ''}")
                    else:
                        print("   ⚠️  No data in response")
                        print(f"   Response keys: {list(execution_result.keys())}")
                        
                else:
                    print(f"   ❌ SQL execution failed: {execute_response.text}")
                
                # If we got data but not all of it, continue to next approach
                if execute_response.status_code in [200, 202] and 'data' in execution_result:
                    if len(execution_result['data']) <= 10000:
                        print("   Trying next approach...")
                        continue
                    else:
                        return True
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    success = test_superset_connection()
    exit(0 if success else 1) 