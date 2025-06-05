# Coverage Data Flow and Data Dictionary

This document explains how data flows into the `CoverageData` object and provides a comprehensive data dictionary for all models used in the coverage analysis system.

## Overview

The Coverage Analysis system processes delivery unit data and service delivery data to create comprehensive coverage analysis and visualizations. The main entry point is the `CoverageData` object which orchestrates the loading and processing of data from multiple sources.

## Data Flow Architecture

### 1. Entry Points

The system supports two main modes of operation:

**API Mode (Recommended for Multiple Opportunities):**
- Loads service delivery data from Superset database
- Fetches delivery unit data from CommCare API for each opportunity
- Automatically handles multiple opportunities/projects

**Local File Mode:**
- Loads from local Excel (delivery units) and CSV (service delivery) files
- Suitable for single opportunity analysis or when API access is not available

### 2. Data Loading Flow

```mermaid
graph TD
    A[Environment Variables] --> B[coverage_master.py main()]
    B --> C{USE_API=True?}
    
    C -->|Yes| D[Load Service Delivery from Superset]
    C -->|No| E[Select Local Excel & CSV Files]
    
    D --> F[Group by Opportunity]
    E --> G[Load Single Opportunity Data]
    
    F --> H[For Each Opportunity]
    G --> H
    
    H --> I[Fetch Delivery Units from CommCare API]
    H --> J[Create CoverageData Object]
    
    I --> K[CoverageData.load_delivery_units_from_df()]
    J --> L[CoverageData.load_service_delivery_from_dataframe()]
    
    K --> M[Process Delivery Units]
    L --> M
    
    M --> N[Compute Metadata & Statistics]
    N --> O[Generate Reports & Visualizations]
```

### 3. Detailed Data Loading Process

#### Step 1: Environment Setup
The system reads configuration from `.env` file:
- `COMMCARE_API_KEY`: API key for CommCare access
- `COMMCARE_USERNAME`: CommCare username
- `USE_API`: Boolean flag to enable API mode
- `OPPORTUNITY_DOMAIN_MAPPING`: JSON mapping of opportunity names to CommCare domains
- `SUPERSET_URL`, `SUPERSET_USERNAME`, `SUPERSET_PASSWORD`, `SUPERSET_QUERY_ID`: Superset database connection details

#### Step 2: Service Delivery Data Loading
**API Mode:**
```python
# From src/utils/data_loader.py
service_df_by_opportunity = load_service_delivery_df_by_opportunity_from_superset(
    superset_url, superset_username, superset_password, superset_query_id
)
```

**Local Mode:**
```python
service_df = load_csv_data(csv_file_path)
```

#### Step 3: Delivery Unit Data Loading
**API Mode:**
```python
# For each opportunity in the service delivery data
for opportunity_name in service_df_by_opportunity.keys():
    domain = opportunity_domain_mapping[opportunity_name]
    coverage_data = get_coverage_data_from_du_api_and_service_dataframe(
        domain, username, api_key, service_df
    )
```

**Local Mode:**
```python
coverage_data = get_coverage_data_from_excel(excel_file_path)
coverage_data.load_service_delivery_from_dataframe(service_df)
```

#### Step 4: CoverageData Object Creation
The `CoverageData` class method `load_delivery_units_from_df()` processes delivery unit data:

1. **Data Cleaning**: Cleans and validates the delivery units DataFrame
2. **Object Creation**: Creates `DeliveryUnit`, `ServiceArea`, and `FLW` objects
3. **Relationship Building**: Links delivery units to service areas and FLWs
4. **Metadata Computation**: Pre-computes statistics and derived data

#### Step 5: Service Delivery Integration
The `load_service_delivery_from_dataframe()` method:

1. **Point Creation**: Creates `ServiceDeliveryPoint` objects from GPS coordinates
2. **Association**: Links service points to their corresponding delivery units
3. **FLW Enhancement**: Updates FLW objects with service delivery information
4. **Date Tracking**: Computes active dates and completion dates

#### Step 6: Metadata Computation
Two key methods compute derived data:

**`_compute_metadata_from_delivery_unit_data()`:**
- Service area progress statistics
- Building density calculations
- FLW service area assignments
- Status counts and distributions

**`_compute_metadata_from_service_delivery_data()`:**
- Delivery unit completion dates
- FLW active date ranges
- Service delivery patterns

## Data Dictionary

### CoverageData (Main Container)

The `CoverageData` class is the main container that holds all coverage analysis data and provides computed statistics.

#### Core Collections
| Property | Type | Description |
|----------|------|-------------|
| `service_areas` | `Dict[str, ServiceArea]` | Dictionary of service areas keyed by service area ID |
| `delivery_units` | `Dict[str, DeliveryUnit]` | Dictionary of delivery units keyed by delivery unit name |
| `service_points` | `List[ServiceDeliveryPoint]` | List of all service delivery points (GPS coordinates) |
| `flws` | `Dict[str, FLW]` | Dictionary of Field Level Workers keyed by CommCare ID |
| `delivery_units_df` | `pd.DataFrame` | Original DataFrame used to create delivery units |

#### Project Identification
| Property | Type | Description |
|----------|------|-------------|
| `project_space` | `str` | CommCare project space identifier |
| `opportunity_name` | `str` | Human-readable opportunity/project name |

#### Cached Metadata
| Property | Type | Description |
|----------|------|-------------|
| `flw_commcare_id_to_name_map` | `Dict[str, str]` | Mapping from CommCare IDs to human-readable FLW names |
| `unique_service_area_ids` | `List[str]` | Sorted list of all service area IDs |
| `unique_flw_names` | `List[str]` | Sorted list of all FLW names |
| `unique_status_values` | `List[str]` | List of all possible delivery unit status values |
| `delivery_status_counts` | `Dict[str, int]` | Count of delivery units by status |

#### Pre-computed Statistics
| Property | Type | Description |
|----------|------|-------------|
| `flw_service_area_stats` | `Dict[str, Dict[str, Any]]` | Statistics for each FLW's performance in each service area |
| `service_area_building_density` | `Dict[str, float]` | Building density (buildings per sq km) for each service area |
| `service_area_progress` | `Dict[str, Dict[str, Any]]` | Progress statistics for each service area |
| `travel_distances` | `Dict[str, float]` | Estimated travel distances for each service area |

#### Computed Properties
| Property | Type | Description |
|----------|------|-------------|
| `total_delivery_units` | `int` | Total number of delivery units |
| `total_service_areas` | `int` | Total number of service areas |
| `total_flws` | `int` | Total number of Field Level Workers |
| `total_buildings` | `int` | Total buildings across all delivery units |
| `total_completed_dus` | `int` | Number of completed delivery units |
| `total_visited_dus` | `int` | Number of visited but not completed delivery units |
| `total_unvisited_dus` | `int` | Number of unvisited delivery units |
| `completion_percentage` | `float` | Overall completion percentage |

### DeliveryUnit Model

Represents a geographic area assigned to an FLW for service delivery.

#### Core Properties
| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Case ID from CommCare (unique identifier) |
| `du_name` | `str` | Human-readable alphanumeric name generated by Dimagi |
| `service_area_id` | `str` | Service area identifier (format: oa_id-sa_id) |
| `flw_commcare_id` | `str` | CommCare ID of assigned Field Level Worker |
| `status` | `str` | Status: 'completed', 'visited', or None (unvisited) |
| `wkt` | `str` | Well-Known Text geometry string defining the delivery unit boundary |

#### Physical Properties
| Property | Type | Description |
|----------|------|-------------|
| `buildings` | `int` | Number of buildings in the delivery unit |
| `surface_area` | `float` | Area in square meters |
| `delivery_count` | `int` | Number of service deliveries completed |
| `delivery_target` | `int` | Target number of service deliveries |
| `centroid` | `tuple` | Geographic center point (latitude, longitude) |

#### Tracking Properties
| Property | Type | Description |
|----------|------|-------------|
| `du_checkout_remark` | `str` | Remark entered when FLW checked out of the delivery unit |
| `checked_out_date` | `str` | Date when FLW checked out |
| `checked_in_date` | `str` | Date when FLW first checked in |
| `last_modified_date` | `datetime` | Last modification timestamp from CommCare |
| `computed_du_completion_date` | `datetime` | Computed completion date based on service deliveries or check-in |

#### Relationships
| Property | Type | Description |
|----------|------|-------------|
| `service_points` | `List[ServiceDeliveryPoint]` | List of service delivery points within this delivery unit |

#### Computed Properties
| Property | Type | Description |
|----------|------|-------------|
| `geometry` | `BaseGeometry` | Shapely geometry object created from WKT |
| `completion_percentage` | `float` | Percentage of delivery target completed |

### ServiceArea Model

Represents a collection of delivery units grouped together for administrative purposes.

#### Core Properties
| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Service area identifier (unique within opportunity) |
| `delivery_units` | `List[DeliveryUnit]` | List of delivery units in this service area |
| `travel_distance` | `float` | Estimated travel distance between delivery units (TSP algorithm) |

#### Computed Properties
| Property | Type | Description |
|----------|------|-------------|
| `total_buildings` | `int` | Sum of buildings across all delivery units |
| `total_surface_area` | `float` | Sum of surface area across all delivery units |
| `total_units` | `int` | Number of delivery units in the service area |
| `completed_units` | `int` | Number of completed delivery units |
| `completion_percentage` | `float` | Percentage of delivery units completed |
| `is_completed` | `bool` | True if all delivery units are completed |
| `assigned_flws` | `List[str]` | List of FLW IDs assigned to this service area |
| `total_deliveries` | `int` | Sum of service deliveries across all delivery units |
| `building_density` | `float` | Buildings per square kilometer |

### FLW (Field Level Worker) Model

Represents a field worker responsible for service delivery.

#### Core Properties
| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | CommCare user ID (unique identifier) |
| `name` | `str` | Human-readable FLW name |
| `cc_username` | `str` | CommCare username |
| `service_areas` | `List[str]` | List of service area IDs assigned to this FLW |

#### Performance Metrics
| Property | Type | Description |
|----------|------|-------------|
| `assigned_units` | `int` | Number of delivery units assigned |
| `completed_units` | `int` | Number of delivery units completed |
| `status_counts` | `Dict[str, int]` | Count of delivery units by status |

#### Activity Tracking
| Property | Type | Description |
|----------|------|-------------|
| `first_service_delivery_date` | `datetime` | Date of first service delivery |
| `last_service_delivery_date` | `datetime` | Date of last service delivery |
| `first_du_checkin` | `datetime` | Date of first delivery unit check-in |
| `last_du_checkin` | `datetime` | Date of last delivery unit check-in |
| `dates_active` | `List[datetime]` | List of unique dates when FLW was active |

#### Relationships
| Property | Type | Description |
|----------|------|-------------|
| `service_points` | `List[ServiceDeliveryPoint]` | List of service delivery points created by this FLW |
| `delivery_units` | `List[DeliveryUnit]` | List of delivery units assigned to this FLW |

#### Computed Properties
| Property | Type | Description |
|----------|------|-------------|
| `completion_rate` | `float` | Percentage of assigned units completed |
| `days_active` | `int` | Number of unique days the FLW was active |
| `delivery_units_completed_per_day` | `float` | Average delivery units completed per active day |

### ServiceDeliveryPoint Model

Represents a single service delivery event with GPS coordinates.

#### Core Properties
| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Visit ID (unique identifier for the service delivery event) |
| `latitude` | `float` | GPS latitude coordinate |
| `longitude` | `float` | GPS longitude coordinate |
| `flw_id` | `str` | Field Level Worker identifier |
| `flw_commcare_id` | `str` | CommCare ID of the FLW who made the delivery |
| `flw_cc_username` | `str` | CommCare username of the FLW |

#### Delivery Information
| Property | Type | Description |
|----------|------|-------------|
| `status` | `str` | Status of the service delivery |
| `du_name` | `str` | Name of the delivery unit where service was provided |
| `visit_date` | `str` | Date and time of the service delivery |

#### Quality Control
| Property | Type | Description |
|----------|------|-------------|
| `flagged` | `bool` | Whether the delivery point has been flagged for review |
| `flag_reason` | `str` | Reason for flagging (if applicable) |
| `accuracy_in_m` | `float` | GPS accuracy in meters |

#### Computed Properties
| Property | Type | Description |
|----------|------|-------------|
| `geometry` | `Point` | Shapely Point geometry object |

## Usage Examples

### Loading Coverage Data

```python
from src.models import CoverageData
from src.utils.data_loader import get_coverage_data_from_du_api_and_service_dataframe

# Load from API
coverage_data = get_coverage_data_from_du_api_and_service_dataframe(
    domain="your-commcare-domain",
    user="your-username", 
    api_key="your-api-key",
    service_df=service_delivery_dataframe
)

# Access the data
print(f"Total delivery units: {coverage_data.total_delivery_units}")
print(f"Completion rate: {coverage_data.completion_percentage:.1f}%")
print(f"Number of FLWs: {coverage_data.total_flws}")
```

### Computing Top-Level Indicators

Following the pattern from `opportunity_comparison_statistics.py`, you can compute indicators like:

```python
def compute_project_indicators(coverage_data: CoverageData) -> Dict[str, Any]:
    """Compute top-level indicators for a project"""
    
    # Basic coverage metrics
    total_dus = len(coverage_data.delivery_units)
    completed_dus = sum(1 for du in coverage_data.delivery_units.values() if du.status == 'completed')
    coverage_rate = (completed_dus / total_dus * 100) if total_dus > 0 else 0
    
    # FLW performance metrics
    active_flws = len([flw for flw in coverage_data.flws.values() if flw.days_active > 0])
    avg_completion_rate = sum(flw.completion_rate for flw in coverage_data.flws.values()) / len(coverage_data.flws)
    
    # Service area metrics
    completed_sas = sum(1 for sa in coverage_data.service_areas.values() if sa.is_completed)
    sa_completion_rate = (completed_sas / len(coverage_data.service_areas) * 100) if coverage_data.service_areas else 0
    
    # Quality metrics
    total_service_points = len(coverage_data.service_points)
    flagged_points = sum(1 for sp in coverage_data.service_points if sp.flagged)
    
    return {
        'total_delivery_units': total_dus,
        'completed_delivery_units': completed_dus,
        'coverage_rate_percent': coverage_rate,
        'total_service_areas': len(coverage_data.service_areas),
        'completed_service_areas': completed_sas,
        'service_area_completion_rate_percent': sa_completion_rate,
        'total_flws': len(coverage_data.flws),
        'active_flws': active_flws,
        'average_flw_completion_rate_percent': avg_completion_rate,
        'total_service_deliveries': total_service_points,
        'flagged_service_deliveries': flagged_points,
        'quality_rate_percent': ((total_service_points - flagged_points) / total_service_points * 100) if total_service_points > 0 else 0
    }
```

This data structure provides a comprehensive foundation for building coverage analysis, performance metrics, and comparison statistics across different opportunities and projects. 