### Goal
Build a self-contained Leaflet HTML map to showcase CommCare water points with compelling styling and image-rich popups, loading records from an Excel file in `water_data/` and images from the same directory.

### Constraints and alignment
- Reuse the same mapping approach as existing code: Leaflet via CDN, single HTML output, no external build step.
- Keep output as a single portable HTML file (images embedded as base64 thumbnails in popups to ensure portability).
- Follow code organization patterns in `src/create_delivery_map.py` (inline CSS/JS, JSON data embedded in the page).

### Data inputs
**Project Structure**: Multiple water survey projects in `water_data/`, each with:
- Excel file: `{PROJECT} CCC Waterbody Survey - August 7.xlsx` (one row per water point)
- Image folder: `{PROJECT} Pics-Aug 7/` containing water point photos

**Current Projects**:
- `COWACDI` - 56 water points in Borno state (Mafa LGA)
- `CWINS` - 85 water points in Niger state (Chanchaga LGA)

**Excel File Structure** (25 columns, identical across projects):
- **Core identifiers**: `number`, `formid` (UUID), `username` (data collector ID)
- **Location hierarchy**: `form.state`, `form.lga`, `form.ward`, `form.community`
- **Coordinates**: `form.gps` (format: "lat lon altitude accuracy" - space-separated)
- **Images**: `form.photo1`, `form.photo2`, `form.photo3` (CommCare API URLs)
- **Water point details**: 
  - `form.water_point_type` (e.g., "piped_water", "borehole_hand_pump", "protected_wells")
  - `form.is_piped`, `form.has_dispenser`
  - `form.if_available_is_the_chlorine_dispenser_functional`
  - `form.other_treatment`, `form.usage_level`, `form.hh_estimate`
- **Metadata**: `form.time_of_visit`, `form.notes`, `completed_time`, `started_time`, `received_on`
- **CommCare links**: `form_link`, `hq_user`

**Image Naming Convention**:
- Pattern: `photo{1-3}-{username}-form_{formid}.jpg`
- Examples: `photo1-cowws1-form_08e8640b-ffff-4d52-8f75-169ae1ea721c.jpg`
- Usernames match project: COWACDI uses `cowws1`, `cowws2`, etc.; CWINS uses `cwws1`, `cwws2`, etc.

### UX and styling
- Modern dark-on-light basemap (OSM default) with subtle shadows and a scale bar.
- Compact, elegant popup layout: title, key facts, image carousel (thumbnails with click-to-expand full-size in a lightbox-style overlay).
- Marker design: small, high-contrast circular markers with white stroke and hover glow; color palette driven by an optional status/type column if present.
- Optional controls: search box (by name/id), fullscreen toggle, locate control, legend block (if status/type found).

### Features (phased)
1) MVP
   - Parse GPS coordinates from space-separated format in `form.gps` column
   - Generate markers and popups with location hierarchy (state > LGA > ward > community)
   - Link and embed 1-3 images per water point using formid-based filename matching
   - Display water point characteristics (type, piped status, dispenser info, usage level)
   - Output single HTML file under `water_map_output_{timestamp}/water_points_map_{timestamp}.html`

2) Enhanced interactivity
   - Marker clustering for large datasets (141 total points across both projects)
   - Client-side search by community name, water point type, or usage level
   - Project-based filtering (COWACDI vs CWINS) with distinct marker colors
   - Responsive image gallery with keyboard navigation and lightbox overlay

3) Polish
   - Color-coded markers by water point type (piped_water, borehole_hand_pump, protected_wells)
   - Custom styling for location hierarchy breadcrumbs in popups
   - Dynamic legend showing water point types and project distribution
   - Household estimate aggregation display

### Implementation outline

**Data Model** - `waterpoint.py`:
```python
@dataclass
class WaterPoint:
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
    altitude: Optional[float]
    accuracy: Optional[float]
    
    # Water point characteristics
    water_point_type: str  # piped_water, borehole_hand_pump, protected_wells, etc.
    is_piped: bool
    has_dispenser: bool
    chlorine_dispenser_functional: Optional[bool]
    other_treatment: Optional[str]
    usage_level: str  # high, medium, low
    household_estimate: Optional[int]
    
    # Survey metadata
    time_of_visit: datetime
    notes: Optional[str]
    completed_time: datetime
    started_time: datetime
    received_on: datetime
    
    # Images (local file paths after processing)
    photo1_path: Optional[str]
    photo2_path: Optional[str] 
    photo3_path: Optional[str]
    
    # CommCare references
    form_link: str
    hq_user: str
    
    @property
    def project_name(self) -> str:
        """Extract project name from username prefix"""
        if self.username.startswith('cowws'):
            return 'COWACDI'
        elif self.username.startswith('cwws'):
            return 'CWINS'
        return 'UNKNOWN'
```

**Main Implementation** - `create_water_map.py`:
- `create_water_points_map(project_name: Optional[str] = None, output_dir: Optional[str] = None) -> str`
- Auto-detect projects if not specified; process single project or all projects
- Parse GPS coordinates from space-separated format: "lat lon altitude accuracy"
- Link images using exact filename pattern: `photo{1-3}-{username}-form_{formid}.jpg`
- Build WaterPoint objects and serialize to JSON for embedding in HTML
- Copy original images to output directory for fast, reliable access
- Generate single-file HTML with Leaflet CSS/JS from CDN and inline styling
- Auto-open generated map in default browser

### Open questions / assumptions
- **Multi-project handling**: Should the map display both projects simultaneously with filtering options, or generate separate maps for each project?
- **Image optimization**: What thumbnail size works best for popup performance vs. quality? (suggested: 512px width)
- **Popup content priority**: Which fields should be prominently displayed vs. in expandable details?
- **Color scheme**: Should markers be colored by project (COWACDI vs CWINS) or by water point type?
- **Missing data handling**: Some water points may have broken image links or missing photos - how to handle gracefully?

### Next steps
- Implement `waterpoint.py` data model with GPS parsing and image path resolution
- Create `create_water_map.py` MVP to process both projects and generate interactive HTML map
- Test with current dataset (141 total water points: 56 COWACDI + 85 CWINS)

