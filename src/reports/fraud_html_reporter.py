import pandas as pd
import os
from datetime import datetime
import json

class FraudHTMLReporter:
    """
    Generate HTML dashboard for fraud investigation results.
    Start simple, iterate to add more features.
    """

    def __init__(self, fraud_results_df, raw_visit_data_df=None, baseline_data=None):
        """
        Initialize with fraud results and optionally raw visit data.
        
        Args:
            fraud_results_df: DataFrame from fraud detection (with BASELINE row removed)
            raw_visit_data_df: Optional raw visit data for detailed investigation
            baseline_data: Optional baseline statistics for investigation pages
        """
        # Store original DataFrame WITH baseline row for investigation pages
        self.fraud_results_with_baseline = fraud_results_df.copy()
        
        self.fraud_results = fraud_results_df.copy()
        self.raw_visits = raw_visit_data_df
        self.baseline_data = baseline_data  # NEW: Store baseline data
        
        # Remove baseline row if present (for main dashboard)
        self.fraud_results = self.fraud_results[
            self.fraud_results['flw_id'] != 'BASELINE VALUES'
        ].copy()
        
        self.fraud_results = self.fraud_results.sort_values(
            'fraud_composite_score', ascending=False
        ).reset_index(drop=True)

    def generate_investigation_pages(self, output_dir, top_n=25):
        """Generate individual investigation pages for top N FLWs"""
        from .flw_investigation_pages import FLWInvestigationPageGenerator
        
        # Pass the original DataFrame WITH baseline row
        generator = FLWInvestigationPageGenerator(self.fraud_results_with_baseline, self.baseline_data)
        return generator.generate_investigation_pages(output_dir, top_n)
    
    def generate_dashboard(self, output_path, top_n=50, title="Fraud Detection Dashboard"):
        """
        Generate complete HTML dashboard file.
        
        Args:
            output_path: Path where HTML file should be saved
            top_n: Number of top fraud cases to focus on
            title: Dashboard title
        """
        print(f"Generating HTML dashboard: {output_path}")
        
        # Prepare data
        top_flws = self._prepare_top_flws(top_n)
        summary_stats = self._calculate_summary_stats()
        
        # Generate HTML content
        html_content = self._generate_html_template(
            title=title,
            summary_stats=summary_stats,
            top_flws=top_flws
        )
        
        # Write to file with robust encoding handling
        with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to: {output_path}")
        print(f"Open in browser: file://{os.path.abspath(output_path)}")
        
        return output_path
    
    def _clean_text_for_html(self, text):
        """Clean text to ensure it's safe for HTML and UTF-8 encoding"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and handle encoding issues
        text_str = str(text)
        
        # Replace problematic characters
        text_str = text_str.encode('utf-8', errors='replace').decode('utf-8')
        
        # Escape HTML special characters
        text_str = text_str.replace('&', '&amp;')
        text_str = text_str.replace('<', '&lt;')
        text_str = text_str.replace('>', '&gt;')
        text_str = text_str.replace('"', '&quot;')
        text_str = text_str.replace("'", '&#x27;')
        
        return text_str
    
    def _prepare_top_flws(self, top_n):
        """Prepare top N FLWs for dashboard display"""
        
        # Get top N FLWs with valid fraud scores
        valid_scores = self.fraud_results.dropna(subset=['fraud_composite_score'])
        top_flws = valid_scores.head(top_n).copy()
        
        # Add derived columns for easier display
        top_flws['risk_level'] = top_flws['fraud_composite_score'].apply(self._categorize_risk)
        top_flws['risk_class'] = top_flws['fraud_composite_score'].apply(self._get_risk_css_class)
        
        # Get failed algorithms for each FLW
        top_flws['failed_algorithms'] = top_flws.apply(self._get_failed_algorithms, axis=1)
        
        # Clean text fields to prevent encoding issues
        if 'name' in top_flws.columns:
            top_flws['name'] = top_flws['name'].apply(self._clean_text_for_html)
        
        # Format dates
        if 'last_visit_date' in top_flws.columns:
            top_flws['last_visit_date'] = pd.to_datetime(
                top_flws['last_visit_date'], errors='coerce'
            ).dt.strftime('%Y-%m-%d')
        
        return top_flws
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics for dashboard header"""
        
        valid_scores = self.fraud_results.dropna(subset=['fraud_composite_score'])
        
        if len(valid_scores) == 0:
            return {
                'total_flws': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'avg_fraud_score': 0.0
            }
        
        stats = {
            'total_flws': len(valid_scores),
            'high_risk_count': len(valid_scores[valid_scores['fraud_composite_score'] >= 0.5]),
            'medium_risk_count': len(valid_scores[
                (valid_scores['fraud_composite_score'] >= 0.2) & 
                (valid_scores['fraud_composite_score'] < 0.5)
            ]),
            'low_risk_count': len(valid_scores[valid_scores['fraud_composite_score'] < 0.2]),
            'avg_fraud_score': round(valid_scores['fraud_composite_score'].mean(), 3)
        }
        
        return stats
    
    def _categorize_risk(self, fraud_score):
        """Categorize fraud score into risk levels"""
        if pd.isna(fraud_score):
            return "Unknown"
        elif fraud_score >= 0.5:
            return "High Risk"
        elif fraud_score >= 0.2:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _get_risk_css_class(self, fraud_score):
        """Get CSS class for risk level styling"""
        if pd.isna(fraud_score):
            return "risk-unknown"
        elif fraud_score >= 0.5:
            return "risk-high"
        elif fraud_score >= 0.2:
            return "risk-medium"
        else:
            return "risk-low"
    
    def _get_failed_algorithms(self, row):
        """Get list of failed fraud algorithms for display"""
        failed = []
        
        # Check individual fraud algorithm columns
        fraud_cols = [col for col in row.index if col.startswith('fraud_') 
                     and not col.endswith(('_score', '_percentile', '_category', '_weight', '_details'))]
        
        for col in fraud_cols:
            if pd.notna(row[col]) and row[col] > 0.3:  # Threshold for "failed"
                # Clean up algorithm name for display
                algo_name = col.replace('fraud_', '').replace('_', ' ').title()
                failed.append(algo_name)
        
        return ', '.join(failed) if failed else 'None'
    
    def _generate_html_template(self, title, summary_stats, top_flws):
        """Generate complete HTML dashboard"""
        
        # Convert top FLWs to JSON for JavaScript (with safe serialization)
        flw_data = []
        for _, row in top_flws.iterrows():
            row_dict = {}
            for col, val in row.items():
                # Clean all text values for JSON safety
                if isinstance(val, str):
                    row_dict[col] = self._clean_text_for_html(val)
                elif pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val
            flw_data.append(row_dict)
        
        # Clean title
        title_clean = self._clean_text_for_html(title)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_clean}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="dashboard-header">
            <h1>{title_clean}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>
        
        <!-- Summary Stats -->
        <section class="summary-section">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{summary_stats['total_flws']}</div>
                    <div class="stat-label">Total FLWs</div>
                </div>
                <div class="stat-card risk-high">
                    <div class="stat-number">{summary_stats['high_risk_count']}</div>
                    <div class="stat-label">High Risk</div>
                </div>
                <div class="stat-card risk-medium">
                    <div class="stat-number">{summary_stats['medium_risk_count']}</div>
                    <div class="stat-label">Medium Risk</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary_stats['avg_fraud_score']}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
            </div>
        </section>
        
        <!-- Controls -->
        <section class="controls-section">
            <div class="controls-grid">
                <input type="text" id="searchInput" placeholder="Search FLW name or ID..." class="search-input">
                <select id="riskFilter" class="filter-select">
                    <option value="">All Risk Levels</option>
                    <option value="High Risk">High Risk</option>
                    <option value="Medium Risk">Medium Risk</option>
                    <option value="Low Risk">Low Risk</option>
                </select>
                <button onclick="exportTop20()" class="export-btn">Export Top 20</button>
            </div>
        </section>
        
        <!-- FLW Table -->
        <section class="table-section">
            <table id="flwTable" class="flw-table">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">FLW Name ↕</th>
                        <th onclick="sortTable(1)">Fraud Score ↕</th>
                        <th onclick="sortTable(2)">Risk Level ↕</th>
                        <th onclick="sortTable(3)">Failed Tests ↕</th>
                        <th onclick="sortTable(4)">Visits ↕</th>
                        <th onclick="sortTable(5)">Last Visit ↕</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="flwTableBody">
                    {self._generate_table_rows(top_flws)}
                </tbody>
            </table>
        </section>
        
        <!-- Detail Panel (initially hidden) -->
        <section id="detailPanel" class="detail-panel" style="display: none;">
            <div class="detail-header">
                <h3 id="detailFLWName">FLW Details</h3>
                <button onclick="closeDetails()" class="close-btn">×</button>
            </div>
            <div id="detailContent" class="detail-content">
                <!-- Details will be populated by JavaScript -->
            </div>
        </section>
    </div>
    
    <script>
        // Embed data for JavaScript
        const flwData = {json.dumps(flw_data, default=str, indent=8, ensure_ascii=True)};
        
        {self._get_javascript_functions()}
    </script>
</body>
</html>"""
        
        return html_template
    
    def _generate_table_rows(self, top_flws):
        """Generate HTML table rows for FLWs"""
        rows = []
        
        for idx, row in top_flws.iterrows():
            fraud_score = row.get('fraud_composite_score', 0)
            fraud_score_display = f"{fraud_score:.3f}" if pd.notna(fraud_score) else "N/A"
            
            # Clean all text for HTML safety
            name_clean = self._clean_text_for_html(row.get('name', row['flw_id']))
            flw_id_clean = self._clean_text_for_html(row['flw_id'])
            risk_level_clean = self._clean_text_for_html(row.get('risk_level', 'Unknown'))
            failed_algorithms_clean = self._clean_text_for_html(row.get('failed_algorithms', 'None'))
            last_visit_clean = self._clean_text_for_html(row.get('last_visit_date', 'N/A'))
            
            # NEW: Create investigation page link (for top 25 only)
            if idx < 25:  # Only top 25 get investigation pages
                investigation_file = f"flw_investigation_{idx+1:03d}.html"
                name_link = f'<a href="{investigation_file}" class="flw-name-link"><strong>{name_clean}</strong></a>'
                action_button = f'<a href="{investigation_file}" class="action-btn">Investigate</a>'
            else:
                name_link = f'<strong>{name_clean}</strong>'
                action_button = '<span class="action-btn disabled">No Page</span>'
            
            row_html = f"""
                <tr class="{row.get('risk_class', '')}" onclick="showDetails('{flw_id_clean}')">
                    <td>{name_link}</td>
                    <td>{fraud_score_display}</td>
                    <td><span class="risk-badge {row.get('risk_class', '')}">{risk_level_clean}</span></td>
                    <td>{failed_algorithms_clean}</td>
                    <td>{row.get('n_visits', 'N/A')}</td>
                    <td>{last_visit_clean}</td>
                    <td onclick="event.stopPropagation();">{action_button}</td>
                </tr>"""
            rows.append(row_html)
        
        return '\n'.join(rows)
    
    def _get_css_styles(self):
        """Return CSS styles for the dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .dashboard-header h1 {
            color: #2c3e50;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .summary-section {
            margin-bottom: 30px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }
        
        .stat-card.risk-high {
            border-left-color: #e74c3c;
        }
        
        .stat-card.risk-medium {
            border-left-color: #f39c12;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .controls-section {
            margin-bottom: 20px;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 15px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .search-input, .filter-select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .export-btn, .action-btn {
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .export-btn:hover, .action-btn:hover {
            background: #2980b9;
        }
        
        .action-btn.disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            color: #7f8c8d;
        }
        
        .action-btn.disabled:hover {
            background: #bdc3c7;
        }
        
        .flw-name-link {
            color: #3498db;
            text-decoration: none;
        }
        
        .flw-name-link:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        .table-section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .flw-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .flw-table th {
            background: #f8f9fa;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            cursor: pointer;
            user-select: none;
        }
        
        .flw-table th:hover {
            background: #e9ecef;
        }
        
        .flw-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .flw-table tbody tr {
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .flw-table tbody tr:hover {
            background: #f8f9fa;
        }
        
        .risk-high {
            border-left: 4px solid #e74c3c;
        }
        
        .risk-medium {
            border-left: 4px solid #f39c12;
        }
        
        .risk-low {
            border-left: 4px solid #27ae60;
        }
        
        .risk-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .risk-badge.risk-high {
            background: #fee;
            color: #e74c3c;
        }
        
        .risk-badge.risk-medium {
            background: #fef9e7;
            color: #f39c12;
        }
        
        .risk-badge.risk-low {
            background: #eef;
            color: #27ae60;
        }
        
        .detail-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            max-width: 800px;
            max-height: 80%;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            overflow: hidden;
        }
        
        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #999;
        }
        
        .detail-content {
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        @media (max-width: 768px) {
            .controls-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .flw-table {
                font-size: 14px;
            }
            
            .detail-panel {
                width: 95%;
                height: 90%;
            }
        }
        """
    
    def _get_javascript_functions(self):
        """Return JavaScript functions for interactivity"""
        return """
        // Global variables
        let sortDirection = {};
        let filteredData = [...flwData];
        
        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {
            filterTable();
        });
        
        // Risk filter
        document.getElementById('riskFilter').addEventListener('change', function(e) {
            filterTable();
        });
        
        function filterTable() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const riskFilter = document.getElementById('riskFilter').value;
            
            filteredData = flwData.filter(row => {
                const matchesSearch = !searchTerm || 
                    (row.name && row.name.toLowerCase().includes(searchTerm)) ||
                    (row.flw_id && row.flw_id.toLowerCase().includes(searchTerm));
                
                const matchesRisk = !riskFilter || row.risk_level === riskFilter;
                
                return matchesSearch && matchesRisk;
            });
            
            updateTable();
        }
        
        function updateTable() {
            const tbody = document.getElementById('flwTableBody');
            tbody.innerHTML = '';
            
            filteredData.forEach((row, idx) => {
                const tr = document.createElement('tr');
                tr.className = row.risk_class || '';
                tr.onclick = () => showDetails(row.flw_id);
                
                const fraudScore = row.fraud_composite_score !== null ? 
                    row.fraud_composite_score.toFixed(3) : 'N/A';
                
                // Create investigation link for top 25
                let nameDisplay, actionButton;
                if (idx < 25) {
                    const investigationFile = `flw_investigation_${String(idx + 1).padStart(3, '0')}.html`;
                    nameDisplay = `<a href="${investigationFile}" class="flw-name-link"><strong>${row.name || row.flw_id}</strong></a>`;
                    actionButton = `<a href="${investigationFile}" class="action-btn">Investigate</a>`;
                } else {
                    nameDisplay = `<strong>${row.name || row.flw_id}</strong>`;
                    actionButton = '<span class="action-btn disabled">No Page</span>';
                }
                
                tr.innerHTML = `
                    <td>${nameDisplay}</td>
                    <td>${fraudScore}</td>
                    <td><span class="risk-badge ${row.risk_class || ''}">${row.risk_level || 'Unknown'}</span></td>
                    <td>${row.failed_algorithms || 'None'}</td>
                    <td>${row.n_visits || 'N/A'}</td>
                    <td>${row.last_visit_date || 'N/A'}</td>
                    <td onclick="event.stopPropagation();">${actionButton}</td>
                `;
                
                tbody.appendChild(tr);
            });
        }
        
        function sortTable(columnIndex) {
            const columns = ['name', 'fraud_composite_score', 'risk_level', 'failed_algorithms', 'n_visits', 'last_visit_date'];
            const column = columns[columnIndex];
            
            if (!sortDirection[column]) sortDirection[column] = 'asc';
            else sortDirection[column] = sortDirection[column] === 'asc' ? 'desc' : 'asc';
            
            filteredData.sort((a, b) => {
                let aVal = a[column];
                let bVal = b[column];
                
                // Handle nulls
                if (aVal === null || aVal === undefined) aVal = '';
                if (bVal === null || bVal === undefined) bVal = '';
                
                // Convert to numbers if possible
                if (!isNaN(aVal) && !isNaN(bVal)) {
                    aVal = parseFloat(aVal);
                    bVal = parseFloat(bVal);
                }
                
                if (sortDirection[column] === 'asc') {
                    return aVal > bVal ? 1 : -1;
                } else {
                    return aVal < bVal ? 1 : -1;
                }
            });
            
            updateTable();
        }
        
        function showDetails(flwId) {
            const flw = flwData.find(f => f.flw_id === flwId);
            if (!flw) return;
            
            document.getElementById('detailFLWName').textContent = `${flw.name || flw.flw_id} - Details`;
            
            const detailsHtml = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4>Basic Info</h4>
                        <p><strong>FLW ID:</strong> ${flw.flw_id || 'N/A'}</p>
                        <p><strong>Name:</strong> ${flw.name || 'N/A'}</p>
                        <p><strong>Fraud Score:</strong> ${flw.fraud_composite_score ? flw.fraud_composite_score.toFixed(3) : 'N/A'}</p>
                        <p><strong>Risk Level:</strong> ${flw.risk_level || 'Unknown'}</p>
                        <p><strong>Total Visits:</strong> ${flw.n_visits || 'N/A'}</p>
                        <p><strong>Last Visit:</strong> ${flw.last_visit_date || 'N/A'}</p>
                    </div>
                    <div>
                        <h4>Failed Tests</h4>
                        <p>${flw.failed_algorithms || 'None'}</p>
                        <br>
                        <h4>MUAC Details</h4>
                        <p>${flw.fraud_muac_distribution_details || 'No MUAC data available'}</p>
                    </div>
                </div>
            `;
            
            document.getElementById('detailContent').innerHTML = detailsHtml;
            document.getElementById('detailPanel').style.display = 'block';
        }
        
        function closeDetails() {
            document.getElementById('detailPanel').style.display = 'none';
        }
        
        function investigateFLW(flwId) {
            alert(`Investigation workflow for FLW ${flwId} would start here.\\n\\nIn a full implementation, this could:\\n- Generate investigation checklist\\n- Export FLW-specific data\\n- Mark as "Under Investigation"\\n- Send to field team`);
        }
        
        function exportTop20() {
            const top20 = filteredData.slice(0, 20);
            const csvContent = convertToCSV(top20);
            downloadCSV(csvContent, 'top_20_fraud_cases.csv');
        }
        
        function convertToCSV(data) {
            const headers = ['FLW ID', 'Name', 'Fraud Score', 'Risk Level', 'Failed Tests', 'Visits', 'Last Visit'];
            const rows = data.map(row => [
                row.flw_id || '',
                row.name || '',
                row.fraud_composite_score || '',
                row.risk_level || '',
                row.failed_algorithms || '',
                row.n_visits || '',
                row.last_visit_date || ''
            ]);
            
            return [headers, ...rows].map(row => 
                row.map(field => `"${field}"`).join(',')
            ).join('\\n');
        }
        
        function downloadCSV(csvContent, fileName) {
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            a.click();
            window.URL.revokeObjectURL(url);
        }
        
        // Close detail panel when clicking outside
        document.addEventListener('click', function(e) {
            const detailPanel = document.getElementById('detailPanel');
            if (e.target === detailPanel) {
                closeDetails();
            }
        });
        
        // Initialize table
        filterTable();
        """

# Example usage function
def create_fraud_dashboard_example(fraud_results_csv_path, output_html_path):
    """
    Example function showing how to use the FraudHTMLReporter.
    
    Args:
        fraud_results_csv_path: Path to fraud results CSV file
        output_html_path: Where to save the HTML dashboard
    """
    # Load fraud results
    fraud_df = pd.read_csv(fraud_results_csv_path)
    
    # Create reporter and generate dashboard
    reporter = FraudHTMLReporter(fraud_df)
    
    dashboard_path = reporter.generate_dashboard(
        output_path=output_html_path,
        top_n=50,
        title="FLW Fraud Detection Dashboard"
    )
    
    return dashboard_path

if __name__ == "__main__":
    # Example usage
    print("Fraud HTML Reporter - Ready for integration")
    print("Example usage:")
    print("  reporter = FraudHTMLReporter(fraud_results_df)")
    print("  reporter.generate_dashboard('fraud_dashboard.html')")