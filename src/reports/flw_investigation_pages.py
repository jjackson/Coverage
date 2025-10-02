import pandas as pd
import os
from datetime import datetime

class FLWInvestigationPageGenerator:
    """
    Generate individual investigation pages for high-risk FLWs.
    Creates detailed fraud analysis pages with summary tables and navigation.
    """
    
    def __init__(self, fraud_results_df, baseline_data=None):
        """
        Initialize with fraud results and baseline data.
        
        Args:
            fraud_results_df: DataFrame from fraud detection (with BASELINE row)
            baseline_data: Optional baseline statistics for comparison
        """
        self.fraud_results = fraud_results_df.copy()
        self.baseline_data = baseline_data
        
        # Extract baseline row BEFORE removing it
        baseline_mask = self.fraud_results['flw_id'] == 'BASELINE VALUES'
        self.baseline_row = self.fraud_results[baseline_mask].iloc[0] if baseline_mask.sum() > 0 else None
        
        # NOW remove baseline row from main results
        self.fraud_results = self.fraud_results[
            self.fraud_results['flw_id'] != 'BASELINE VALUES'
        ].copy()
        
        # Sort by fraud score (highest first) - compatible with older pandas
        self.fraud_results = self.fraud_results.sort_values(
            'fraud_composite_score', ascending=False
        ).reset_index(drop=True)
    
    def generate_investigation_pages(self, output_dir, top_n=25):
        """
        Generate investigation pages for top N high-risk FLWs.
        
        Args:
            output_dir: Directory to save investigation pages
            top_n: Number of top FLWs to generate pages for
            
        Returns:
            List of generated file paths
        """
        print(f"Generating investigation pages for top {top_n} FLWs...")
        
        # Get top N FLWs with valid fraud scores
        valid_scores = self.fraud_results.dropna(subset=['fraud_composite_score'])
        top_flws = valid_scores.head(top_n)
        
        generated_files = []
        
        for idx, (_, flw_row) in enumerate(top_flws.iterrows(), 1):
            filename = f"flw_investigation_{idx:03d}.html"
            filepath = os.path.join(output_dir, filename)
            
            # Generate page content
            html_content = self._generate_investigation_page(
                flw_row, 
                rank=idx, 
                total_pages=len(top_flws)
            )
            
            # Write HTML file
            with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
                f.write(html_content)
            
            generated_files.append(filepath)
            
            if idx % 5 == 0:
                print(f"Generated {idx}/{len(top_flws)} investigation pages...")
        
        print(f"Investigation pages complete: {len(generated_files)} files")
        return generated_files
    
    def _generate_investigation_page(self, flw_row, rank, total_pages):
        """Generate HTML content for individual FLW investigation page"""
        
        flw_id = flw_row['flw_id']
        flw_name = flw_row.get('name', flw_id)
        fraud_score = flw_row.get('fraud_composite_score', 0)
        risk_level = self._get_risk_level(fraud_score)
        risk_class = self._get_risk_css_class(fraud_score)
        
        # Generate fraud analysis table
        fraud_table_html = self._generate_fraud_analysis_table(flw_row)
        
        # Generate navigation
        nav_html = self._generate_navigation(rank, total_pages)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investigation: {self._clean_text_for_html(flw_name)}</title>
    <style>
        {self._get_investigation_css()}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="investigation-header">
            <div class="header-main">
                <h1>FLW Investigation Report</h1>
                <div class="flw-info">
                    <div class="flw-name">{self._clean_text_for_html(flw_name)}</div>
                    <div class="flw-id">ID: {self._clean_text_for_html(flw_id)}</div>
                    <div class="fraud-score-badge {risk_class}">
                        Fraud Score: {fraud_score:.3f} ({risk_level})
                    </div>
                </div>
            </div>
            <div class="header-meta">
                <span>Rank: #{rank} of {total_pages}</span>
                <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </header>

        <!-- Navigation -->
        {nav_html}

        <!-- Fraud Analysis Section -->
        <section class="analysis-section">
            <h2>Fraud Algorithm Analysis</h2>
            <div class="table-container">
                {fraud_table_html}
            </div>
        </section>

        <!-- Placeholder for future sections -->
        <section class="future-section">
            <h2>MUAC & Age Distribution Analysis</h2>
            <p class="placeholder">Charts and detailed analysis will be added in next iteration.</p>
        </section>

        <!-- Footer Navigation -->
        {nav_html}
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_fraud_analysis_table(self, flw_row):
        """Generate the fraud analysis summary table grouped by field"""
        
        # Define field mappings based on actual columns in the data
        # Format: (field_id, pattern_col, completion_col, field_name, actual_col, baseline_col, completion_rate_col)
        field_mappings = [
            # Single Pattern Fields (completion_col = None)
            ('gender', 'fraud_gender_sva', None, 'childs_gender', 'pct_female', 'pct_female', 'pct_childs_gender_blank'),
            ('age_imbalance', 'fraud_yearly_age_imbalance', None, 'childs_age_in_month', None, None, 'pct_childs_age_in_month_blank'),
            ('age_perfection', 'fraud_monthly_age_perfection', None, 'childs_age_in_month (perfection)', None, None, 'pct_childs_age_in_month_blank'),
            ('va_child_unwell_today', 'fraud_va_child_unwell_today_sva', 'fraud_va_child_unwell_today_completion_anomaly', 'va_child_unwell_today', 'pct_va_child_unwell_today_yes', 'pct_va_child_unwell_today_yes', 'pct_va_child_unwell_today_blank'),
            ('unwell_completion', 'fraud_unwell_completion_anomaly', None, 'va_child_unwell_today (completion)', None, None, 'pct_va_child_unwell_today_blank'),
            
            # Dual Indicator Fields (both pattern and completion)
            ('glasses', 'fraud_glasses_sva', 'fraud_glasses_completion_anomaly', 'have_glasses', 'pct_have_glasses_yes', 'pct_have_glasses_yes', 'pct_have_glasses_blank'),
            ('diarrhea', 'fraud_diarrhea_sva', 'fraud_diarrhea_completion_anomaly', 'diarrhea_last_month', 'pct_diarrhea_last_month_yes', 'pct_diarrhea_last_month_yes', 'pct_diarrhea_last_month_blank'),
            ('malnutrition', 'fraud_mal_sva', 'fraud_mal_completion_anomaly', 'diagnosed_with_mal_past_3_months', 'pct_diagnosed_with_mal_past_3_months_yes', 'pct_diagnosed_with_mal_past_3_months_yes', 'pct_diagnosed_with_mal_past_3_months_blank'),
            ('muac_colour', 'fraud_muac_colour_sva', 'fraud_muac_colour_completion_anomaly', 'muac_colour', None, None, 'pct_muac_colour_blank'),
            ('received_va_dose_before', 'fraud_received_va_dose_before_sva', 'fraud_received_va_dose_before_completion_anomaly', 'received_va_dose_before', None, None, 'pct_received_va_dose_before_blank'),
            ('recent_va_dose', 'fraud_recent_va_dose_sva', 'fraud_recent_va_dose_completion_anomaly', 'recent_va_dose', None, None, 'pct_recent_va_dose_blank'),
            ('any_vaccine', 'fraud_received_any_vaccine_sva', 'fraud_received_any_vaccine_completion_anomaly', 'received_any_vaccine', 'pct_received_any_vaccine_yes', 'pct_received_any_vaccine_yes', 'pct_received_any_vaccine_blank'),
            ('under_treatment', 'fraud_under_treatment_for_mal_sva', 'fraud_under_treatment_for_mal_completion_anomaly', 'under_treatment_for_mal', 'pct_under_treatment_for_mal_yes', 'pct_under_treatment_for_mal_yes', 'pct_under_treatment_for_mal_blank'),
        ]
        
        # Process each field and collect data
        field_data = []
        
        for field_id, pattern_col, completion_col, field_name, actual_col, baseline_col, completion_rate_col in field_mappings:
            # Get pattern score
            pattern_score = flw_row.get(pattern_col, None) if pattern_col in flw_row.index else None
            
            # Get completion score  
            completion_score = flw_row.get(completion_col, None) if completion_col and completion_col in flw_row.index else None
            
            # Skip fields where both scores are missing
            if pd.isna(pattern_score) and pd.isna(completion_score):
                continue
                
            # Calculate max score for sorting
            scores_for_sort = []
            if pd.notna(pattern_score):
                scores_for_sort.append(pattern_score)
            if pd.notna(completion_score):
                scores_for_sort.append(completion_score)
            max_score = max(scores_for_sort) if scores_for_sort else 0
            
            # Get actual value
            if actual_col and actual_col in flw_row.index:
                actual_value = flw_row.get(actual_col, None)
                if 'age' in field_id:
                    actual_display = "See chart"
                elif 'muac_dist' in field_id:
                    actual_display = "See chart"
                elif pd.notna(actual_value):
                    actual_display = f"{actual_value:.1%}"
                else:
                    actual_display = "N/A"
            else:
                if 'age' in field_id or 'muac' in field_id:
                    actual_display = "See chart"
                else:
                    actual_display = "N/A"
            
            # Get baseline value
            baseline_display = "N/A"  # Default
            
            if baseline_col and self.baseline_row is not None:
                if baseline_col in self.baseline_row.index:
                    baseline_value = self.baseline_row.get(baseline_col, None)
                    if 'age' in field_id:
                        baseline_display = "Even distribution"
                    elif 'muac_distribution' in field_id:
                        baseline_display = "Bell curve"
                    elif pd.notna(baseline_value) and baseline_value != 0:
                        baseline_display = f"{baseline_value:.1%}"
                    else:
                        baseline_display = "0.0%"
                else:
                    # Column doesn't exist in baseline row
                    if 'age' in field_id:
                        baseline_display = "Even distribution"
                    elif 'muac_distribution' in field_id:
                        baseline_display = "Bell curve"
            else:
                # No baseline row or baseline column
                if 'age' in field_id:
                    baseline_display = "Even distribution"
                elif 'muac_distribution' in field_id:
                    baseline_display = "Bell curve"
            
            # Get completion rate
            if completion_rate_col and completion_rate_col in flw_row.index:
                completion_blank = flw_row.get(completion_rate_col, None)
                if pd.notna(completion_blank):
                    completion_rate = 1 - completion_blank
                    completion_rate_display = f"{completion_rate:.1%}"
                else:
                    completion_rate_display = "N/A"
            else:
                completion_rate_display = "N/A"
            
            # Generate issue description (based on pattern score only)
            issue_description = self._generate_field_issue_description(
                field_name, pattern_score, actual_value if actual_col and actual_col in flw_row.index else None,
                baseline_value if baseline_col and self.baseline_row is not None and baseline_col in self.baseline_row.index else None,
                field_id, flw_row
            )
            
            field_data.append({
                'field_name': field_name,  # Use actual field name instead of display_name
                'pattern_score': pattern_score,
                'completion_score': completion_score,
                'max_score': max_score,
                'actual_display': actual_display,
                'baseline_display': baseline_display,
                'completion_rate_display': completion_rate_display
            })
        
        # Sort by max score (highest first)
        field_data.sort(key=lambda x: x['max_score'], reverse=True)
        
        # Build table rows
        table_rows = []
        
        for field in field_data:
            # Format scores
            pattern_score_display = f"{field['pattern_score']:.3f}" if pd.notna(field['pattern_score']) else "N/A"
            completion_score_display = f"{field['completion_score']:.3f}" if pd.notna(field['completion_score']) else "N/A"
            
            # Get CSS classes
            pattern_class = self._get_score_css_class(field['pattern_score'])
            completion_class = self._get_score_css_class(field['completion_score'])
            
            # Row class based on max score
            row_class = self._get_score_css_class(field['max_score'])
            
            table_rows.append(f"""
                <tr class="score-row {row_class}">
                    <td class="field-name">{field['field_name']}</td>
                    <td class="fraud-score {pattern_class}">{pattern_score_display}</td>
                    <td class="fraud-score {completion_class}">{completion_score_display}</td>
                    <td class="actual-value">{field['actual_display']}</td>
                    <td class="baseline-value">{field['baseline_display']}</td>
                    <td class="completion-rate">{field['completion_rate_display']}</td>
                </tr>
            """)
        
        # Combine into full table
        table_html = f"""
            <table class="fraud-analysis-table">
                <thead>
                    <tr>
                        <th>Field</th>
                        <th>Pattern Score</th>
                        <th>Completion Score</th>
                        <th>Actual Value</th>
                        <th>Baseline Expected</th>
                        <th>Completion Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        """
        
        return table_html
    
    def _generate_field_issue_description(self, field_name, pattern_score, actual_value, baseline_value, field_id, flw_row):
        """Generate human-readable issue description based on pattern score"""
        
        if pd.isna(pattern_score) or pattern_score is None:
            return "No pattern data"
        
        if pattern_score < 0.3:
            return "Normal pattern"
        
        # Field-specific descriptions
        if 'gender' in field_id and pd.notna(actual_value) and pd.notna(baseline_value):
            if actual_value > 0.7:
                return f"Extreme female bias ({actual_value:.0%} vs expected {baseline_value:.0%})"
            elif actual_value < 0.3:
                return f"Extreme male bias ({actual_value:.0%} vs expected {baseline_value:.0%})"
            else:
                return f"Gender imbalance ({actual_value:.0%} vs expected {baseline_value:.0%})"
        
        elif 'age' in field_id:
            if pattern_score > 0.7:
                return "Severe age distribution anomaly"
            else:
                return "Age distribution imbalance"
        
        elif 'muac_dist' in field_id:
            # Try to get MUAC details
            muac_details = flw_row.get('fraud_muac_distribution_details', '')
            if muac_details and 'failed_' in str(muac_details):
                return f"MUAC pattern issues: {muac_details}"
            else:
                return "MUAC distribution anomaly"
        
        elif 'unwell_overreport' in field_id:
            if pd.notna(actual_value):
                return f"Unusually high unwell rate ({actual_value:.0%})"
            else:
                return "Unwell overreporting detected"
        
        elif pd.notna(actual_value) and pd.notna(baseline_value):
            # Generic yes/no field
            if actual_value > baseline_value * 2:
                return f"Unusually high rate ({actual_value:.0%} vs expected {baseline_value:.0%})"
            elif actual_value < baseline_value * 0.5:
                return f"Unusually low rate ({actual_value:.0%} vs expected {baseline_value:.0%})"
            else:
                return f"Moderate deviation from baseline"
        
        # Generic fallback
        if pattern_score > 0.7:
            return "High fraud risk detected"
        elif pattern_score > 0.5:
            return "Moderate fraud risk"
        else:
            return "Minor anomaly detected"
    
    def _generate_issue_description(self, algo_name, fraud_score, actual_value, baseline_value, flw_row, algo_config):
        """Generate human-readable issue description"""
        
        if pd.isna(fraud_score) or fraud_score is None:
            return "No data available"
        
        if fraud_score < 0.3:
            return "Normal pattern"
        
        # Specific descriptions based on algorithm type
        if algo_name == "Gender Distribution" and pd.notna(actual_value) and pd.notna(baseline_value):
            if actual_value > 0.7:
                return f"Extreme female bias ({actual_value:.0%} vs expected {baseline_value:.0%})"
            elif actual_value < 0.3:
                return f"Extreme male bias ({actual_value:.0%} vs expected {baseline_value:.0%})"
            else:
                return f"Moderate gender imbalance ({actual_value:.0%} vs expected {baseline_value:.0%})"
        
        elif "Age" in algo_name:
            if fraud_score > 0.7:
                return "Severe age distribution anomaly"
            else:
                return "Moderate age distribution imbalance"
        
        elif algo_name == "MUAC Distribution":
            # Try to get MUAC details
            muac_details = flw_row.get('fraud_muac_distribution_details', '')
            if muac_details and 'failed_' in str(muac_details):
                return f"MUAC pattern issues: {muac_details}"
            else:
                return "MUAC distribution anomaly detected"
        
        elif algo_name in ["Glasses Field", "Diarrhea Field", "Malnutrition Diagnosis", "Unwell Reporting"]:
            if pd.notna(actual_value) and pd.notna(baseline_value):
                if actual_value > baseline_value * 2:
                    return f"Unusually high rate ({actual_value:.0%} vs expected {baseline_value:.0%})"
                elif actual_value < baseline_value * 0.5:
                    return f"Unusually low rate ({actual_value:.0%} vs expected {baseline_value:.0%})"
                else:
                    return f"Moderate deviation from baseline"
            else:
                return "Response pattern anomaly"
        
        # Generic fallback
        if fraud_score > 0.7:
            return "High fraud risk detected"
        elif fraud_score > 0.5:
            return "Moderate fraud risk"
        else:
            return "Minor anomaly detected"
    
    def _generate_navigation(self, current_rank, total_pages):
        """Generate navigation controls"""
        
        prev_link = ""
        next_link = ""
        
        if current_rank > 1:
            prev_file = f"flw_investigation_{current_rank-1:03d}.html"
            prev_link = f'<a href="{prev_file}" class="nav-btn prev-btn">? Previous FLW</a>'
        
        if current_rank < total_pages:
            next_file = f"flw_investigation_{current_rank+1:03d}.html"
            next_link = f'<a href="{next_file}" class="nav-btn next-btn">Next FLW ?</a>'
        
        dashboard_link = '<a href="fraud_dashboard.html" class="nav-btn dashboard-btn">? Back to Dashboard</a>'
        
        nav_html = f"""
            <nav class="page-navigation">
                <div class="nav-left">
                    {prev_link}
                </div>
                <div class="nav-center">
                    {dashboard_link}
                    <span class="page-indicator">Page {current_rank} of {total_pages}</span>
                </div>
                <div class="nav-right">
                    {next_link}
                </div>
            </nav>
        """
        
        return nav_html
    
    def _get_risk_level(self, fraud_score):
        """Get risk level description"""
        if pd.isna(fraud_score):
            return "Unknown"
        elif fraud_score >= 0.7:
            return "High Risk"
        elif fraud_score >= 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _get_risk_css_class(self, fraud_score):
        """Get CSS class for risk level styling"""
        if pd.isna(fraud_score):
            return "risk-unknown"
        elif fraud_score >= 0.7:
            return "risk-high"
        elif fraud_score >= 0.3:
            return "risk-medium"
        else:
            return "risk-low"
    
    def _get_score_css_class(self, score):
        """Get CSS class for individual fraud score"""
        if pd.isna(score):
            return "score-na"
        elif score >= 0.7:
            return "score-high"
        elif score >= 0.3:
            return "score-medium"
        else:
            return "score-low"
    
    def _clean_text_for_html(self, text):
        """Clean text to ensure it's safe for HTML"""
        if pd.isna(text) or text is None:
            return ""
        
        text_str = str(text)
        text_str = text_str.encode('utf-8', errors='replace').decode('utf-8')
        
        # Escape HTML special characters
        text_str = text_str.replace('&', '&amp;')
        text_str = text_str.replace('<', '&lt;')
        text_str = text_str.replace('>', '&gt;')
        text_str = text_str.replace('"', '&quot;')
        text_str = text_str.replace("'", '&#x27;')
        
        return text_str
    
    def _get_investigation_css(self):
        """Return CSS styles for investigation pages"""
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
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header Styles */
        .investigation-header {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .header-main {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        
        .investigation-header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .flw-info {
            text-align: right;
        }
        
        .flw-name {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .flw-id {
            color: #7f8c8d;
            margin-bottom: 10px;
        }
        
        .fraud-score-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }
        
        .fraud-score-badge.risk-high {
            background: #e74c3c;
        }
        
        .fraud-score-badge.risk-medium {
            background: #f39c12;
        }
        
        .fraud-score-badge.risk-low {
            background: #27ae60;
        }
        
        .header-meta {
            display: flex;
            justify-content: space-between;
            color: #7f8c8d;
            font-size: 14px;
            border-top: 1px solid #ecf0f1;
            padding-top: 15px;
        }
        
        /* Navigation Styles */
        .page-navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .nav-left, .nav-right {
            flex: 1;
        }
        
        .nav-center {
            flex: 2;
            text-align: center;
        }
        
        .nav-right {
            text-align: right;
        }
        
        .nav-btn {
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
            margin: 0 5px;
            transition: background-color 0.2s;
        }
        
        .nav-btn.prev-btn, .nav-btn.next-btn {
            background: #3498db;
            color: white;
        }
        
        .nav-btn.dashboard-btn {
            background: #95a5a6;
            color: white;
            margin-bottom: 10px;
        }
        
        .nav-btn:hover {
            opacity: 0.8;
        }
        
        .page-indicator {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        /* Analysis Section */
        .analysis-section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .analysis-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        /* Table Styles */
        .table-container {
            overflow-x: auto;
        }
        
        .fraud-analysis-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        .fraud-analysis-table th {
            background: #f8f9fa;
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            white-space: nowrap;
        }
        
        .fraud-analysis-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            vertical-align: top;
        }
        
        .fraud-analysis-table tr:hover {
            background: #f8f9fa;
        }
        
        .field-name {
            font-weight: 500;
            min-width: 150px;
        }
        
        .fraud-score {
            font-weight: bold;
            text-align: center;
            min-width: 80px;
        }
        
        .fraud-score.score-high {
            color: #e74c3c;
            background: #fdf2f2;
        }
        
        .fraud-score.score-medium {
            color: #f39c12;
            background: #fef9e7;
        }
        
        .fraud-score.score-low {
            color: #27ae60;
            background: #eaf5ea;
        }
        
        .fraud-score.score-na {
            color: #95a5a6;
            background: #f8f9fa;
        }
        
        .actual-value, .baseline-value {
            text-align: center;
            min-width: 100px;
            font-family: monospace;
        }
        
        .completion-rate {
            text-align: center;
            min-width: 90px;
            font-family: monospace;
        }
        
        .issue-description {
            max-width: 300px;
            word-wrap: break-word;
        }
        
        /* Score row highlighting */
        .score-row.score-high {
            border-left: 4px solid #e74c3c;
        }
        
        .score-row.score-medium {
            border-left: 4px solid #f39c12;
        }
        
        .score-row.score-low {
            border-left: 4px solid #27ae60;
        }
        
        /* Future sections */
        .future-section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .future-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        
        .placeholder {
            color: #7f8c8d;
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header-main {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .flw-info {
                text-align: left;
                margin-top: 20px;
            }
            
            .page-navigation {
                flex-direction: column;
                gap: 15px;
            }
            
            .nav-left, .nav-center, .nav-right {
                text-align: center;
            }
            
            .fraud-analysis-table {
                font-size: 12px;
            }
            
            .fraud-analysis-table th,
            .fraud-analysis-table td {
                padding: 8px 6px;
            }
        }
        """


# Integration function to add to fraud_html_reporter.py
def add_investigation_pages_to_dashboard(fraud_results_df, baseline_data, output_dir, top_n=25):
    """
    Helper function to generate investigation pages and update main dashboard links.
    
    Args:
        fraud_results_df: Fraud detection results
        baseline_data: Baseline statistics  
        output_dir: Output directory
        top_n: Number of investigation pages to generate
        
    Returns:
        List of generated investigation page paths
    """
    # Generate investigation pages
    generator = FLWInvestigationPageGenerator(fraud_results_df, baseline_data)
    investigation_files = generator.generate_investigation_pages(output_dir, top_n)
    
    print(f"Generated {len(investigation_files)} investigation pages")
    return investigation_files


# Example usage function
def generate_fraud_investigation_example():
    """
    Example showing how to use the FLWInvestigationPageGenerator
    """
    # Assuming you have fraud results and baseline data
    # fraud_results_df = pd.read_csv("fraud_rankings.csv")  
    # baseline_data = json.load(open("baseline_stats.json"))
    
    # Create generator
    # generator = FLWInvestigationPageGenerator(fraud_results_df, baseline_data)
    
    # Generate pages for top 25 high-risk FLWs  
    # output_files = generator.generate_investigation_pages("./output", top_n=25)
    
    print("Example usage:")
    print("generator = FLWInvestigationPageGenerator(fraud_results_df, baseline_data)")
    print("files = generator.generate_investigation_pages('./output', top_n=25)")


if __name__ == "__main__":
    generate_fraud_investigation_example()
