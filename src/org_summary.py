import pandas as pd
from datetime import date, timedelta

def generate_summary(coverage_data_objects, group_by='opportunity'):
    """
    Generates a summary that can be grouped by either opportunity or FLW level
    
    Args:
        coverage_data_objects: Dict of coverage data objects keyed by opportunity
        group_by: Either 'opportunity' or 'flw' to determine grouping level
    
    Returns:
        summary_df: pd.DataFrame with aggregated metrics
        topline_stats: dict with aggregate metrics
    """
    all_summaries = []
    
    for org, cov in coverage_data_objects.items():
        service_df = cov.create_service_points_dataframe()
        if service_df is None or service_df.empty:
            continue
            
        # Basic cleaning
        service_df = service_df[
            (service_df['du_name'].notna()) &
            (service_df['visit_date'].notna())
        ].copy()
        
        service_df['visit_day'] = (
            pd.to_datetime(service_df['visit_date'], errors='coerce', utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        service_df = service_df[service_df['visit_day'].notna()]
        
        # Add opportunity column
        service_df['opportunity'] = org
        
        # Get today's date for recency calculations
        today = pd.to_datetime(date.today())
        last_7_days = today - timedelta(days=6)
        
        # Group by either opportunity or FLW
        group_col = 'opportunity' if group_by == 'opportunity' else 'flw_id'
        
        # Calculate metrics
        if group_by == 'opportunity':
            summary = service_df.groupby('opportunity').agg(
                total_visits=('visit_id', 'count'),
                total_unique_dus_worked=('du_name', pd.Series.nunique),
                unique_days_worked=('visit_day', pd.Series.nunique),
                date_first_active=('visit_day', 'min'),
                date_last_active=('visit_day', 'max'),
            ).reset_index()
        else:
            summary = service_df.groupby(['flw_id', 'opportunity']).agg(
                total_visits=('visit_id', 'count'),
                total_unique_dus_worked=('du_name', pd.Series.nunique),
                unique_days_worked=('visit_day', pd.Series.nunique),
                date_first_active=('visit_day', 'min'),
                date_last_active=('visit_day', 'max'),
            ).reset_index()
        
        # Calculate derived metrics
        summary['active_period_days'] = (summary['date_last_active'] - summary['date_first_active']).dt.days + 1
        summary['days_since_active'] = (today - summary['date_last_active']).dt.days
        summary['avrg_forms_per_day'] = round(summary['total_visits'] / summary['unique_days_worked'], 2)
        summary['dus_per_day'] = round(summary['total_unique_dus_worked'] / summary['unique_days_worked'], 1)
        
        # Calculate 7-day rolling stats
        recent = service_df[service_df['visit_day'] >= last_7_days]
        if group_by == 'opportunity':
            recent_grouped = recent.groupby('opportunity').agg(
                visits_last7=('visit_id', 'count'),
                dus_last7=('du_name', pd.Series.nunique)
            ).reset_index()
        else:
            recent_grouped = recent.groupby(['flw_id', 'opportunity']).agg(
                visits_last7=('visit_id', 'count'),
                dus_last7=('du_name', pd.Series.nunique)
            ).reset_index()
        
        recent_grouped['avrg_forms_per_day_mavrg'] = round(recent_grouped['visits_last7'] / 7.0, 2)
        recent_grouped['dus_per_day_mavrg'] = round(recent_grouped['dus_last7'] / 7.0, 2)
        
        # Merge recent activity with full summary
        summary = summary.merge(recent_grouped, on=['opportunity'] if group_by == 'opportunity' else ['flw_id', 'opportunity'], how='left')
        summary.fillna({'avrg_forms_per_day_mavrg': 0, 'dus_per_day_mavrg': 0}, inplace=True)
        
        all_summaries.append(summary)
    
    # Combine all summaries
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # Calculate topline stats
    topline_stats = {
        'total_visits': int(combined_summary['total_visits'].sum()),
        'total_dus_worked': int(combined_summary['total_unique_dus_worked'].sum()),
        'average_dus_per_day': round(combined_summary['dus_per_day'].mean(), 2),
        'average_forms_per_day': round(combined_summary['avrg_forms_per_day'].mean(), 2),
    }
    
    return combined_summary, topline_stats
