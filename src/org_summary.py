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
            summary['flw_id'] = None
            summary['flw_name'] = None

            # Add completed and visited DUs for opportunity level
            for org, cov in coverage_data_objects.items():
                if org in summary['opportunity'].values:
                    summary.loc[summary['opportunity'] == org, 'total_dus_completed'] = cov.total_completed_dus
                    summary.loc[summary['opportunity'] == org, 'total_dus_visited'] = cov.total_visited_dus
        else:
            summary = service_df.groupby(['flw_id', 'opportunity']).agg(
                total_visits=('visit_id', 'count'),
                total_unique_dus_worked=('du_name', pd.Series.nunique),
                unique_days_worked=('visit_day', pd.Series.nunique),
                date_first_active=('visit_day', 'min'),
                date_last_active=('visit_day', 'max'),
                flw_name=('flw_name', 'first')
            ).reset_index()

            # Add completed and visited DUs for FLW level
            for org, cov in coverage_data_objects.items():
                # Get the mapping between flw_id and cchq_user_owner_id from service points
                flw_id_to_commcare = {}
                for point in cov.service_points:
                    if point.flw_id and point.flw_commcare_id:
                        flw_id_to_commcare[point.flw_id] = point.flw_commcare_id
                
                for flw_id in summary[summary['opportunity'] == org]['flw_id'].unique():
                    # Get the corresponding commcare_id
                    commcare_id = flw_id_to_commcare.get(flw_id)
                    if commcare_id and commcare_id in cov.flws:
                        flw = cov.flws[commcare_id]
                        mask = (summary['opportunity'] == org) & (summary['flw_id'] == flw_id)
                        summary.loc[mask, 'total_dus_completed'] = flw.status_counts.get('completed', 0)
                        summary.loc[mask, 'total_dus_visited'] = flw.status_counts.get('visited', 0)
        
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

    # change the date format of 'date_first_active' in 'DD-MM-YYYY'
    combined_summary['date_first_active'] = combined_summary['date_first_active'].dt.strftime('%m-%d-%Y')

    # change the date format of 'date_last_active' in 'DD-MM-YYYY'
    combined_summary['date_last_active'] = combined_summary['date_last_active'].dt.strftime('%m-%d-%Y')

    # sort the 'combined_summary' data frame with 'days_since_active' , highest number on top
    combined_summary = combined_summary.sort_values(by='days_since_active', ascending=False)

    
    # Calculate topline stats
    topline_stats = {
        'total_visits': int(combined_summary['total_visits'].sum()),
        'total_dus_worked': int(combined_summary['total_unique_dus_worked'].sum()),
        'average_dus_per_day': round(combined_summary['dus_per_day'].mean(), 2),
        'average_forms_per_day': round(combined_summary['avrg_forms_per_day'].mean(), 2),
    }
    
    # Define the desired column order
    column_order = [
        'flw_name',
        'flw_id',
        'opportunity',
        'total_visits',
        'date_first_active',
        'date_last_active',
        'days_since_active',
        'active_period_days',
        'unique_days_worked',
        'avrg_forms_per_day',
        'avrg_forms_per_day_mavrg',
        'total_unique_dus_worked',
        'dus_per_day',
        'dus_per_day_mavrg',
        'total_dus_completed',
        'total_dus_visited'
    ]

    # Ensure all columns exist
    for col in column_order:
        if col not in combined_summary.columns:
            combined_summary[col] = 0  # Initialize with 0 instead of None for numeric columns

    # Reorder columns
    combined_summary = combined_summary[column_order]
    
    return combined_summary, topline_stats
