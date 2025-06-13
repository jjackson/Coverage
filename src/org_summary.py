import pandas as pd
from datetime import date, timedelta

def generate_flw_summary(coverage_data):
    """
    Generates a summary per FLW similar to the SQL-based flw summary query,
    including 7-day rolling averages for forms and DUs. Also returns aggregate metrics
    that match create_statistics.py output.

    Returns:
        summary_df: pd.DataFrame (per-FLW breakdown)
        topline_stats: dict (aggregate metrics for summary)
    """
    service_df = coverage_data.create_service_points_dataframe()
    if service_df is None or service_df.empty:
        return pd.DataFrame(), {}


    # Basic cleaning
    service_df = service_df[
        (service_df['du_name'].notna()) &
        (service_df['visit_date'].notna())
    ].copy()

    service_df['visit_day'] = (
        pd.to_datetime(service_df['visit_date'], errors='coerce', utc=True)
        .dt.tz_localize(None)  # remove timezone if present
        .dt.normalize()  # round down to midnight (YYYY-MM-DD)
    )
    service_df = service_df[service_df['visit_day'].notna()]

    #get today's date
    today = pd.to_datetime(date.today())
    last_7_days = today - timedelta(days=6)

    # Total summary per FLW
    grouped = service_df.groupby(['flw_id'])
    flw_summary = grouped.agg(
        flw_name=('flw_name', 'first'),
        date_first_active=('visit_day', 'min'),
        date_last_active=('visit_day', 'max'),
        total_visits=('visit_id', 'count'),
        total_unique_dus_worked=('du_name', pd.Series.nunique),
        unique_days_worked=('visit_day', pd.Series.nunique),
    ).reset_index()

    # Get DU status counts by merging with delivery_units_df
    du_status_counts = coverage_data.delivery_units_df.groupby('flw_commcare_id').agg(
        total_dus_completed=('du_status', lambda x: (x == 'completed').sum()),
        total_dus_visited=('du_status', lambda x: (x == 'visited').sum())
    ).reset_index()

    # Merge the DU status counts with the FLW summary
    flw_summary = flw_summary.merge(
        du_status_counts,
        left_on='flw_id',
        right_on='flw_commcare_id',
        how='left'
    ).fillna(0)

    flw_summary['active_period_days'] = (flw_summary['date_last_active'] - flw_summary['date_first_active']).dt.days + 1 #
    flw_summary['days_since_active'] = (today - flw_summary['date_last_active']).dt.days #
    flw_summary['avrg_forms_per_day'] = round(flw_summary['total_visits'] / flw_summary['unique_days_worked'], 2) #
    flw_summary['dus_per_day'] = round(flw_summary['total_unique_dus_worked'] / flw_summary['unique_days_worked'], 1)

    # 7-day rolling stats
    recent = service_df[service_df['visit_day'] >= last_7_days]
    recent_grouped = recent.groupby('flw_id').agg(
        visits_last7=('visit_id', 'count'),
        dus_last7=('du_name', pd.Series.nunique)
    ).reset_index()
    recent_grouped['avrg_forms_per_day_mavrg'] = round(recent_grouped['visits_last7'] / 7.0, 2)
    recent_grouped['dus_per_day_mavrg'] = round(recent_grouped['dus_last7'] / 7.0, 2)

    # Merge recent activity with full summary
    summary = pd.merge(flw_summary, recent_grouped, on='flw_id', how='left')
    summary.fillna({'forms_per_day_last_7d': 0, 'dus_per_day_last_7d': 0}, inplace=True)

    #order summary in a this column order
    summary = summary[['flw_id', 'flw_name', 'total_visits', 'date_first_active', 'date_last_active', 'days_since_active', 'active_period_days', 'unique_days_worked', 'avrg_forms_per_day', 'avrg_forms_per_day_mavrg', 'total_unique_dus_worked', 'dus_per_day', 'dus_per_day_mavrg', 'total_dus_completed', 'total_dus_visited']]

    # Topline stats matching create_statistics.py
    topline_stats = {
        'total_visits': int(summary['total_visits'].sum()),
        'total_dus_worked': int(summary['total_unique_dus_worked'].sum()),
        'average_dus_per_day': round(summary['dus_per_day'].mean(), 2),
        'average_forms_per_day': round(summary['avrg_forms_per_day'].mean(), 2),
        'total_dus_completed': coverage_data.total_completed_dus,
        'total_dus_visited': coverage_data.total_visited_dus,
        'total_delivery_units': coverage_data.total_delivery_units,
        'total_buildings': coverage_data.total_buildings,
        'total_flws': coverage_data.total_flws
    }

    return summary, topline_stats
