
import pandas as pd
from datetime import timedelta

def generate_flw_summary(coverage_data):
    """
    Generates a summary per FLW including:
    - Active date bounds
    - Total visits and DUs
    - Rolling 7-day averages
    - Flags for low activity
    """
    service_df = coverage_data.create_service_points_dataframe()
    if service_df is None or service_df.empty:
        return pd.DataFrame()

    # Clean visit_day
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

    today = pd.Timestamp.today().normalize()
    last_7_days = today - timedelta(days=6)

    # Grouped FLW stats
    grouped = service_df.groupby(['flw_id'])

    flw_summary = grouped.agg(
        total_visits=('visit_id', 'count'),
        total_unique_dus_worked=('du_name', pd.Series.nunique),
        date_first_active=('visit_day', 'min'),
        date_last_active=('visit_day', 'max'),
        unique_days_worked=('visit_day', pd.Series.nunique)
    ).reset_index()

    flw_summary['active_period_days'] = (
        (flw_summary['date_last_active'] - flw_summary['date_first_active']).dt.days + 1
    )
    flw_summary['days_since_active'] = (today - flw_summary['date_last_active']).dt.days
    flw_summary['pct_days_working'] = round(
        (flw_summary['unique_days_worked'] / flw_summary['active_period_days']) * 100, 2
    )
    flw_summary['avrg_forms_per_day'] = round(
        flw_summary['total_visits'] / flw_summary['unique_days_worked'], 2
    )
    flw_summary['dus_per_day'] = round(
        flw_summary['total_unique_dus_worked'] / flw_summary['unique_days_worked'], 1
    )

    # 7-day activity
    recent = service_df[service_df['visit_day'] >= last_7_days]
    recent_grouped = recent.groupby('flw_id').agg(
        visits_last7=('visit_id', 'count'),
        dus_last7=('du_name', pd.Series.nunique)
    ).reset_index()
    recent_grouped['forms_per_day_last_7d'] = round(recent_grouped['visits_last7'] / 7.0, 2)
    recent_grouped['dus_per_day_last_7d'] = round(recent_grouped['dus_last7'] / 7.0, 2)

    summary = pd.merge(flw_summary, recent_grouped, on='flw_id', how='left')
    summary.fillna({'forms_per_day_last_7d': 0, 'dus_per_day_last_7d': 0}, inplace=True)

    # Add DU completion counts using flw_commcare_id
    du_df = coverage_data.delivery_units_df
    if 'flw_commcare_id' in du_df.columns:
        du_counts = du_df.groupby('flw_commcare_id').agg(
            total_dus_completed=('du_status', lambda x: (x == 'completed').sum()),
            total_dus_visited=('du_status', lambda x: (x == 'visited').sum())
        ).reset_index()

        summary = summary.merge(du_counts, how='left', left_on='flw_id', right_on='flw_commcare_id')
        summary.drop(columns=['flw_commcare_id'], inplace=True, errors='ignore')
    else:
        summary['total_dus_completed'] = None
        summary['total_dus_visited'] = None

    return summary
