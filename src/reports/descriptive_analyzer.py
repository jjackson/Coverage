import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze_aggregation_functions(df, function_list, existing_scores_df=None, output_path="aggregation_analysis.xlsx"):
    """
    Analyze aggregation functions against data and correlate with existing scores
    
    Args:
        df: Raw visit data with flw_id column
        function_list: List of aggregation functions to test
        existing_scores_df: DataFrame with FLW-level scores (SVAs, etc.)
        output_path: Path for Excel output file
        
    Returns:
        dict: Results summary for each function
    """
    print(f"Starting analysis of {len(function_list)} aggregation functions...")
    
    # Step 1: Calculate aggregation function values for each FLW
    flw_results = []
    flw_groups = df.groupby('flw_id')
    
    print(f"Processing {len(flw_groups)} FLWs...")
    
    for flw_id, group in flw_groups:
        flw_row = {'flw_id': flw_id, 'n_visits': len(group)}
        
        # Apply each aggregation function
        for func in function_list:
            func_name = func.__name__
            try:
                result = func(group)
                flw_row[func_name] = result
            except Exception as e:
                print(f"Error applying {func_name} to FLW {flw_id}: {e}")
                flw_row[func_name] = None
        
        flw_results.append(flw_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(flw_results)
    print(f"DEBUG: results_df shape: {results_df.shape}")
    print(f"DEBUG: results_df columns: {list(results_df.columns)}")
    
    # Step 2: Merge with existing scores if provided
    if existing_scores_df is not None:
        print("Merging with existing scores...")
        # Remove BASELINE row if it exists
        existing_clean = existing_scores_df[existing_scores_df['flw_id'] != 'BASELINE VALUES'].copy()
        print(f"DEBUG: existing_clean shape: {existing_clean.shape}")
        results_df = results_df.merge(existing_clean, on='flw_id', how='left')
        print(f"DEBUG: merged results_df shape: {results_df.shape}")
    
    # Step 3: Generate analysis for each function
    analysis_results = {}
    
    # Collect valid sheets before writing
    sheets_to_write = {}
    summary_data = []
    
    for func in function_list:
        func_name = func.__name__
        
        if func_name not in results_df.columns:
            print(f"DEBUG: {func_name} not in results_df columns, skipping")
            continue
            
        values = results_df[func_name].dropna()
        
        if len(values) == 0:
            print(f"DEBUG: {func_name} has no valid values, skipping")
            continue
        
        print(f"Analyzing {func_name}... ({len(values)} valid values)")
        
        # Basic descriptive stats
        try:
            stats_dict = {
                'function': func_name,
                'n_flws': len(values),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'p05': float(values.quantile(0.05)),
                'p10': float(values.quantile(0.10)),
                'p15': float(values.quantile(0.15)),
                'p20': float(values.quantile(0.20)),
                'p25': float(values.quantile(0.25)),
                'p30': float(values.quantile(0.30)),
                'p35': float(values.quantile(0.35)),
                'p40': float(values.quantile(0.40)),
                'p45': float(values.quantile(0.45)),
                'p50': float(values.quantile(0.50)),  # Same as median
                'p55': float(values.quantile(0.55)),
                'p60': float(values.quantile(0.60)),
                'p65': float(values.quantile(0.65)),
                'p70': float(values.quantile(0.70)),
                'p75': float(values.quantile(0.75)),
                'p80': float(values.quantile(0.80)),
                'p85': float(values.quantile(0.85)),
                'p90': float(values.quantile(0.90)),
                'p95': float(values.quantile(0.95)),
                'p100': float(values.quantile(1.00))  # Same as max
            }
            
            summary_data.append(stats_dict)
            
            # Detailed analysis for this function
            func_analysis = _analyze_single_function(results_df, func_name, existing_scores_df)
            analysis_results[func_name] = func_analysis
            
            # Store sheet data for writing
            if func_analysis['detailed_data'] is not None and len(func_analysis['detailed_data']) > 0:
                # Truncate sheet name to Excel limit (31 chars)
                sheet_name = func_name[:31]
                sheets_to_write[sheet_name] = func_analysis['detailed_data']
                print(f"DEBUG: Prepared sheet '{sheet_name}' with {len(func_analysis['detailed_data'])} rows")
            else:
                print(f"DEBUG: {func_name} detailed_data is empty, skipping sheet")
                
        except Exception as e:
            print(f"ERROR analyzing {func_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Only write Excel if we have valid data
    if summary_data or sheets_to_write:
        try:
            print(f"DEBUG: Writing Excel with {len(summary_data)} summary rows and {len(sheets_to_write)} detail sheets")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Write summary tab (always write this, even if empty)
                summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame([{'function': 'No valid functions analyzed'}])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                print(f"DEBUG: Wrote Summary sheet with {len(summary_df)} rows")
                
                # Write individual function detail sheets
                for sheet_name, sheet_data in sheets_to_write.items():
                    try:
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"DEBUG: Wrote sheet '{sheet_name}' with {len(sheet_data)} rows")
                    except Exception as e:
                        print(f"ERROR writing sheet '{sheet_name}': {e}")
                
                # Write correlation matrix if we have existing scores
                if existing_scores_df is not None and len(results_df) > 0:
                    try:
                        corr_analysis = _correlation_analysis(results_df, function_list)
                        if corr_analysis is not None and len(corr_analysis) > 0:
                            corr_analysis.to_excel(writer, sheet_name='Correlations', index=False)
                            print(f"DEBUG: Wrote Correlations sheet with {len(corr_analysis)} rows")
                        else:
                            # Write placeholder if correlation analysis is empty
                            placeholder_df = pd.DataFrame([{'Message': 'No correlations could be calculated'}])
                            placeholder_df.to_excel(writer, sheet_name='Correlations', index=False)
                            print("DEBUG: Wrote placeholder Correlations sheet")
                    except Exception as e:
                        print(f"ERROR writing correlations: {e}")
                        # Write error message as sheet
                        error_df = pd.DataFrame([{'Error': f'Correlation analysis failed: {str(e)}'}])
                        error_df.to_excel(writer, sheet_name='Correlations', index=False)
            
            print(f"Analysis complete. Results saved to {output_path}")
            
        except Exception as e:
            print(f"ERROR writing Excel file: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("WARNING: No valid data to write to Excel file")
        # Create a minimal Excel file with just an error message
        error_df = pd.DataFrame([{'Error': 'No valid aggregation function results to analyze'}])
        error_df.to_excel(output_path, sheet_name='Error', index=False)
    
    return analysis_results

def _analyze_single_function(results_df, func_name, existing_scores_df):
    """
    Detailed analysis for a single aggregation function
    """
    values = results_df[func_name].dropna()
    
    analysis = {
        'function_name': func_name,
        'basic_stats': {
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'percentiles': {
                f'p{p:02d}': values.quantile(p/100) 
                for p in range(0, 101, 5)  # p00, p05, p10, p15, ..., p95, p100
            }
        }
    }
    
    # Identify outliers
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_low = results_df[results_df[func_name] < lower_bound]['flw_id'].tolist()
    outliers_high = results_df[results_df[func_name] > upper_bound]['flw_id'].tolist()
    
    analysis['outliers'] = {
        'low_outliers': outliers_low,
        'high_outliers': outliers_high,
        'low_threshold': lower_bound,
        'high_threshold': upper_bound
    }
    
    # Create detailed data for Excel output
    detailed_data = results_df[['flw_id', 'n_visits', func_name]].copy()
    detailed_data = detailed_data.dropna(subset=[func_name])
    detailed_data = detailed_data.sort_values(func_name, ascending=False)
    
    # Add percentile ranks
    detailed_data[f'{func_name}_percentile'] = detailed_data[func_name].rank(pct=True) * 100
    
    # Add outlier flags
    detailed_data['outlier_flag'] = 'Normal'
    detailed_data.loc[detailed_data[func_name] < lower_bound, 'outlier_flag'] = 'Low Outlier'
    detailed_data.loc[detailed_data[func_name] > upper_bound, 'outlier_flag'] = 'High Outlier'
    
    # Add existing scores if available
    if existing_scores_df is not None:
        score_cols = [col for col in results_df.columns if col.startswith(('sva_', 'anomaly_score', 'baseline_percentile'))]
        for col in score_cols:
            if col in results_df.columns:
                detailed_data[col] = results_df.set_index('flw_id')[col]
    
    analysis['detailed_data'] = detailed_data
    
    return analysis


def _correlation_analysis(results_df, function_list):
    """
    Calculate correlations between aggregation functions and existing scores
    """
    # Get all numeric columns for correlation
    numeric_cols = []
    
    # Add aggregation function columns
    for func in function_list:
        func_name = func.__name__
        if func_name in results_df.columns:
            numeric_cols.append(func_name)
    
    # Add existing score columns
    existing_score_cols = [col for col in results_df.columns 
                          if col.startswith(('sva_', 'anomaly_score', 'baseline_percentile', 'mean_'))]
    numeric_cols.extend(existing_score_cols)
    
    # Calculate correlation matrix
    if len(numeric_cols) > 1:
        corr_matrix = results_df[numeric_cols].corr()
        
        # Convert to long format for easier reading in Excel
        corr_long = []
        for i, row_name in enumerate(corr_matrix.index):
            for j, col_name in enumerate(corr_matrix.columns):
                if i != j:  # Skip diagonal
                    corr_long.append({
                        'Variable_1': row_name,
                        'Variable_2': col_name,
                        'Correlation': corr_matrix.iloc[i, j],
                        'Abs_Correlation': abs(corr_matrix.iloc[i, j])
                    })
        
        corr_df = pd.DataFrame(corr_long)
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        
        return corr_df
    else:
        return pd.DataFrame({'Message': ['Not enough numeric columns for correlation analysis']})


# Example usage function
def run_standard_analysis(df, existing_scores_df=None, output_dir="."):
    """
    Run analysis with the standard set of aggregation functions
    """

    try:
        from .aggregation_functions import (
            pct_child_unwell_blank,
            pct_child_unwell_yes_of_nonblank,  # NEW
            distance_from_uniform_yearly_bins,
            distance_from_uniform_6month_bins,
            distance_from_uniform_3month_bins
        )
        
        functions_to_test = [
            pct_child_unwell_blank,                    # How often is field blank?
            pct_child_unwell_yes_of_nonblank,          # Of non-blank, how often "yes"?
            distance_from_uniform_yearly_bins,
            distance_from_uniform_6month_bins, 
            distance_from_uniform_3month_bins
        ]
        
    except ImportError as e:
        print(f"ERROR: Could not import aggregation functions: {e}")
        print("Make sure aggregation_functions.py is in the same directory")
        raise

    output_path = os.path.join(output_dir, "aggregation_analysis.xlsx")
    
    return analyze_aggregation_functions(
        df=df,
        function_list=functions_to_test,
        existing_scores_df=existing_scores_df,
        output_path=output_path
    )
