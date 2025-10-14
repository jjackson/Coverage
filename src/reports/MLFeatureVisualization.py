"""
ML Feature Distribution Visualizations

Generates comparison plots of features between real and fake FLWs
to assess discriminative power before ML training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_all_visualizations(df, output_dir, log_callback=None):
    """
    Generate all feature distribution visualizations
    
    Args:
        df: DataFrame with aggregated FLW features (must have 'classification' column)
        output_dir: Directory to save PNG files
        log_callback: Optional function to log progress messages
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    log("Generating feature distribution visualizations...")
    
    # Split into real and fake
    real_df = df[df['classification'] == 'real']
    fake_df = df[df['classification'] == 'fake']
    
    log(f"  Real FLWs: {len(real_df)}, Fake FLWs: {len(fake_df)}")
    
    # 1. Age distribution comparison
    age_file = output_path / "age_distribution_comparison.png"
    _plot_age_distributions(real_df, fake_df, age_file)
    log(f"  Generated: {age_file.name}")
    
    # 2. MUAC binary features comparison
    muac_binary_file = output_path / "muac_binary_features_comparison.png"
    _plot_muac_binary_features(real_df, fake_df, muac_binary_file)
    log(f"  Generated: {muac_binary_file.name}")
    
    # 3. MUAC continuous features comparison
    muac_continuous_file = output_path / "muac_continuous_features_comparison.png"
    _plot_muac_continuous_features(real_df, fake_df, muac_continuous_file)
    log(f"  Generated: {muac_continuous_file.name}")
    
    log("Feature visualizations complete!")
    
    return [str(age_file), str(muac_binary_file), str(muac_continuous_file)]


def _plot_age_distributions(real_df, fake_df, output_file):
    """Plot side-by-side age bin distributions"""
    
    # Age bin columns
    age_bins = [f'pct_age_{i*5}_{i*5+4}' for i in range(12)]
    bin_labels = [f'{i*5}-{i*5+4}' for i in range(12)]
    
    # Calculate mean percentages for each bin
    real_pcts = [real_df[col].mean() * 100 for col in age_bins if col in real_df.columns]
    fake_pcts = [fake_df[col].mean() * 100 for col in age_bins if col in fake_df.columns]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = np.arange(len(bin_labels))
    
    # Real FLWs
    bars1 = ax1.bar(x_pos, real_pcts, color='#2E5984', alpha=0.8)
    ax1.set_xlabel('Age (months)')
    ax1.set_ylabel('Percentage of children')
    ax1.set_title(f'Real FLWs (n={len(real_df)})')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.set_ylim(0, max(max(real_pcts), max(fake_pcts)) * 1.2)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars1, real_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Fake FLWs
    bars2 = ax2.bar(x_pos, fake_pcts, color='#D62728', alpha=0.8)
    ax2.set_xlabel('Age (months)')
    ax2.set_ylabel('Percentage of children')
    ax2.set_title(f'Fake FLWs (n={len(fake_df)})')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.set_ylim(0, max(max(real_pcts), max(fake_pcts)) * 1.2)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars2, fake_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Age Distribution Comparison: Real vs Fake FLWs', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_muac_binary_features(real_df, fake_df, output_file):
    """Plot comparison of MUAC binary feature rates"""
    
    binary_features = [
        'muac_increasing_to_peak',
        'muac_decreasing_from_peak', 
        'muac_no_skipped_bins',
        'muac_no_plateau',
        'muac_bins_sufficient',
        'muac_peak_reasonable'
    ]
    
    feature_labels = [
        'Increasing to peak',
        'Decreasing from peak',
        'No skipped bins',
        'No plateau',
        'Bins sufficient (=5)',
        'Peak reasonable (=50%)'
    ]
    
    # Calculate percentages (feature=1)
    real_pcts = []
    fake_pcts = []
    deltas = []
    
    for feat in binary_features:
        if feat in real_df.columns and feat in fake_df.columns:
            # Filter out -1 sentinel values before calculating percentage
            real_valid = real_df[real_df[feat] != -1]
            fake_valid = fake_df[fake_df[feat] != -1]
            
            if len(real_valid) > 0 and len(fake_valid) > 0:
                real_pct = (real_valid[feat] == 1).mean() * 100
                fake_pct = (fake_valid[feat] == 1).mean() * 100
                real_pcts.append(real_pct)
                fake_pcts.append(fake_pct)
                deltas.append(abs(fake_pct - real_pct))
    
    if len(real_pcts) == 0:
        print("Warning: No valid MUAC binary features found for visualization")
        return
    
    # Trim labels to match actual data
    feature_labels = feature_labels[:len(real_pcts)]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(feature_labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, real_pcts, width, label='Real FLWs', 
                   color='#2E5984', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, fake_pcts, width, label='Fake FLWs',
                   color='#D62728', alpha=0.8)
    
    ax.set_xlabel('MUAC Feature')
    ax.set_ylabel('Percentage with feature = 1 (passes test)')
    ax.set_title(f'MUAC Binary Features: Real (n={len(real_df)}) vs Fake (n={len(fake_df)})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add percentage labels and deltas
    for i, (bar1, bar2, delta) in enumerate(zip(bars1, bars2, deltas)):
        # Real percentage
        ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
               f'{real_pcts[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        # Fake percentage
        ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
               f'{fake_pcts[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        # Delta
        max_height = max(bar1.get_height(), bar2.get_height())
        ax.text(x_pos[i], max_height + 5, f'?={delta:.1f}pp',
               ha='center', va='bottom', fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def _plot_muac_continuous_features(real_df, fake_df, output_file):
    """Plot distributions of continuous MUAC features"""
    
    continuous_features = [
        ('muac_peak_concentration', 'MUAC Peak Concentration'),
        ('muac_bins_with_data', 'MUAC Bins with Data'),
        ('muac_completion_rate', 'MUAC Completion Rate')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (feat, label) in zip(axes, continuous_features):
        if feat in real_df.columns and feat in fake_df.columns:
            # Filter out -1 sentinel values
            real_data = real_df[real_df[feat] != -1][feat].dropna()
            fake_data = fake_df[fake_df[feat] != -1][feat].dropna()
            
            if len(real_data) > 0 and len(fake_data) > 0:
                # Create histograms with density normalization
                bins = 20
                ax.hist(real_data, bins=bins, alpha=0.6, label='Real', 
                       color='#2E5984', density=True, weights=np.ones(len(real_data)) * 100 / len(real_data))
                ax.hist(fake_data, bins=bins, alpha=0.6, label='Fake', 
                       color='#D62728', density=True, weights=np.ones(len(fake_data)) * 100 / len(fake_data))
                
                ax.set_xlabel(label)
                ax.set_ylabel('Percentage of FLWs')
                ax.set_title(f'{label}\nReal: u={real_data.mean():.3f}, Fake: u={fake_data.mean():.3f}')
                ax.legend()
    
    plt.suptitle('MUAC Continuous Features Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
