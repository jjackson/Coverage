#!/usr/bin/env python3
"""
MUAC Sparkline Grid by Feature Score

Generates two sparkline grids showing MUAC distributions grouped by how many
binary features they pass (0-6 score). One grid for real FLWs (with photos),
one for fake FLWs (without photos).

Usage:
    python muac_sparkline_by_score.py <features_csv_path> <output_dir>
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import random
from PIL import Image
import glob


class MUACSparklineByScore:
    """Generate MUAC sparkline grids grouped by feature pass score"""
    
    def __init__(self, features_csv_path, output_dir):
        self.features_csv_path = Path(features_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # MUAC bin column names in the features CSV
        self.muac_bin_columns = [
            'muac_9_5_10_5',
            'muac_10_5_11_5',
            'muac_11_5_12_5',
            'muac_12_5_13_5',
            'muac_13_5_14_5',
            'muac_14_5_15_5',
            'muac_15_5_16_5',
            'muac_16_5_17_5',
            'muac_17_5_18_5',
            'muac_18_5_19_5',
            'muac_19_5_20_5',
            'muac_20_5_21_5'
        ]
        
        # Grid settings
        self.grid_cols = 15
        self.image_dpi = 300
        self.photo_strip_width = 2.0
        self.photo_count = 10
    
    
    def generate_both_grids(self):
        """Generate sparkline grids for both real and fake FLWs"""
        df = self.load_data()
        
        # Generate real FLWs grid (with photos)
        print("\nGenerating Real FLWs grid...")
        real_df = df[df['classification'] == 'real'].copy()
        real_file = self._generate_grid(real_df, 'Real', include_photos=True)
        
        # Generate fake FLWs grid (no photos)
        print("\nGenerating Fake FLWs grid...")
        fake_df = df[df['classification'] == 'fake'].copy()
        fake_file = self._generate_grid(fake_df, 'Fake', include_photos=False)
        
        print(f"\nGenerated:")
        print(f"  {real_file}")
        print(f"  {fake_file}")
        
        return [real_file, fake_file]
    
    def _generate_grid(self, df, label, include_photos):
        """Generate a single sparkline grid"""
        
        # Filter to FLWs with valid MUAC data
        valid_df = df[df['muac_features_passed'] >= 0].copy()
        
        if len(valid_df) == 0:
            print(f"No valid MUAC data for {label} FLWs")
            return None
        
        print(f"  {len(valid_df)} {label} FLWs with valid MUAC data")
        
        # Organize into sections by score
        sections = self._organize_by_score(valid_df)
        
        # Generate the grid
        fig = self._create_sparkline_grid(sections, label, include_photos)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"muac_sparklines_{label.lower()}_{timestamp}.png"
        
        fig.savefig(output_file, dpi=self.image_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(output_file)
    
    def _organize_by_score(self, df):
        """Organize FLWs into sections by muac_features_passed score"""
        sections = []
        
        # Group by score (6 down to 0)
        for score in range(6, -1, -1):
            score_df = df[df['muac_features_passed'] == score].copy()
            
            if len(score_df) == 0:
                continue
            
            # Sort by visits descending
            score_df = score_df.sort_values('visits', ascending=False)
            
            # Create section
            if score == 6:
                title = f'{score}/6 features passed (authentic pattern) - {len(score_df)} FLWs'
                bg_color = '#e8f5e9'  # Light green
                border_color = 'green'
            elif score >= 4:
                title = f'{score}/6 features passed (minor/moderate issues) - {len(score_df)} FLWs'
                bg_color = '#fff9c4'  # Light yellow
                border_color = 'orange'
            else:
                title = f'{score}/6 features passed (major issues) - {len(score_df)} FLWs'
                bg_color = '#ffebee'  # Light red
                border_color = 'red'
            
            sections.append({
                'title': title,
                'data': score_df,
                'background_color': bg_color,
                'border_color': border_color
            })
        
        return sections
    
    def _create_sparkline_grid(self, sections, label, include_photos):
        """Create the sparkline grid visualization"""
        cols = self.grid_cols
        
        # Calculate rows needed
        section_layouts = []
        total_rows_needed = 0
        
        for i, section in enumerate(sections):
            num_charts = len(section['data'])
            charts_rows = max(1, num_charts // cols + (1 if num_charts % cols > 0 else 0))
            header_rows = 1
            spacing_rows = 1 if i > 0 else 0
            
            section_layouts.append({
                'section': section,
                'header_rows': header_rows,
                'charts_rows': charts_rows,
                'spacing_rows': spacing_rows,
                'total_rows': spacing_rows + header_rows + charts_rows,
                'start_row': total_rows_needed
            })
            
            total_rows_needed += spacing_rows + header_rows + charts_rows
        
        # Create figure
        fig_width = max(15, cols * 0.8)
        if include_photos:
            fig_width += self.photo_strip_width
        
        fig_height = max(10, total_rows_needed * 0.4)
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.image_dpi)
        
        # Create grid spec
        if include_photos:
            main_width_ratio = fig_width - self.photo_strip_width
            photo_width_ratio = self.photo_strip_width
            width_ratios = [main_width_ratio, photo_width_ratio]
            
            gs_main = gridspec.GridSpec(total_rows_needed, 2, figure=fig,
                                      width_ratios=width_ratios, wspace=0.05)
            
            gs_sparklines = gridspec.GridSpecFromSubplotSpec(
                total_rows_needed, cols, gs_main[:, 0], hspace=0.1, wspace=0.02
            )
            
            axes = []
            for row in range(total_rows_needed):
                row_axes = []
                for col in range(cols):
                    ax = fig.add_subplot(gs_sparklines[row, col])
                    row_axes.append(ax)
                axes.append(row_axes)
            axes = np.array(axes)
            
        else:
            axes = []
            for row in range(total_rows_needed):
                row_axes = []
                for col in range(cols):
                    ax = plt.subplot(total_rows_needed, cols, row * cols + col + 1)
                    row_axes.append(ax)
                axes.append(row_axes)
            axes = np.array(axes)
        
        # Process each section
        for layout in section_layouts:
            section = layout['section']
            section_data = section['data']
            start_row = layout['start_row']
            
            # Spacing row
            if layout['spacing_rows'] > 0:
                for col in range(cols):
                    axes[start_row, col].axis('off')
                start_row += layout['spacing_rows']
            
            # Header row
            for col in range(cols):
                ax = axes[start_row, col]
                if col == 0:
                    ax.text(0.05, 0.5, section['title'],
                           transform=ax.transAxes, fontsize=12, fontweight='bold',
                           verticalalignment='center')
                ax.axis('off')
            
            # Chart rows
            chart_idx = 0
            charts_start_row = start_row + layout['header_rows']
            
            for charts_row in range(layout['charts_rows']):
                for col in range(cols):
                    if chart_idx >= len(section_data):
                        axes[charts_start_row + charts_row, col].axis('off')
                        continue
                    
                    ax = axes[charts_start_row + charts_row, col]
                    data_row = section_data.iloc[chart_idx]
                    
                    # Get MUAC bin counts
                    muac_values = self._extract_muac_bins(data_row)
                    x_positions = range(len(muac_values))
                    
                    # Create bar chart
                    ax.bar(x_positions, muac_values, width=0.8, color='#2E5984',
                          alpha=0.8, edgecolor='none')
                    
                    # Styling
                    ax.set_facecolor(section['background_color'])
                    for spine in ax.spines.values():
                        spine.set_color(section['border_color'])
                        spine.set_linewidth(1.5 if section['border_color'] == 'red' else 1.0)
                    
                    # Remove axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(left=False, right=False, top=False, bottom=False)
                    
                    # Set limits
                    max_val = max(muac_values) if max(muac_values) > 0 else 1
                    ax.set_ylim(0, max_val * 1.1)
                    ax.set_xlim(-0.5, len(muac_values) - 0.5)
                    
                    chart_idx += 1
        
        # Add photo strip if enabled
        if include_photos:
            self._add_photo_strip(fig, gs_main, total_rows_needed)
        
        # Overall title
        total_charts = sum(len(section['data']) for section in sections)
        title_text = f'{label} FLWs MUAC Distribution Analysis: {total_charts} FLWs grouped by feature score'
        fig.suptitle(title_text, fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.1, wspace=0.02)
        
        return fig
    
    def _extract_muac_bins(self, row):
        """Extract MUAC bin counts from a row"""
        values = []
        for col in self.muac_bin_columns:
            if col in row.index:
                val = row[col]
                values.append(val if pd.notna(val) else 0)
            else:
                values.append(0)
        return values
    
    def _add_photo_strip(self, fig, gs_main, total_rows_needed):
        """Add vertical photo strip (simplified version)"""
        photo_files = self._find_photo_files()
        if not photo_files:
            return
        
        sample_size = min(self.photo_count, len(photo_files))
        selected_photos = random.sample(photo_files, sample_size)
        
        strip_width_px = int(self.photo_strip_width * self.image_dpi)
        max_photo_height_px = int(strip_width_px * 1.2)
        
        gs_photos = gridspec.GridSpecFromSubplotSpec(
            sample_size + 1, 1,
            gs_main[:, 1],
            hspace=0.02
        )
        
        # Header
        ax_header = fig.add_subplot(gs_photos[0, 0])
        ax_header.text(0.5, 0.7, 'Sample MUAC Photos',
                      ha='center', va='center', fontsize=10, fontweight='bold',
                      transform=ax_header.transAxes)
        ax_header.text(0.5, 0.3, 'One time- and gps-\nstamped photo taken\nfor every child',
                      ha='center', va='center', fontsize=10, style='italic', color='#666666',
                      transform=ax_header.transAxes)
        ax_header.axis('off')
        
        # Photos
        loaded_photos = []
        for photo_path in selected_photos:
            photo_array = self._load_and_resize_photo(photo_path, strip_width_px, max_photo_height_px)
            if photo_array is not None:
                loaded_photos.append(photo_array)
        
        for i, photo_array in enumerate(loaded_photos):
            if i + 1 < sample_size + 1:
                ax_photo = fig.add_subplot(gs_photos[i + 1, 0])
                ax_photo.imshow(photo_array, aspect='auto')
                ax_photo.axis('off')
    
    def _find_photo_files(self):
        """Find photo files in data/muac photos directory"""
        search_paths = [
            Path(self.output_dir).parent / "data" / "muac photos",
            Path(self.output_dir) / "data" / "muac photos",
            Path("data") / "muac photos",
            Path("data") / "muac_photos"
        ]
        
        for data_dir in search_paths:
            if data_dir.exists():
                photo_files = []
                for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
                    photo_files.extend(glob.glob(str(data_dir / pattern)))
                if photo_files:
                    return photo_files
        
        return []
    
    def _load_and_resize_photo(self, photo_path, target_width_px, max_height_px):
        """Load and resize photo"""
        try:
            with Image.open(photo_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                aspect_ratio = img.width / img.height
                
                if aspect_ratio > 1:
                    new_width = target_width_px
                    new_height = int(target_width_px / aspect_ratio)
                else:
                    new_height = min(max_height_px, int(target_width_px / aspect_ratio))
                    new_width = int(new_height * aspect_ratio)
                
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return np.array(img_resized)
        except Exception as e:
            print(f"Error loading photo {photo_path}: {e}")
            return None

    @classmethod
    def from_dataframe(cls, df, output_dir):
        """Create instance from DataFrame instead of CSV file"""
        instance = cls.__new__(cls)
        instance.features_csv_path = None
        instance.output_dir = Path(output_dir)
        instance.output_dir.mkdir(exist_ok=True)
        instance._df = df  # Store DataFrame directly
        
        instance.muac_bin_columns = [
            'muac_9_5_10_5_visits',
            'muac_10_5_11_5_visits',
            'muac_11_5_12_5_visits',
            'muac_12_5_13_5_visits',
            'muac_13_5_14_5_visits',
            'muac_14_5_15_5_visits',
            'muac_15_5_16_5_visits',
            'muac_16_5_17_5_visits',
            'muac_17_5_18_5_visits',
            'muac_18_5_19_5_visits',
            'muac_19_5_20_5_visits',
            'muac_20_5_21_5_visits'
        ]
        
        instance.grid_cols = 15
        instance.image_dpi = 300
        instance.photo_strip_width = 2.0
        instance.photo_count = 10
        
        return instance

    def load_data(self):
        """Load features - either from CSV or from stored DataFrame"""
        if hasattr(self, '_df'):
            df = self._df
            print(f"Using in-memory DataFrame with {len(df)} FLWs")
        else:
            print(f"Loading data from {self.features_csv_path}")
            df = pd.read_csv(self.features_csv_path)
        
        # Verify required columns exist
        required_cols = ['classification', 'muac_features_passed', 'visits']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Loaded {len(df)} FLWs")
        print(f"  Real: {(df['classification']=='real').sum()}")
        print(f"  Fake: {(df['classification']=='fake').sum()}")
        
        return df


def main():
    if len(sys.argv) < 3:
        print("Usage: python muac_sparkline_by_score.py <features_csv_path> <output_dir>")
        print("\nExample:")
        print("  python muac_sparkline_by_score.py ml_features_360flws_67890visits.csv ./output")
        sys.exit(1)
    
    features_csv = sys.argv[1]
    output_dir = sys.argv[2]
    
    generator = MUACSparklineByScore(features_csv, output_dir)
    generator.generate_both_grids()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
