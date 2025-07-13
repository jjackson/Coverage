#!/usr/bin/env python3
"""
MUAC Sparkline Grid Report Generator with Photo Strip

Creates a grid of MUAC distribution sparklines from CSV data with a vertical photo strip.
Integrates with the existing report generator framework.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime
import os
import random
from PIL import Image
import glob

class MUACSparklineReport:
    """MUAC Sparkline Grid Report with Photo Strip - integrates with existing report framework"""
    
    def __init__(self, df, output_dir, log_function, params_frame):
        self.df = df
        self.output_dir = output_dir
        self.log = log_function
        self.params_frame = params_frame
        
        # MUAC distribution columns in order
        self.muac_columns = [
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
        
        # Parameter variables (will be set up in setup_parameters)
        self.grid_rows = None
        self.image_dpi = None
        self.include_labels = None
        
    @classmethod
    def setup_parameters(cls, parent_frame):
        """Setup UI parameters for this report type"""
        # Clear any existing widgets
        for widget in parent_frame.winfo_children():
            widget.destroy()
            
        # Grid columns parameter
        col_frame = ttk.Frame(parent_frame)
        col_frame.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(col_frame, text="Grid Columns:").grid(row=0, column=0, sticky=tk.W)
        cls.grid_cols_var = tk.IntVar(value=15)
        ttk.Spinbox(col_frame, from_=6, to=20, textvariable=cls.grid_cols_var, 
                   width=10).grid(row=0, column=1, padx=(5, 0))
        ttk.Label(col_frame, text="(6-20 columns)").grid(row=0, column=2, padx=(5, 0))
        
        # Image DPI parameter
        dpi_frame = ttk.Frame(parent_frame)
        dpi_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(dpi_frame, text="Image DPI:").grid(row=0, column=0, sticky=tk.W)
        cls.image_dpi_var = tk.IntVar(value=300)
        ttk.Spinbox(dpi_frame, from_=150, to=600, textvariable=cls.image_dpi_var, 
                   width=10).grid(row=0, column=1, padx=(5, 0))
        ttk.Label(dpi_frame, text="(150-600 DPI)").grid(row=0, column=2, padx=(5, 0))
        
        # Photo strip parameters
        photo_frame = ttk.Frame(parent_frame)
        photo_frame.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        cls.enable_photo_strip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(photo_frame, text="Include photo strip", 
                       variable=cls.enable_photo_strip_var).grid(row=0, column=0, sticky=tk.W)
        
        # Number of photos parameter
        photo_count_frame = ttk.Frame(parent_frame)
        photo_count_frame.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(photo_count_frame, text="Number of photos:").grid(row=0, column=0, sticky=tk.W)
        cls.photo_count_var = tk.IntVar(value=10)
        ttk.Spinbox(photo_count_frame, from_=6, to=15, textvariable=cls.photo_count_var, 
                   width=10).grid(row=0, column=1, padx=(5, 0))
        ttk.Label(photo_count_frame, text="(6-15 photos)").grid(row=0, column=2, padx=(5, 0))
        
        # Photo strip width parameter
        photo_width_frame = ttk.Frame(parent_frame)
        photo_width_frame.grid(row=4, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(photo_width_frame, text="Photo strip width:").grid(row=0, column=0, sticky=tk.W)
        cls.photo_strip_width_var = tk.StringVar(value="2.0")
        width_combo = ttk.Combobox(photo_width_frame, textvariable=cls.photo_strip_width_var, 
                                  values=["1.5", "2.0", "2.5"], width=8, state="readonly")
        width_combo.grid(row=0, column=1, padx=(5, 0))
        ttk.Label(photo_width_frame, text="inches").grid(row=0, column=2, padx=(5, 0))
        
        # Info label
        info_frame = ttk.Frame(parent_frame)
        info_frame.grid(row=5, column=0, sticky=tk.W, pady=10)
        
        info_text = ("Generates a grid of MUAC distribution bar charts with sample photos.\n"
                    "Flagged cases (red borders) shown first,\n"
                    "then standard ranges (light green background).\n"
                    "Photos randomly sampled from data/muac photos directory.\n"
                    "Ranges: <200, 200-500, 500-1000, >1000 MUAC readings.")
        ttk.Label(info_frame, text=info_text, foreground="gray").grid(row=0, column=0, sticky=tk.W)
        
    def _get_parameters(self):
        """Get current parameter values from the UI"""
        try:
            self.grid_cols = self.__class__.grid_cols_var.get()
            self.image_dpi = self.__class__.image_dpi_var.get()
            self.enable_photo_strip = self.__class__.enable_photo_strip_var.get()
            self.photo_count = self.__class__.photo_count_var.get()
            self.photo_strip_width = float(self.__class__.photo_strip_width_var.get())
        except AttributeError:
            # Fallback to defaults if parameters not set up
            self.grid_cols = 15
            self.image_dpi = 300
            self.enable_photo_strip = True
            self.photo_count = 10
            self.photo_strip_width = 2.0
    
    def _find_photo_files(self):
        """Find JPEG files in the data/muac photos directory"""
        try:
            # Look for data/muac photos directory relative to output directory
            data_dir = Path(self.output_dir).parent / "data" / "muac photos"
            
            if not data_dir.exists():
                # Try alternative paths
                data_dir = Path(self.output_dir) / "data" / "muac photos"
                if not data_dir.exists():
                    data_dir = Path("data") / "muac photos"
                    if not data_dir.exists():
                        # Try without spaces in case of different naming
                        data_dir = Path(self.output_dir).parent / "data" / "muac_photos"
                        if not data_dir.exists():
                            data_dir = Path("data") / "muac_photos"
            
            if not data_dir.exists():
                self.log("Warning: MUAC photos directory not found. Skipping photo strip.")
                self.log("Expected location: data/muac photos/ or data/muac_photos/")
                return []
            
            # Find all JPEG files
            photo_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
            photo_files = []
            
            for pattern in photo_patterns:
                photo_files.extend(glob.glob(str(data_dir / pattern)))
            
            self.log(f"Found {len(photo_files)} photo files in {data_dir}")
            return photo_files
            
        except Exception as e:
            self.log(f"Error finding photo files: {str(e)}")
            return []
    
    def _select_photos_with_priority(self, photo_files, sample_size):
        """Select photos prioritizing red/yellow/green prefixed files, then random"""
        if not photo_files:
            return []
        
        # Separate priority photos (red, yellow, green prefixed) from others
        priority_photos = []
        other_photos = []
        
        for photo_path in photo_files:
            filename = Path(photo_path).name.lower()
            if filename.startswith(('red', 'yellow', 'green')):
                priority_photos.append(photo_path)
            else:
                other_photos.append(photo_path)
        
        self.log(f"Found {len(priority_photos)} priority photos (red/yellow/green)")
        self.log(f"Found {len(other_photos)} other photos")
        
        # Select photos with priority system
        selected_photos = []
        
        # First: take all priority photos if we have room, or sample them if too many
        if len(priority_photos) <= sample_size:
            selected_photos.extend(priority_photos)
            remaining_slots = sample_size - len(priority_photos)
            self.log(f"Using all {len(priority_photos)} priority photos")
        else:
            # Too many priority photos, randomly sample from them
            selected_photos.extend(random.sample(priority_photos, sample_size))
            remaining_slots = 0
            self.log(f"Randomly selected {sample_size} from {len(priority_photos)} priority photos")
        
        # Second: fill remaining slots with random other photos
        if remaining_slots > 0 and other_photos:
            additional_photos = random.sample(other_photos, min(remaining_slots, len(other_photos)))
            selected_photos.extend(additional_photos)
            self.log(f"Added {len(additional_photos)} random photos to fill remaining slots")
        
        # Shuffle the final list so priority photos aren't always on top
        random.shuffle(selected_photos)
        
        return selected_photos
    
    def _load_and_resize_photo(self, photo_path, target_width_px, max_height_px):
        """Load and resize a photo to fit the strip dimensions"""
        try:
            with Image.open(photo_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate aspect ratio
                aspect_ratio = img.width / img.height
                
                # Calculate target dimensions
                if aspect_ratio > 1:  # Landscape
                    new_width = target_width_px
                    new_height = int(target_width_px / aspect_ratio)
                else:  # Portrait or square
                    new_height = min(max_height_px, int(target_width_px / aspect_ratio))
                    new_width = int(new_height * aspect_ratio)
                
                # Resize image
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to numpy array for matplotlib
                return np.array(img_resized)
                
        except Exception as e:
            self.log(f"Error loading photo {photo_path}: {str(e)}")
            return None
    
    def _add_photo_strip(self, fig, gs_main, total_rows_needed):
        """Add vertical photo strip to the right of the main grid"""
        if not self.enable_photo_strip:
            return
            
        # Find photo files
        photo_files = self._find_photo_files()
        if not photo_files:
            self.log("No photos found, skipping photo strip")
            return
        
        # Randomly sample photos with priority for red/yellow/green
        sample_size = min(self.photo_count, len(photo_files))
        selected_photos = self._select_photos_with_priority(photo_files, sample_size)
        
        # Calculate photo strip dimensions
        strip_width_px = int(self.photo_strip_width * self.image_dpi)
        max_photo_height_px = int(strip_width_px * 1.2)  # Allow some variation in height
        
        # Create subplot for photo strip
        gs_photos = gridspec.GridSpecFromSubplotSpec(
            sample_size + 1, 1,  # +1 for header
            gs_main[:, 1],  # Right column of main grid
            hspace=0.02
        )
        
        # Add header
        ax_header = fig.add_subplot(gs_photos[0, 0])
        
        # Add header - first line bold
        ax_header.text(0.5, 0.7, 'Sample MUAC Photos', 
                      ha='center', va='center', fontsize=10, fontweight='bold',
                      transform=ax_header.transAxes)
        
        # Add header - remaining lines normal weight
        ax_header.text(0.5, 0.3, 'One time- and gps-\nstamped photo taken \nfor every child', 
                      ha='center', va='center', fontsize=10, fontweight='normal',
                      style='italic', color='#666666',
                      transform=ax_header.transAxes)
        
        ax_header.axis('off')
        
        # Add photos
        loaded_photos = []
        for i, photo_path in enumerate(selected_photos):
            photo_array = self._load_and_resize_photo(photo_path, strip_width_px, max_photo_height_px)
            if photo_array is not None:
                loaded_photos.append(photo_array)
        
        # Display photos
        for i, photo_array in enumerate(loaded_photos):
            if i + 1 < sample_size + 1:  # Make sure we don't exceed grid
                ax_photo = fig.add_subplot(gs_photos[i + 1, 0])
                ax_photo.imshow(photo_array, aspect='auto')
                ax_photo.axis('off')
                
                # Add thin border
                for spine in ax_photo.spines.values():
                    spine.set_color('lightgray')
                    spine.set_linewidth(0.5)
                    spine.set_visible(True)
        
        self.log(f"Added photo strip with {len(loaded_photos)} photos")
            
    def _filter_and_organize_data(self):
        """Filter for sufficient data and organize by problematic status and quartiles"""
        self.log("Filtering data for sufficient records...")
        
        # Filter for sufficient data only
        sufficient_df = self.df[self.df['data_sufficiency'] == 'SUFFICIENT'].copy()
        self.log(f"Found {len(sufficient_df)} records with sufficient data (from {len(self.df)} total)")
        
        if len(sufficient_df) == 0:
            raise ValueError("No records found with data_sufficiency = 'SUFFICIENT'")
        
        # Fill NaN values in MUAC columns with 0
        for col in self.muac_columns:
            if col in sufficient_df.columns:
                sufficient_df[col] = sufficient_df[col].fillna(0)
        
        # Handle flag_problematic column - convert to boolean
        sufficient_df['flag_problematic_bool'] = sufficient_df['flag_problematic'].astype(str).str.upper() == 'TRUE'
        
        # Separate problematic and non-problematic
        problematic = sufficient_df[sufficient_df['flag_problematic_bool'] == True].copy()
        non_problematic = sufficient_df[sufficient_df['flag_problematic_bool'] == False].copy()
        
        self.log(f"Problematic cases: {len(problematic)}")
        self.log(f"Non-problematic cases: {len(non_problematic)}")
        
        # Organize sections with proper labels
        sections = []
        
        # First: Flagged cases (attention-grabbing at top)
        if len(problematic) > 0:
            problematic_sorted = problematic.sort_values('valid_muac_count', ascending=False)
            sections.append({
                'title': f'Flagged Distributions ({len(problematic_sorted)} FLWs)',
                'data': problematic_sorted,
                'background_color': '#ffeeee',
                'border_color': 'red'
            })
        
        # Then: Non-problematic cases in ascending order using standard ranges
        if len(non_problematic) > 0:
            # Define standard ranges
            ranges = [
                (0, 199, "FLWs with <200 MUAC readings"),
                (200, 499, "FLWs with 200-500 MUAC readings"), 
                (500, 999, "FLWs with 500-1000 MUAC readings"),
                (1000, float('inf'), "FLWs with >1000 valid MUAC readings")
            ]
            
            for min_val, max_val, title_prefix in ranges:
                if max_val == float('inf'):
                    range_data = non_problematic[non_problematic['valid_muac_count'] >= min_val]
                else:
                    range_data = non_problematic[
                        (non_problematic['valid_muac_count'] >= min_val) & 
                        (non_problematic['valid_muac_count'] <= max_val)
                    ]
                
                if len(range_data) > 0:
                    range_data_sorted = range_data.sort_values('valid_muac_count', ascending=False)
                    sections.append({
                        'title': f'{title_prefix} ({len(range_data_sorted)} FLWs)',
                        'data': range_data_sorted,
                        'background_color': '#f5fff5',
                        'border_color': 'lightgreen'
                    })
                    self.log(f"Range {min_val}-{max_val}: {len(range_data_sorted)} cases")
        
        return sections
    
    def _create_sparkline_data(self, row):
        """Extract MUAC distribution data for a single row"""
        values = []
        for col in self.muac_columns:
            if col in row.index:
                values.append(row[col] if pd.notna(row[col]) else 0)
            else:
                values.append(0)
        return values
    
    def _generate_sparkline_grid(self, sections):
        """Generate the sparkline grid visualization with section headers and optional photo strip"""
        cols = self.grid_cols
        
        # Calculate space needed for each section
        section_layouts = []
        total_rows_needed = 0
        
        for i, section in enumerate(sections):
            num_charts = len(section['data'])
            charts_rows = max(1, num_charts // cols + (1 if num_charts % cols > 0 else 0))
            
            # Add extra space for flagged sections that have subtitle
            if 'Flagged' in section['title']:
                header_rows = 2  # Two rows for title + subtitle
            else:
                header_rows = 1  # One row for section title
            
            # Add spacing row above section headers (except for first section)
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
        
        self.log(f"Creating sectioned grid: {total_rows_needed} total rows x {cols} columns")
        
        # Create figure with GridSpec for main content and photo strip
        fig_width = max(15, cols * 0.8)
        if self.enable_photo_strip:
            fig_width += self.photo_strip_width  # Add width for photo strip
        
        fig_height = max(10, total_rows_needed * 0.4)
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.image_dpi)
        
        # Create main grid spec
        if self.enable_photo_strip:
            # Calculate width ratios: main grid gets most space, photo strip gets fixed width
            main_width_ratio = fig_width - self.photo_strip_width
            photo_width_ratio = self.photo_strip_width
            width_ratios = [main_width_ratio, photo_width_ratio]
            
            gs_main = gridspec.GridSpec(total_rows_needed, 2, figure=fig, 
                                      width_ratios=width_ratios, wspace=0.05)
            
            # Create nested GridSpec for the sparkline grid (left column)
            gs_sparklines = gridspec.GridSpecFromSubplotSpec(
                total_rows_needed, cols, gs_main[:, 0], hspace=0.1, wspace=0.02
            )
            
            # Create subplots for sparklines using the nested grid
            axes = []
            for row in range(total_rows_needed):
                row_axes = []
                for col in range(cols):
                    ax = fig.add_subplot(gs_sparklines[row, col])
                    row_axes.append(ax)
                axes.append(row_axes)
            
            # Convert to numpy array for easier indexing
            axes = np.array(axes)
            
        else:
            # Original layout without photo strip
            axes = []
            for row in range(total_rows_needed):
                row_axes = []
                for col in range(cols):
                    ax = plt.subplot(total_rows_needed, cols, row * cols + col + 1)
                    row_axes.append(ax)
                axes.append(row_axes)
            axes = np.array(axes)
        
        # Process each section (same as before)
        for layout in section_layouts:
            section = layout['section']
            section_data = section['data']
            start_row = layout['start_row']
            
            # Skip spacing row if present
            if layout['spacing_rows'] > 0:
                # Hide spacing row
                for col in range(cols):
                    ax = axes[start_row, col]
                    ax.axis('off')
                start_row += layout['spacing_rows']
            
            # Handle section headers
            if 'Flagged' in section['title']:
                # Two-row header for flagged sections
                # First row: Main title
                for col in range(cols):
                    ax = axes[start_row, col]
                    
                    if col == 0:  # Only add text to first column
                        ax.text(0.05, 0.5, section['title'], 
                               transform=ax.transAxes, fontsize=12, fontweight='bold',
                               verticalalignment='center')
                    
                    # Clean up header axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(left=False, right=False, top=False, bottom=False, 
                                  labelleft=False, labelright=False, labeltop=False, labelbottom=False)
                    ax.axis('off')
                
                # Second row: Subtitle
                for col in range(cols):
                    ax = axes[start_row + 1, col]
                    
                    if col == 0:  # Only add text to first column
                        subtitle_text = "Distribution flagged if any of following hold: <5 bins, not continually increasing to peak, not continually decreasing from peak, 3+ bin plateau"
                        ax.text(0.05, 0.5, subtitle_text, 
                               transform=ax.transAxes, fontsize=10, fontweight='normal',
                               verticalalignment='center', style='italic', color='#666666')
                    
                    # Clean up header axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(left=False, right=False, top=False, bottom=False, 
                                  labelleft=False, labelright=False, labeltop=False, labelbottom=False)
                    ax.axis('off')
            else:
                # Single-row header for non-flagged sections
                for col in range(cols):
                    ax = axes[start_row, col]
                    
                    if col == 0:  # Only add text to first column
                        ax.text(0.05, 0.5, section['title'], 
                               transform=ax.transAxes, fontsize=12, fontweight='bold',
                               verticalalignment='center')
                    
                    # Clean up header axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(left=False, right=False, top=False, bottom=False, 
                                  labelleft=False, labelright=False, labeltop=False, labelbottom=False)
                    ax.axis('off')
            
            # Add charts for this section
            chart_idx = 0
            charts_start_row = start_row + layout['header_rows']
            
            for charts_row in range(layout['charts_rows']):
                for col in range(cols):
                    if chart_idx >= len(section_data):
                        # Hide unused chart positions
                        ax = axes[charts_start_row + charts_row, col]
                        ax.axis('off')
                        continue
                    
                    ax = axes[charts_start_row + charts_row, col]
                    data_row = section_data.iloc[chart_idx]
                    
                    # Get MUAC distribution data
                    muac_values = self._create_sparkline_data(data_row)
                    x_positions = range(len(muac_values))
                    
                    # Create bar chart sparkline
                    ax.bar(x_positions, muac_values, width=0.8, color='#2E5984', alpha=0.8, edgecolor='none')
                    
                    # Apply section styling
                    ax.set_facecolor(section['background_color'])
                    for spine in ax.spines.values():
                        spine.set_color(section['border_color'])
                        spine.set_linewidth(1.0 if section['border_color'] == 'lightgreen' else 1.5)
                    
                    # Completely remove all axis elements
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(left=False, right=False, top=False, bottom=False, 
                                  labelleft=False, labelright=False, labeltop=False, labelbottom=False)
                    
                    # Set limits but hide axes
                    max_val = max(muac_values) if max(muac_values) > 0 else 1
                    ax.set_ylim(0, max_val * 1.1)
                    ax.set_xlim(-0.5, len(muac_values) - 0.5)
                    
                    chart_idx += 1
        
        # Add photo strip if enabled
        if self.enable_photo_strip:
            self._add_photo_strip(fig, gs_main, total_rows_needed)
        
        # Add overall title with counts
        total_charts = sum(len(section['data']) for section in sections)
        total_muac_readings = sum(
            section['data']['valid_muac_count'].sum() 
            for section in sections
        )
        title_text = f'MUAC Distribution Analysis:  {total_charts} FLWs; {total_muac_readings:,} MUAC Readings'
        if self.enable_photo_strip:
            title_text 
        fig.suptitle(title_text, fontsize=14, y=0.98)
        
        # Adjust layout with proper spacing for headers
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.1, wspace=0.02)
        
        return fig
    
    def generate(self):
        """Generate the MUAC sparkline grid report"""
        try:
            # Get parameters from UI
            self._get_parameters()
            self.log(f"Parameters: {self.grid_cols} columns, {self.image_dpi} DPI")
            if self.enable_photo_strip:
                self.log(f"Photo strip: {self.photo_count} photos, {self.photo_strip_width}\" wide")
            
            # Filter and organize data into sections
            sections = self._filter_and_organize_data()
            
            # Generate the sparkline grid
            self.log("Generating sparkline grid...")
            fig = self._generate_sparkline_grid(sections)
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.output_dir) / f"muac_sparklines_{timestamp}.png"
            
            self.log(f"Saving to: {output_file.name}")
            fig.savefig(output_file, dpi=self.image_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.log("? MUAC sparkline grid generated successfully!")
            
            return [str(output_file)]
            
        except Exception as e:
            self.log(f"? Error generating MUAC sparklines: {str(e)}")
            raise

# For integration with the reports package
# This should be added to your reports/__
