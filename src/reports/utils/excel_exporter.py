"""Excel export functionality with frozen panes and formatting"""

import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


class ExcelExporter:
    def __init__(self, log_func):
        self.log = log_func
    
    def export_to_excel(self, data_dict, output_path):
        """
        Export multiple DataFrames to Excel with separate tabs
        
        Args:
            data_dict: Dictionary where keys are tab names, values are DataFrames
            output_path: Path for the output Excel file
        """
        try:
            wb = Workbook()
            # Remove default sheet
            wb.remove(wb.active)
            
            for tab_name, df in data_dict.items():
                if df is None or len(df) == 0:
                    continue
                
                # Create worksheet
                ws = wb.create_sheet(title=self._clean_sheet_name(tab_name))
                
                # Add data
                for r in dataframe_to_rows(df, index=False, header=True):
                    ws.append(r)
                
                # Format the sheet
                self._format_worksheet(ws, df)
            
            # Save the workbook
            wb.save(output_path)
            self.log(f"Excel export complete: {len(data_dict)} tabs created")
            return output_path
            
        except Exception as e:
            self.log(f"Error creating Excel file: {str(e)}")
            return None
    
    def _clean_sheet_name(self, name):
        """Clean sheet name to be Excel-compatible"""
        # Excel sheet names can't exceed 31 chars and can't contain certain characters
        invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
        clean_name = name
        for char in invalid_chars:
            clean_name = clean_name.replace(char, '_')
        return clean_name[:31]
    
    def _format_worksheet(self, ws, df):
        """Apply formatting to worksheet"""
        if ws.max_row == 0:
            return
        
        # Format header row
        header_fill = PatternFill(start_color='E6E6FA', end_color='E6E6FA', fill_type='solid')
        header_font = Font(bold=True)
        
        for cell in ws[1]:  # First row
            cell.fill = header_fill
            cell.font = header_font
        
        # Set frozen panes
        # Freeze header row and first few columns (typically ID columns)
        freeze_col = 'C'  # Default to column C (first 2 columns frozen)
        
        # Adjust freeze column based on content
        if 'flw_id' in df.columns or any('id' in col.lower() for col in df.columns[:3]):
            freeze_col = 'D' if len(df.columns) > 3 else 'C'
        
        ws.freeze_panes = f'{freeze_col}2'  # Freeze first columns and header row
        
        # Auto-adjust column widths (with reasonable limits)
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            # Set width with reasonable bounds
            adjusted_width = min(max(max_length + 2, 10), 50)
            ws.column_dimensions[column_letter].width = adjusted_width


