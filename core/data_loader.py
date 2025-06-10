"""Data loading utilities for various file formats."""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any


class DataLoader:
    """Handle loading data from various sources and formats."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']
    
    def load_csv(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath, **kwargs)
    
    def load_excel(self, filepath: Union[str, Path], sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from Excel file."""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    def load_json(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        return pd.read_json(filepath, **kwargs)
    
    def load_parquet(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from Parquet file."""
        return pd.read_parquet(filepath, **kwargs)
    
    def auto_load(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Automatically detect file format and load data."""
        file_path = Path(filepath)
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv(filepath, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        elif suffix == '.json':
            return self.load_json(filepath, **kwargs)
        elif suffix == '.parquet':
            return self.load_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
