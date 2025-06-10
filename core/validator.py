"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class DataValidator:
    """Validate data quality and consistency."""
    
    def __init__(self):
        pass
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset."""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        return {
            'missing_count': missing_count.to_dict(),
            'missing_percent': missing_percent.to_dict(),
            'total_missing': missing_count.sum(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist()
        }
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows in the dataset."""
        duplicate_count = df.duplicated().sum()
        duplicate_percent = (duplicate_count / len(df)) * 100
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_percent': duplicate_percent,
            'has_duplicates': duplicate_count > 0
        }
    
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data types of columns."""
        return {
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
