"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union


class Preprocessor:
    """Handle data cleaning and preprocessing tasks."""
    
    def __init__(self):
        pass
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop', 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_copy = df.copy()
        
        if columns:
            df_subset = df_copy[columns]
        else:
            df_subset = df_copy
            
        if strategy == 'drop':
            return df_copy.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = df_subset.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            return df_copy
        elif strategy == 'fill_median':
            numeric_cols = df_subset.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
            return df_copy
        elif strategy == 'fill_mode':
            for col in df_subset.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            return df_copy
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        return df.drop_duplicates(subset=subset)
    
    def normalize_columns(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize numeric columns to 0-1 range."""
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            df_copy[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
        
        return df_copy
    
    def standardize_columns(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Standardize numeric columns to mean=0, std=1."""
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            df_copy[col] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
        
        return df_copy
