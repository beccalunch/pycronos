"""Descriptive statistics and analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class DescriptiveAnalysis:
    """Perform descriptive statistical analysis."""
    
    def __init__(self):
        pass
    
    def summary_statistics(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate summary statistics for numeric columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        return df[columns].describe()
    
    def correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    
    def value_counts(self, df: pd.DataFrame, column: str, normalize: bool = False) -> pd.Series:
        """Get value counts for a specific column."""
        return df[column].value_counts(normalize=normalize)
    
    def missing_value_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        return {
            'missing_by_column': missing_count.to_dict(),
            'missing_percent_by_column': missing_percent.to_dict(),
            'total_missing_values': missing_count.sum(),
            'rows_with_missing': df.isnull().any(axis=1).sum()
        }
