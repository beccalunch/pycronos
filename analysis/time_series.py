"""Time series analysis utilities."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple


class TimeSeriesAnalysis:
    """Perform time series analysis."""
    
    def __init__(self):
        pass
    
    def decompose_series(self, ts: pd.Series, period: Optional[int] = None, 
                        model: str = 'additive') -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(ts, period=period, model=model)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except ImportError:
            raise ImportError("statsmodels is required for time series decomposition")
    
    def calculate_moving_average(self, ts: pd.Series, window: int) -> pd.Series:
        """Calculate moving average of time series."""
        return ts.rolling(window=window).mean()
    
    def calculate_exponential_smoothing(self, ts: pd.Series, alpha: float = 0.3) -> pd.Series:
        """Calculate exponentially smoothed time series."""
        return ts.ewm(alpha=alpha).mean()
    
    def detect_outliers(self, ts: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in time series data."""
        if method == 'iqr':
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ts[(ts < lower_bound) | (ts > upper_bound)]
            
            return {
                'outliers': outliers,
                'outlier_indices': outliers.index.tolist(),
                'num_outliers': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
