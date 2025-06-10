"""Statistical analysis functions."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple


class StatisticalAnalysis:
    """Perform statistical tests and analysis."""
    
    def __init__(self):
        pass
    
    def t_test(self, sample1: pd.Series, sample2: pd.Series, 
              alternative: str = 'two-sided') -> Dict[str, float]:
        """Perform independent t-test between two samples."""
        statistic, p_value = stats.ttest_ind(sample1.dropna(), sample2.dropna(), 
                                           alternative=alternative)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def chi_square_test(self, observed: pd.Series, expected: Optional[pd.Series] = None) -> Dict[str, float]:
        """Perform chi-square goodness of fit test."""
        if expected is None:
            # Equal distribution
            expected = np.full(len(observed), observed.sum() / len(observed))
        
        statistic, p_value = stats.chisquare(observed, expected)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def normality_test(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test."""
        data_clean = data.dropna()
        
        if len(data_clean) < 3:
            return {'error': 'Insufficient data for normality test'}
        
        statistic, p_value = stats.shapiro(data_clean)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'test_used': 'Shapiro-Wilk'
        }
