"""
Cross-validation strategies for time series data.

This module provides various CV strategies to test reliability
against the public leaderboard.
"""

from typing import List, Tuple, Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.utils import get_logger

logger = get_logger(name="cv_strategy", level="INFO")


class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit with purge gap to prevent data leakage.
    
    Parameters
    ----------
    n_splits : int
        Number of splits
    purge_gap : int
        Number of periods to purge after train set (default: 0)
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 0):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold sizes
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Train: from start to current fold
            train_end = n_samples - test_size * (self.n_splits - i)
            
            # Purge gap
            purge_end = min(train_end + self.purge_gap, n_samples)
            
            # Validation: next test_size samples after purge
            val_start = purge_end
            val_end = min(val_start + test_size, n_samples)
            
            if val_end <= val_start:
                continue
                
            train_idx = indices[:train_end]
            val_idx = indices[val_start:val_end]
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class RollingWindowSplit:
    """
    Rolling window cross-validation.
    
    Parameters
    ----------
    n_splits : int
        Number of splits
    train_size : int
        Size of training window
    test_size : int
        Size of validation window
    """
    
    def __init__(self, n_splits: int = 5, train_size: int = 1000, test_size: int = 200):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate step size
        total_window = self.train_size + self.test_size
        max_start = n_samples - total_window
        step = max(1, max_start // (self.n_splits - 1)) if self.n_splits > 1 else max_start
        
        for i in range(self.n_splits):
            start = min(i * step, max_start)
            train_end = start + self.train_size
            val_end = min(train_end + self.test_size, n_samples)
            
            if val_end <= train_end:
                continue
                
            train_idx = indices[start:train_end]
            val_idx = indices[train_end:val_end]
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CustomHoldoutSplit:
    """
    Simple holdout split for last N periods.
    
    Parameters
    ----------
    holdout_size : int
        Number of last periods for validation
    """
    
    def __init__(self, holdout_size: int = 180):
        self.holdout_size = holdout_size
        
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate single train/validation split."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        split_idx = n_samples - self.holdout_size
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1


class PurgedWalkForwardSplit:
    """
    Purged Walk-Forward Split for time series.
    
    Combines walk-forward validation with purge gap to prevent data leakage.
    Uses fixed training window to focus on recent data patterns.
    
    Parameters
    ----------
    n_splits : int
        Number of splits
    train_size : int
        Fixed training window size
    test_size : int
        Validation window size
    purge_gap : int
        Gap between train and validation to prevent leakage (default: 5)
        
    Examples
    --------
    >>> cv = PurgedWalkForwardSplit(n_splits=3, train_size=100, test_size=30, purge_gap=5)
    >>> for train_idx, val_idx in cv.split(X):
    ...     print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        train_size: int = 2000,
        test_size: int = 500,
        purge_gap: int = 5
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged walk-forward splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate total window needed
        total_window = self.train_size + self.purge_gap + self.test_size
        max_start = n_samples - total_window
        
        if max_start < 0:
            raise ValueError(
                f"Not enough data ({n_samples}) for given window sizes "
                f"(need at least {total_window})"
            )
        
        # Calculate step size for walk-forward
        step = max(1, max_start // (self.n_splits - 1)) if self.n_splits > 1 else 0
        
        for i in range(self.n_splits):
            # Start position for this fold
            start = min(i * step, max_start)
            
            # Training window [start, start+train_size)
            train_start = start
            train_end = start + self.train_size
            
            # Purge gap [train_end, train_end+purge_gap)
            purge_end = train_end + self.purge_gap
            
            # Validation window [purge_end, purge_end+test_size)
            val_start = purge_end
            val_end = min(val_start + self.test_size, n_samples)
            
            if val_end <= val_start:
                logger.warning(f"Fold {i+1}: Not enough validation data, skipping")
                continue
            
            train_idx = indices[train_start:train_end]
            val_idx = indices[val_start:val_end]
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def get_cv_strategy(strategy_name: str, **kwargs):
    """
    Factory function to get CV strategy.
    
    Parameters
    ----------
    strategy_name : str
        Name of CV strategy:
        - 'timeseries': sklearn TimeSeriesSplit
        - 'purged': PurgedTimeSeriesSplit
        - 'rolling': RollingWindowSplit
        - 'holdout': CustomHoldoutSplit
        - 'purged_walkforward': PurgedWalkForwardSplit
    **kwargs
        Parameters for the CV strategy
        
    Returns
    -------
    CV strategy object
    """
    if strategy_name == 'timeseries':
        n_splits = kwargs.get('n_splits', 5)
        return TimeSeriesSplit(n_splits=n_splits)
    
    elif strategy_name == 'purged':
        n_splits = kwargs.get('n_splits', 5)
        purge_gap = kwargs.get('purge_gap', 0)
        return PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    
    elif strategy_name == 'rolling':
        n_splits = kwargs.get('n_splits', 5)
        train_size = kwargs.get('train_size', 1000)
        test_size = kwargs.get('test_size', 200)
        return RollingWindowSplit(n_splits=n_splits, train_size=train_size, test_size=test_size)
    
    elif strategy_name == 'holdout':
        holdout_size = kwargs.get('holdout_size', 180)
        return CustomHoldoutSplit(holdout_size=holdout_size)
    
    elif strategy_name == 'purged_walkforward':
        n_splits = kwargs.get('n_splits', 5)
        train_size = kwargs.get('train_size', 2000)
        test_size = kwargs.get('test_size', 500)
        purge_gap = kwargs.get('purge_gap', 5)
        return PurgedWalkForwardSplit(
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            purge_gap=purge_gap
        )
    
    else:
        raise ValueError(f"Unknown CV strategy: {strategy_name}")


def evaluate_cv_reliability(
    data: pd.DataFrame,
    cv_strategy,
    strategy_name: str,
    metric_calculator,
    pseudo_lb_score: float
) -> dict:
    """
    Evaluate CV strategy reliability.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    cv_strategy
        CV strategy object
    strategy_name : str
        Name of the strategy
    metric_calculator
        Metric calculator (CompetitionMetric)
    pseudo_lb_score : float
        Score on pseudo public LB
        
    Returns
    -------
    dict
        Reliability metrics
    """
    allocations = np.ones(len(data))
    forward_returns = data['forward_returns'].values
    risk_free_rate = data['risk_free_rate'].values
    
    # Collect all OOF predictions
    oof_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(data)):
        # Baseline: Position = 1.0
        val_allocations = np.ones(len(val_idx))
        val_forward_returns = forward_returns[val_idx]
        val_risk_free_rate = risk_free_rate[val_idx]
        
        # Calculate score
        result = metric_calculator.calculate_score(
            allocations=val_allocations,
            forward_returns=val_forward_returns,
            market_returns=val_forward_returns,
            risk_free_rate=val_risk_free_rate
        )
        
        oof_scores.append(result['score'])
        
        logger.info(f"  {strategy_name} Fold {fold_idx + 1}: Score = {result['score']:.6f}")
    
    # Calculate statistics
    mean_oof = np.mean(oof_scores)
    std_oof = np.std(oof_scores)
    
    # Compare with pseudo LB
    scale_ratio = mean_oof / pseudo_lb_score if pseudo_lb_score != 0 else 0
    absolute_diff = abs(mean_oof - pseudo_lb_score)
    
    return {
        'strategy': strategy_name,
        'mean_oof_score': mean_oof,
        'std_oof_score': std_oof,
        'pseudo_lb_score': pseudo_lb_score,
        'scale_ratio': scale_ratio,
        'absolute_diff': absolute_diff,
        'n_folds': len(oof_scores),
        'fold_scores': oof_scores
    }
