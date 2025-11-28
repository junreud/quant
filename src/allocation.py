"""
Allocation Strategy Module

예측값을 포지션(allocation)으로 변환하는 중앙화된 로직.
"""

import numpy as np
from typing import Union


def smart_allocation(
    y_pred: Union[np.ndarray, float],
    center: float = 1.0,
    sensitivity: float = 20
) -> Union[np.ndarray, float]:
    """
    예측값을 allocation으로 변환 (Proportional mapping).
    
    Parameters
    ----------
    y_pred : array or float
        예측된 수익률
    center : float
        중심값 (기본 1.0 = 시장 추종)
    sensitivity : float
        민감도 (높을수록 aggressive)
        
    Returns
    -------
    array or float
        Allocation [0, 2]
        
    Examples
    --------
    >>> smart_allocation(0.01, center=1.0, sensitivity=20)
    1.2
    
    >>> smart_allocation(-0.01, center=1.0, sensitivity=20)
    0.8
    """
    allocation = center + y_pred * sensitivity
    return np.clip(allocation, 0.0, 2.0)


def binary_allocation(
    y_pred: Union[np.ndarray, float],
    positive_val: float = 1.5,
    negative_val: float = 0.5
) -> Union[np.ndarray, float]:
    """
    예측값을 binary allocation으로 변환.
    
    Parameters
    ----------
    y_pred : array or float
        예측된 수익률
    positive_val : float
        양수일 때 값
    negative_val : float
        음수일 때 값
        
    Returns
    -------
    array or float
        Allocation [0, 2]
    """
    if isinstance(y_pred, (int, float)):
        return positive_val if y_pred > 0 else negative_val
    else:
        return np.where(y_pred > 0, positive_val, negative_val)


# 기본 전략 선택
default_allocation = smart_allocation
