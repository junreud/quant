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


def risk_adjusted_allocation(
    return_pred: Union[np.ndarray, float],
    risk_pred: Union[np.ndarray, float],
    k: float = 1.0,
    max_position: float = 2.0
) -> Union[np.ndarray, float]:
    """
    Risk-Adjusted Allocation (Dual-Model).
    Position = k * (Return / Risk)
    
    Parameters
    ----------
    return_pred : array or float
        수익률 예측값
    risk_pred : array or float
        변동성(Risk) 예측값
    k : float
        스케일링 상수 (Leverage Factor)
    max_position : float
        최대 포지션 제한
        
    Returns
    -------
    array or float
        Allocation [0, max_position]
    """
    # 0 나누기 방지
    safe_risk = np.maximum(risk_pred, 1e-6)
    
    # 샤프 비율 기반 할당
    raw_allocation = k * (return_pred / safe_risk)
    
    # 방향성 유지 (Return이 음수면 포지션도 음수? 아니면 0?)
    # 규칙: 0 <= Position <= 2 이므로, Return이 음수면 0으로 clip됨.
    # 하지만 raw_allocation 자체가 음수일 수 있음.
    
    return np.clip(raw_allocation, 0.0, max_position)


# 기본 전략 선택
default_allocation = smart_allocation
