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


def dynamic_risk_allocation(
    return_pred: Union[np.ndarray, float],
    risk_pred: Union[np.ndarray, float],
    rolling_mean: Union[np.ndarray, float],
    rolling_std: Union[np.ndarray, float],
    k: float = 0.5,
    max_position: float = 2.0
) -> Union[np.ndarray, float]:
    """
    동적 리스크 조정 자산 배분 (Z-Score 기반).
    
    절대적인 수익률 예측값 대신 상대적 강도(Z-Score)를 사용합니다.
    포지션 = k * (Z-Score / 리스크)
    
    Parameters
    ----------
    return_pred : array or float
        현재 수익률 예측값
    risk_pred : array or float
        현재 리스크 예측값
    rolling_mean : array or float
        과거 수익률 예측값들의 이동 평균
    rolling_std : array or float
        과거 수익률 예측값들의 이동 표준편차
    k : float
        스케일링 계수 (레버리지 팩터)
    max_position : float
        최대 포지션 제한
        
    Returns
    -------
    array or float
        자산 배분 비중 [0, max_position]
    """
    # 1. Z-Score (상대적 강도) 계산
    # 0으로 나누기 방지
    safe_std = np.maximum(rolling_std, 1e-8)
    z_score = (return_pred - rolling_mean) / safe_std
    
    # 2. 리스크 조정
    safe_risk = np.maximum(risk_pred, 1e-6)
    
    # 3. 자산 배분 로직
    # Z-Score 기반 배분
    # Z-Score는 "평소보다 얼마나 좋은가"를 시그마 단위로 나타냄.
    # Allocation = k * Z_Score
    # 예: k=0.5이고 Z-Score=1.0 (1시그마 호재)이면 포지션 50%.
    #     Z-Score=2.0 (2시그마 호재)이면 포지션 100%.
    
    # 기존 로직 (Signal / Risk)은 Signal이 너무 작아서 배분이 미미했음.
    # Z-Score는 스케일이 정규화되어 있으므로 k값으로 배분 조절이 용이함.
    
    raw_allocation = k * z_score
    
    return np.clip(raw_allocation, 0.0, max_position)


# 기본 전략 선택
default_allocation = smart_allocation


def triple_model_allocation(
    return_pred: Union[np.ndarray, float],
    risk_vol_pred: Union[np.ndarray, float],
    risk_market_pred: Union[np.ndarray, float],
    market_threshold: float = 0.0,
    k: float = config['allocation']['k'],
    max_position: float = 2.0
) -> Union[np.ndarray, float]:
    """
    3-Model Allocation (Return, Volatility, Market Regime).
    
    Logic:
    1. Base Allocation = k * (Return / Volatility)
    2. Market Regime Filter: If Market_Pred < Threshold, Force Position = 0.
    
    Parameters
    ----------
    return_pred : array or float
        Model A: Return Prediction
    risk_vol_pred : array or float
        Model B: Volatility Prediction (abs returns)
    risk_market_pred : array or float
        Model C: Market Regime Prediction (market returns)
    market_threshold : float
        Threshold for market regime (default 0.0)
    k : float
        Leverage factor
    max_position : float
        Max position limit
        
    Returns
    -------
    array or float
        Final Allocation
    """
    # 1. Base Allocation (Phase 1)
    safe_vol = np.maximum(risk_vol_pred, 1e-6)
    
    # Simple Risk Parity-like allocation
    # Position is proportional to Return/Risk
    base_allocation = k * (return_pred / safe_vol)
    
    # 2. Market Regime Filter (Phase 2)
    # If Market Return is expected to be negative, cut position.
    if isinstance(risk_market_pred, (int, float)):
        if risk_market_pred < market_threshold:
            base_allocation = 0.0
    else:
        # Vectorized operation
        base_allocation = np.where(risk_market_pred < market_threshold, 0.0, base_allocation)
        
    return np.clip(base_allocation, 0.0, max_position)
