"""
Feature Selection Module

피처 선택을 위한 유틸리티 클래스.
"""

import pandas as pd
import numpy as np
from typing import List, Union
import lightgbm as lgb
from src.utils import get_logger

logger = get_logger(name="feature_selection", level="INFO")

class FeatureSelector:
    """
    피처 선택 클래스.
    
    기능:
    1. 상관관계 기반 다중공선성 제거
    2. LGBM Feature Importance 기반 선택
    """
    
    def __init__(self):
        self.selected_features = None
        
    def remove_collinear(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        상관관계가 높은 피처 제거.
        
        Parameters
        ----------
        df : pd.DataFrame
            피처 데이터
        threshold : float
            상관관계 임계값 (절대값 기준)
            
        Returns
        -------
        List[str]
            선택된 피처 리스트
        """
        logger.info(f"🔍 Removing collinear features (threshold={threshold})...")
        
        # 상관계수 행렬 계산
        corr_matrix = df.corr().abs()
        
        # 상삼각행렬만 선택
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 임계값 넘는 컬럼 찾기
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected = [col for col in df.columns if col not in to_drop]
        
        logger.info(f"   Dropped {len(to_drop)} features. Remaining: {len(selected)}")
        return selected

    def select_by_importance(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50, 
                           lgbm_params: dict = None) -> List[str]:
        """
        LGBM Feature Importance 기반 피처 선택.
        
        Parameters
        ----------
        X : pd.DataFrame
            피처 데이터
        y : pd.Series
            타겟 데이터
        top_k : int
            선택할 상위 피처 개수
        lgbm_params : dict
            LGBM 파라미터 (없으면 기본값 사용)
            
        Returns
        -------
        List[str]
            선택된 피처 리스트
        """
        logger.info(f"🔍 Selecting top {top_k} features by LGBM importance...")
        
        if lgbm_params is None:
            lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'seed': 42
            }
            
        # LGBM Dataset
        dtrain = lgb.Dataset(X, label=y)
        
        # Train model (lightweight)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=100)
        
        # Get importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Select top k
        selected = importance.head(top_k)['feature'].tolist()
        
        logger.info(f"   Selected {len(selected)} features.")
        return selected

    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series, method: str = 'spearman', top_k: int = 20) -> List[str]:
        """
        상관관계 기반 피처 선택.
        
        Parameters
        ----------
        X : pd.DataFrame
            피처 데이터
        y : pd.Series
            타겟 데이터
        method : str
            상관계수 방법 ('pearson', 'spearman')
        top_k : int
            선택할 상위 피처 개수
            
        Returns
        -------
        List[str]
            선택된 피처 리스트
        """
        logger.info(f"🔍 Selecting top {top_k} features by {method} correlation...")
        
        # 데이터 병합 (인덱스 기준)
        # X와 y의 인덱스가 맞아야 함
        
        # 상관관계 계산
        corrs = X.corrwith(y, method=method).abs()
        
        # 상위 k개 선정
        selected = corrs.sort_values(ascending=False).head(top_k).index.tolist()
        
        logger.info(f"   Selected {len(selected)} features.")
        return selected
    def select_by_crash_divergence(self, X: pd.DataFrame, y: pd.Series, 
                                 crash_threshold_quantile: float = 0.05, 
                                 top_k: int = 20) -> List[str]:
        """
        시장 폭락(Crash) 시점과 평상시의 Feature 분포 차이(Divergence)가 큰 Feature 선택.
        
        Parameters
        ----------
        X : pd.DataFrame
            피처 데이터
        y : pd.Series
            타겟 데이터 (Market Returns)
        crash_threshold_quantile : float
            Crash로 정의할 하위 분위수 (예: 0.05 = 하위 5%)
        top_k : int
            선택할 상위 피처 개수
            
        Returns
        -------
        List[str]
            선택된 피처 리스트
        """
        logger.info(f"🔍 Selecting top {top_k} features by Crash Divergence (q={crash_threshold_quantile})...")
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Define Crash Mask
        crash_threshold = y_aligned.quantile(crash_threshold_quantile)
        crash_mask = y_aligned < crash_threshold
        
        n_crash = crash_mask.sum()
        logger.info(f"   Identified {n_crash} crash periods (Threshold: {crash_threshold:.4f})")
        
        if n_crash < 10:
            logger.warning("   Too few crash periods for reliable analysis. Returning empty list.")
            return []
            
        # Calculate Means
        crash_means = X_aligned[crash_mask].mean()
        normal_means = X_aligned[~crash_mask].mean()
        
        # Calculate Divergence (Z-score like difference)
        # (Crash Mean - Normal Mean) / Overall Std
        # Avoid division by zero
        overall_std = X_aligned.std() + 1e-8
        divergence = (crash_means - normal_means) / overall_std
        
        # Select top k by absolute divergence
        selected = divergence.abs().sort_values(ascending=False).head(top_k).index.tolist()
        
        logger.info(f"   Selected {len(selected)} features.")
        return selected
