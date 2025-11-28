"""
Data Preprocessing Pipeline

간단하고 재사용 가능한 전처리 파이프라인.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.utils import get_logger

logger = get_logger(name="preprocessing", level="INFO")


class DataPreprocessor:
    """
    데이터 전처리 파이프라인.
    
    사용법:
        preprocessor = DataPreprocessor(fillna_strategy='median')
        X, y = preprocessor.fit_transform(train_df)
        X_test = preprocessor.transform(test_df)
    """
    
    def __init__(self, fillna_strategy: str = 'median'):
        """
        Parameters
        ----------
        fillna_strategy : str
            결측치 처리 방법 ('median', 'mean', 'zero')
        """
        self.fillna_strategy = fillna_strategy
        self.fill_values = {}  # 학습된 fillna 값들
        self.feature_cols = None
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        전처리 파라미터 학습.
        
        Parameters
        ----------
        df : pd.DataFrame
            학습 데이터
        """
        # 피처 컬럼 정의
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_cols]
        
        # Fallback 값 학습 (ffill의 경우 첫 행 결측치 처리를 위해 필요)
        if self.fillna_strategy in ['median', 'ffill']:
            self.fill_values = X.median().to_dict()
        elif self.fillna_strategy == 'mean':
            self.fill_values = X.mean().to_dict()
        elif self.fillna_strategy == 'zero':
            self.fill_values = {col: 0 for col in self.feature_cols}
        elif str(self.fillna_strategy).lower() == 'false' or self.fillna_strategy is None:
            self.fill_values = {} # No filling
        else:
            raise ValueError(f"Unknown strategy: {self.fillna_strategy}")
        
        return self
    
    def transform(self, df: pd.DataFrame, return_target: bool = False) -> pd.DataFrame:
        """
        데이터 전처리.
        
        Parameters
        ----------
        df : pd.DataFrame
            변환할 데이터
        return_target : bool
            target도 함께 반환할지 여부
            
        Returns
        -------
        pd.DataFrame or tuple
            전처리된 피처 (또는 (X, y) tuple)
        """
        if self.feature_cols is None:
            raise ValueError("Call fit() first!")
        
        # 피처 추출
        X = df[self.feature_cols].copy()
        
        # 결측치 처리
        if str(self.fillna_strategy).lower() == 'false' or self.fillna_strategy is None:
            pass # Skip all filling
        else:
            if self.fillna_strategy == 'ffill':
                # 시계열 Forward Fill (단일 종목 가정)
                if 'date_id' in df.columns:
                    # 원본 인덱스 저장
                    original_index = X.index
                    # date_id 기준 정렬을 위해 임시로 추가
                    X['date_id'] = df['date_id']
                    X = X.sort_values('date_id')
                    
                    # Forward Fill 적용
                    X[self.feature_cols] = X[self.feature_cols].ffill()
                    
                    # date_id 제거 및 원래 순서 복구 (필요시)
                    # 주의: 시계열 데이터는 보통 시간순 처리가 중요하므로 정렬 상태 유지가 나을 수 있음
                    # 하지만 여기서는 입력 순서를 보장하기 위해 인덱스로 복구
                    X = X.drop(columns=['date_id'])
                    X = X.reindex(original_index)

                    logger.info("FFilled with date_id")
                else:
                    # date_id가 없으면 그냥 ffill (순서가 시간순이라 가정)
                    X = X.ffill()
                    logger.info("FFilled without date_id")
            
            # 나머지 결측치 (또는 ffill이 아닌 경우) 처리
            for col in self.feature_cols:
                if col in self.fill_values:
                    X[col] = X[col].fillna(self.fill_values[col])
            
            # 남은 NaN은 0으로 (안전장치)
            X = X.fillna(0)
        
        if return_target:
            if 'forward_returns' not in df.columns:
                raise ValueError("Target column 'forward_returns' not found!")
            y = df['forward_returns'].copy()
            # NaN 제거 (타겟이 없는 행은 학습에서 제외)
            valid_mask = ~y.isnull()
            return X[valid_mask], y[valid_mask]
        
        return X
    
    def fit_transform(self, df: pd.DataFrame, return_target: bool = True):
        """
        fit + transform
        
        Returns
        -------
        tuple or DataFrame
            return_target=True면 (X, y), 아니면 X
        """
        self.fit(df)
        if self.fillna_strategy == 'ffill':
            logger.info("FFilled with date_id") 
        elif self.fillna_strategy == 'median':
            logger.info("Filled with median")
        elif self.fillna_strategy == 'mean':
            logger.info("Filled with mean")
        elif self.fillna_strategy == 'zero':
            logger.info("Filled with zero")
        elif self.fillna_strategy == False:
            logger.info("Filled with False")

        return self.transform(df, return_target=return_target)
    
    def get_feature_names(self):
        """피처 이름 반환"""
        return self.feature_cols if self.feature_cols else []


# Legacy 함수들 (하위 호환성)
def load_and_prepare_data(
    train_path: str,
    fillna_strategy: str = "median"
) -> tuple:
    """
    Legacy function for backward compatibility.
    
    Returns
    -------
    tuple
        (X, y, feature_cols)
    """
    logger.info("📁 Loading data...")
    df = pd.read_csv(train_path)
    logger.info(f"   Shape: {df.shape}")
    
    preprocessor = DataPreprocessor(fillna_strategy=fillna_strategy)
    X, y = preprocessor.fit_transform(df, return_target=True)
    
    logger.info(f"   Features: {len(preprocessor.feature_cols)}")
    logger.info(f"   Target: forward_returns")
    logger.info(f"🔧 Filling missing values (strategy: {fillna_strategy})...")
    logger.info(f"✅ Final shape: X={X.shape}, y={y.shape}")
    
    return X, y, preprocessor.feature_cols

