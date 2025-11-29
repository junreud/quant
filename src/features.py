"""
Feature Engineering Pipeline

간단하고 재사용 가능한 피처 엔지니어링 파이프라인.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from src.utils import get_logger

logger = get_logger(name="feature_engineering", level="INFO")

class FeatureEngineer:
    """
    피처 엔지니어링 파이프라인.
    
    사용법:
        fe = FeatureEngineer()
        X_train = fe.fit_transform(train_df)
        X_test = fe.transform(test_df)
    """
    
    def __init__(
        self,
        add_interactions: bool = False,
        use_time_series_features: bool = True,
        use_advanced_features: bool = True,
        use_market_regime_features: bool = True,
    ):
        """
        Parameters
        ----------
        add_interactions : bool
            피처 간 상호작용 추가 여부
        use_time_series_features : bool
            시계열 피처(Lags, Rolling, Momentum) 추가 여부
        use_advanced_features : bool
            고급 시계열 피처(Z-Score, RSI, MACD, Log Returns) 추가 여부
        use_market_regime_features : bool
            시장 국면 피처(Volatility Regime, Trend Regime) 추가 여부
        """
        self.add_interactions = add_interactions
        self.use_time_series_features = use_time_series_features
        self.use_advanced_features = use_advanced_features
        self.use_market_regime_features = use_market_regime_features
        self.feature_cols = None
        self.generated_features = []
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        피처 컬럼명 학습.
        
        Parameters
        ----------
        df : pd.DataFrame
            학습 데이터
        """
        # 제외할 컬럼
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored', 'lagged_market_forward_excess_returns']
        
        # 피처 컬럼 추출
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        피처 변환.
        
        Parameters
        ----------
        df : pd.DataFrame
            변환할 데이터 (date_id 포함 권장)
            
        Returns
        -------
        pd.DataFrame
            변환된 피처 DataFrame
        """
        if self.feature_cols is None:
            raise ValueError("Call fit() first!")
        
        # 기본 피처
        X = df[self.feature_cols].copy()
        
        # 시계열 피처 (선택적)
        if self.use_time_series_features and 'date_id' in df.columns:
            X = self._add_time_series_features(X, df['date_id'])
            logger.info("Added time series features")
            
            # 고급 시계열 피처 (선택적, date_id 필요)
            if self.use_advanced_features:
                X = self._add_advanced_features(X, df['date_id'])
                logger.info("Added advanced features")
        
        # 시장 국면 피처 (선택적, date_id 필요)
        if self.use_market_regime_features and 'date_id' in df.columns:
            X = self._add_market_regime_features(X, df)
            logger.info("Added market regime features")

        # 상호작용 피처 (선택적)
        if self.add_interactions:
            X = self._add_interaction_features(X)
            logger.info("Added interaction features")
        
        # NaN 처리 (시계열 피처로 인한 결측치)
        # FeatureEngineer는 기본적으로 0으로 채우지만, 
        # 외부에서 제어할 수 있도록 개선하거나, Preprocessor의 전략을 따르는 것이 좋음.
        # 여기서는 일단 하드코딩된 0 채우기를 제거하고, 
        # 호출하는 쪽(Pipeline)에서 처리하도록 하거나, 
        # FeatureEngineer 생성자에 옵션을 추가해야 함.
        # 유저 요청: "false면 결측치 처리를 안하는거지"
        # 따라서 여기의 강제 fillna(0)은 제거하거나 옵션화해야 함.
        
        # 하위 호환성을 위해 기본적으로는 채우되, 옵션으로 제어
        if getattr(self, 'fill_na', True):
            X = X.fillna(0)
            logger.info("Filled missing values")
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit + transform"""
        return self.fit(df).transform(df)
    
    def create_risk_target(self, df: pd.DataFrame) -> pd.Series:
        """
        리스크 모델용 타겟 생성 (Absolute Returns).
        """
        if 'forward_returns' not in df.columns:
            raise ValueError("Target column 'forward_returns' not found!")
        return df['forward_returns'].abs()

    def _add_time_series_features(self, X: pd.DataFrame, date_id: pd.Series) -> pd.DataFrame:
        """
        기본 시계열 피처 추가 (Lags, Rolling, Momentum).
        """
        # 정렬을 위해 date_id 추가
        X = X.copy() # 원본 보존
        X['date_id'] = date_id
        X = X.sort_values('date_id')
        
        # 원본 피처 리스트 (이들에 대해서만 파생 피처 생성)
        base_cols = [c for c in X.columns if c != 'date_id']
        
        new_features = []

        # 1. Lags
        lags = [1, 2, 3, 5]
        for lag in lags:
            shifted = X[base_cols].shift(lag)
            shifted.columns = [f"{col}_lag_{lag}" for col in base_cols]
            new_features.append(shifted)
            
        # 2. Rolling Stats (Mean, Std)
        windows = [5, 10, 20]
        for window in windows:
            # Mean
            rolled_mean = X[base_cols].rolling(window=window).mean()
            rolled_mean.columns = [f"{col}_roll_mean_{window}" for col in base_cols]
            new_features.append(rolled_mean)
            
            # Std
            rolled_std = X[base_cols].rolling(window=window).std()
            rolled_std.columns = [f"{col}_roll_std_{window}" for col in base_cols]
            new_features.append(rolled_std)
            
        # 3. Momentum (ROC)
        for lag in [1, 5, 10]:
            diff = X[base_cols].diff(lag)
            diff.columns = [f"{col}_mom_{lag}" for col in base_cols]
            new_features.append(diff)

        # 한 번에 병합 (Fragmentation 방지)
        if new_features:
            X = pd.concat([X] + new_features, axis=1)

        # date_id 제거
        X = X.drop(columns=['date_id'])
        
        return X

    def _add_advanced_features(self, X: pd.DataFrame, date_id: pd.Series) -> pd.DataFrame:
        """
        고급 시계열 피처 추가 (Z-Score, RSI, MACD, Log Returns).
        """
        # 정렬을 위해 date_id 추가
        X = X.copy()
        X['date_id'] = date_id
        X = X.sort_values('date_id')
        
        base_cols = [c for c in X.columns if c != 'date_id' and not '_lag_' in c and not '_roll_' in c and not '_mom_' in c]
        
        new_features = []

        # 1. Rolling Z-Score (Standardization)
        # (Value - Mean) / Std
        windows = [20] # 20일 기준 (한달)
        for window in windows:
            for col in base_cols:
                mean_col = f"{col}_roll_mean_{window}"
                std_col = f"{col}_roll_std_{window}"
                
                # 이미 계산된 Rolling Mean/Std가 있으면 사용, 없으면 계산
                if mean_col in X.columns:
                    roll_mean = X[mean_col]
                else:
                    roll_mean = X[col].rolling(window=window).mean()
                    
                if std_col in X.columns:
                    roll_std = X[std_col]
                else:
                    roll_std = X[col].rolling(window=window).std()
                
                # Z-Score 계산 (분모 0 방지)
                z_score = (X[col] - roll_mean) / (roll_std + 1e-8)
                new_features.append(z_score.rename(f"{col}_zscore_{window}"))

        # 3. Technical Indicators (RSI, MACD, Bollinger Bands)
        for col in base_cols:
            # RSI (14)
            rsi = self._calculate_rsi(X[col], window=14)
            new_features.append(rsi.rename(f"{col}_rsi_14"))
            
            # MACD (12, 26, 9)
            macd, signal, hist = self._calculate_macd(X[col])
            new_features.append(macd.rename(f"{col}_macd"))
            new_features.append(signal.rename(f"{col}_macd_sig"))
            new_features.append(hist.rename(f"{col}_macd_hist"))
            
            # Bollinger Bands (20, 2)
            upper, lower = self._calculate_bollinger_bands(X[col], window=20, num_std=2)
            # 밴드폭 (Bandwidth) 및 위치 (Percent B)
            bb_width = (upper - lower) / (X[col].rolling(20).mean() + 1e-8)
            bb_pos = (X[col] - lower) / (upper - lower + 1e-8)
            new_features.append(bb_width.rename(f"{col}_bb_width"))
            new_features.append(bb_pos.rename(f"{col}_bb_pos"))

        # 한 번에 병합
        if new_features:
            X = pd.concat([X] + new_features, axis=1)

        # date_id 제거
        X = X.drop(columns=['date_id'])
        
        return X

    def _add_market_regime_features(self, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        시장 국면 피처 추가 (Volatility Regime, Trend Regime).
        Past Market Returns를 사용하여 Data Leakage 방지.
        """
        # Market Returns 준비
        market_df = df[['date_id']].drop_duplicates().sort_values('date_id').reset_index(drop=True)
        
        if 'lagged_market_forward_excess_returns' in df.columns:
            # Test set or Pre-calculated
            # date_id별로 첫 번째 값을 가져옴 (unique하다고 가정)
            temp = df[['date_id', 'lagged_market_forward_excess_returns']].drop_duplicates(subset=['date_id'])
            market_df = market_df.merge(temp, on='date_id', how='left')
            market_returns = market_df['lagged_market_forward_excess_returns']
        elif 'market_forward_excess_returns' in df.columns:
            # Train set: Shift 1 to create lagged returns (Past information at time t)
            temp = df[['date_id', 'market_forward_excess_returns']].drop_duplicates(subset=['date_id'])
            market_df = market_df.merge(temp, on='date_id', how='left')
            market_returns = market_df['market_forward_excess_returns'].shift(1)
        else:
            # Market info not available
            return X

        # 1. Volatility Regime
        # 20일, 60일 변동성
        vol_20d = market_returns.rolling(window=20).std()
        vol_60d = market_returns.rolling(window=60).std()
        
        market_df['market_vol_20d'] = vol_20d
        market_df['market_vol_60d'] = vol_60d
        
        # NaN Preserving Comparison
        # 둘 중 하나라도 NaN이면 결과도 NaN
        mask_vol = vol_20d.notna() & vol_60d.notna()
        market_df['market_regime_vol'] = np.nan
        market_df.loc[mask_vol, 'market_regime_vol'] = (vol_20d[mask_vol] > vol_60d[mask_vol]).astype(float)
        
        # 2. Trend Regime
        # 20일, 60일 이동평균
        ma_20d = market_returns.rolling(window=20).mean()
        ma_60d = market_returns.rolling(window=60).mean()
        
        market_df['market_ma_20d'] = ma_20d
        market_df['market_ma_60d'] = ma_60d
        
        # NaN Preserving Comparison
        mask_trend = ma_20d.notna() & ma_60d.notna()
        market_df['market_regime_trend'] = np.nan
        market_df.loc[mask_trend, 'market_regime_trend'] = (ma_20d[mask_trend] > ma_60d[mask_trend]).astype(float)

        # Merge back to X
        # X에는 이미 date_id가 없을 수 있으므로 (transform에서 drop됨), 
        # df['date_id']를 사용하여 병합해야 함.
        # 하지만 transform 구조상 X는 df와 row 순서가 같음 (sort_values를 안 했다면).
        # _add_time_series_features 등에서 sort_values('date_id')를 했으므로 X의 순서가 변경되었을 수 있음.
        # 따라서 X에 date_id를 다시 붙여서 병합하고 drop하는 것이 안전함.
        
        X = X.copy() # Defragment before adding column
        X['date_id'] = df['date_id'] # 원본 df의 date_id 사용 (순서가 맞아야 함)
        # 주의: 이전 단계에서 X를 sort_values했다면 인덱스가 섞였을 수 있음.
        # 하지만 transform 메서드 내에서 X = df[cols].copy()로 시작하고,
        # _add_time_series_features에서 sort_values를 하고 리턴함.
        # 따라서 현재 X는 date_id로 정렬된 상태일 가능성이 높음.
        # 안전을 위해 merge 사용.
        
        features_to_add = ['market_vol_20d', 'market_vol_60d', 'market_regime_vol', 
                           'market_ma_20d', 'market_ma_60d', 'market_regime_trend']
        
        X = X.merge(market_df[['date_id'] + features_to_add], on='date_id', how='left')
        
        # date_id 제거
        X = X.drop(columns=['date_id'])
        
        return X

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) 계산"""
        delta = series.diff()
        
        # NaN safe comparison
        # delta > 0 raises warning if NaN. 
        # We want: if NaN -> NaN (or 0 for calculation), but avoid warning.
        # RSI logic:
        # Gain = average gain (positive delta)
        # Loss = average loss (negative delta)
        
        # Create masks
        is_nan = delta.isna()
        
        # Gain: delta > 0 (NaN treated as False/0 for calculation, but handled safely)
        # np.where(condition, x, y) evaluates x and y? No, but here we construct series.
        
        # Safe positive check
        # fillna(0) for comparison only (doesn't affect original delta)
        # But we need to preserve NaNs in the rolling result if input was NaN?
        # Standard RSI usually ignores initial NaNs.
        
        # Method: Replace NaN with 0 for gain/loss calculation (standard practice for diff)
        # But to avoid warning:
        delta_filled = delta.fillna(0)
        
        gain = (delta.where(delta_filled > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta_filled < 0, 0)).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Moving Average Convergence Divergence (MACD) 계산"""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: int = 2):
        """Bollinger Bands 계산"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        간단한 상호작용 피처 추가.
        """
        # TODO: 필요시 구현 (Feature Selection 후 적용 권장)
        return X
    
    def get_feature_names(self) -> List[str]:
        """피처 이름 반환"""
        return self.feature_cols if self.feature_cols else []

