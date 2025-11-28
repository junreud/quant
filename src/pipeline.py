"""
통합 데이터 파이프라인

전처리 + 피처 엔지니어링을 한 번에!

사용법:
    from src.pipeline import create_pipeline
    
    # 파이프라인 생성
    pipeline = create_pipeline(fillna_strategy='median')
    
    # 학습 데이터
    X_train, y_train = pipeline.fit_transform(train_df)
    
    # 테스트 데이터
    X_test = pipeline.transform(test_df)
"""

import pandas as pd
from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer
from src.feature_selection import FeatureSelector


class Pipeline:
    """전처리 + 피처 엔지니어링 + 피처 선택 통합 파이프라인"""
    
    def __init__(
        self,
        fillna_strategy: str = 'median',
        add_interactions: bool = False,
        use_time_series_features: bool = True,
        use_advanced_features: bool = True,
        use_market_regime_features: bool = True,
        use_feature_selection: bool = False,
        feature_selection_method: str = 'importance',
        top_k_features: int = 50
    ):
        """
        Parameters
        ----------
        fillna_strategy : str
            결측치 처리 방법
        add_interactions : bool
            상호작용 피처 추가 여부
        use_time_series_features : bool
            시계열 피처 추가 여부
        use_advanced_features : bool
            고급 시계열 피처 추가 여부
        use_market_regime_features : bool
            시장 국면 피처 추가 여부
        use_feature_selection : bool
            피처 선택 사용 여부
        feature_selection_method : str
            피처 선택 방법 ('importance', 'collinear')
        top_k_features : int
            선택할 피처 개수 (importance 방식일 때)
        """
        self.preprocessor = DataPreprocessor(fillna_strategy)
        self.feature_engineer = FeatureEngineer(
            add_interactions=add_interactions,
            use_time_series_features=use_time_series_features,
            use_advanced_features=use_advanced_features,
            use_market_regime_features=use_market_regime_features,
        )
        self.use_feature_selection = use_feature_selection
        self.feature_selector = FeatureSelector() if use_feature_selection else None
        self.feature_selection_method = feature_selection_method
        self.top_k_features = top_k_features
        self.selected_features = None
        
    def fit(self, df: pd.DataFrame) -> 'Pipeline':
        """파이프라인 학습"""
        # 1. 전처리 학습
        # 1. 전처리 학습 & 적용
        # fit_transform은 기본적으로 (X, y) 튜플을 반환하므로, 
        # X만 필요하면 return_target=False를 지정해야 함.
        X_pre = self.preprocessor.fit_transform(df, return_target=False)
        
        # FE를 위해 date_id 및 market info 복원
        cols_to_restore = ['date_id', 'market_forward_excess_returns', 'lagged_market_forward_excess_returns']
        for col in cols_to_restore:
            if col in df.columns:
                X_pre[col] = df[col]
            
        # 3. FE 학습
        self.feature_engineer.fit(X_pre)
        
        # 4. 피처 선택 (옵션)
        if self.use_feature_selection:
            # FE 적용하여 모든 후보 피처 생성
            X_fe = self.feature_engineer.transform(X_pre)
            
            if self.feature_selection_method == 'collinear':
                self.selected_features = self.feature_selector.remove_collinear(X_fe)
            elif self.feature_selection_method == 'importance':
                if 'forward_returns' not in df.columns:
                    print("⚠️ Warning: Target not found for feature selection. Skipping.")
                    self.selected_features = X_fe.columns.tolist()
                else:
                    y = df['forward_returns']
                    # NaN 제거 (타겟 기준)
                    valid_mask = ~y.isnull()
                    self.selected_features = self.feature_selector.select_by_importance(
                        X_fe[valid_mask], y[valid_mask], top_k=self.top_k_features
                    )
        
        return self
    
    def transform(self, df: pd.DataFrame, return_target: bool = False) -> pd.DataFrame:
        """파이프라인 적용"""
        # 1. 전처리
        if return_target:
            X, y = self.preprocessor.transform(df, return_target=True)
        else:
            X = self.preprocessor.transform(df, return_target=False)
            
        # FE를 위해 date_id 및 market info 복원 (전처리된 X에 date_id가 없을 수 있음)
        cols_to_restore = ['date_id', 'market_forward_excess_returns', 'lagged_market_forward_excess_returns']
        for col in cols_to_restore:
            if col in df.columns:
                X[col] = df[col]
        
        # 2. 피처 엔지니어링
        X = self.feature_engineer.transform(X)
        
        # 3. 피처 선택
        if self.use_feature_selection and self.selected_features:
            # 선택된 피처만 유지 (존재하는 것만)
            cols_to_use = [c for c in self.selected_features if c in X.columns]
            X = X[cols_to_use]
        
        if return_target:
            return X, y
        return X
    
    def fit_transform(self, df: pd.DataFrame, return_target: bool = True) -> pd.DataFrame:
        """fit + transform"""
        self.fit(df)
        return self.transform(df, return_target=return_target)
    
    def get_feature_names(self):
        """피처 이름 반환"""
        if self.use_feature_selection and self.selected_features:
            return self.selected_features
        return self.feature_engineer.get_feature_names()




def create_pipeline(**kwargs):
    """파이프라인 생성 헬퍼 함수"""
    return Pipeline(**kwargs)
