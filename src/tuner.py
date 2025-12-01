"""
Hyperparameter Tuning Module using Optuna.
"""

import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Tuple
from src.cv_strategy import get_cv_strategy
from src.metric import CompetitionMetric
from src.allocation import smart_allocation
from src.utils import get_logger

logger = get_logger(name="tuner", level="INFO")

class HyperparameterTuner:
    """
    Hyperparameter Tuner using Optuna.
    Optimizes for the competition metric (Adjusted Sharpe).
    """
    
    def __init__(self, n_trials: int = 50, cv_splits: int = 5):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.best_params = None
        self.study = None
        
    def optimize(self, X: pd.DataFrame, y: pd.Series, df_context: pd.DataFrame, model_type: str = 'return') -> Dict[str, Any]:
        """
        Run optimization.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target (for risk model, this should be abs(returns))
        df_context : pd.DataFrame
            DataFrame containing 'forward_returns', 'risk_free_rate' for metric calculation.
            Must be aligned with X.
        model_type : str
            'return' or 'risk'
            
        Returns
        -------
        Dict
            Best parameters
        """
        logger.info(f"Starting {model_type} model optimization with {self.n_trials} trials...")
        
        def objective(trial):
            # Hyperparameter search space
            params = {
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1
            }
            
            if model_type == 'return':
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            elif model_type == 'risk':
                params['objective'] = 'quantile'
                params['metric'] = 'quantile'
                params['alpha'] = 0.75
            elif model_type == 'risk2':
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            
            # CV Strategy
            cv = get_cv_strategy(
                'purged_walkforward',
                n_splits=self.cv_splits,
                train_size=2000,
                test_size=500,
                purge_gap=5
            )
            
            # For risk model, we might want to optimize for Quantile Loss directly
            # For return model, we optimize for IC or Sharpe (via smart_allocation)
            
            scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                # Train
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                # Predict
                pred = model.predict(X_val)
                
                if model_type == 'return':
                    # Optimize for IC (Information Coefficient)
                    # Simple correlation between pred and actual
                    score = np.corrcoef(pred, y_val)[0, 1]
                    if np.isnan(score): score = 0
                elif model_type == 'risk':
                    # Optimize for Quantile Loss (minimize)
                    # Pinball loss for quantile 0.75
                    alpha = 0.75
                    error = y_val - pred
                    loss = np.maximum(alpha * error, (alpha - 1) * error)
                    score = -np.mean(loss) # Maximize negative loss
                elif model_type == 'risk2':
                    # Optimize for RMSE (minimize)
                    rmse = np.sqrt(np.mean((pred - y_val)**2))
                    score = -rmse # Maximize negative RMSE
                
                scores.append(score)
            
            return np.mean(scores)

        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        # Add fixed params back
        if model_type == 'return':
            self.best_params.update({'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbosity': -1, 'random_state': 42})
        elif model_type == 'risk':
            self.best_params.update({'objective': 'quantile', 'metric': 'quantile', 'alpha': 0.75, 'boosting_type': 'gbdt', 'verbosity': -1, 'random_state': 42})
        elif model_type == 'risk2':
             self.best_params.update({'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbosity': -1, 'random_state': 42})
            
        logger.info(f"Best {model_type} Score: {self.study.best_value:.6f}")
        logger.info(f"Best {model_type} Params: {self.best_params}")
        
        return self.best_params
