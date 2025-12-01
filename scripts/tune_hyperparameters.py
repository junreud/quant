"""
Hyperparameter Tuning Script.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml
from src.pipeline import create_pipeline
from src.tuner import HyperparameterTuner
from src.utils import get_logger, load_config

logger = get_logger(name="tuning_script", level="INFO")

def main():
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("🔧 HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    
    # 1. Load Data
    train_path = project_root / config['data']['train']
    logger.info(f"Loading from: {train_path}")
    
    pipeline = create_pipeline(
        fillna_strategy=config['features']['fill_missing_strategy'],
        use_time_series_features=config['features']['use_time_series_features'],
        use_advanced_features=config['features']['use_advanced_features'],
        use_market_regime_features=config['features'].get('use_market_regime_features', True),
        use_feature_selection=config['features']['feature_selection']['enabled'],
        feature_selection_method=config['features']['feature_selection']['return_model']['method'],
        top_k_features=config['features']['feature_selection']['return_model']['top_k']
    )
    
    df = pd.read_csv(train_path)
    X, y = pipeline.fit_transform(df)
    
    # Restore context columns for metric calculation (forward_returns, risk_free_rate)
    # These are needed for the custom metric in tuner
    # X index should match df index if no rows were dropped.
    # Pipeline might drop rows? No, it fills NaNs.
    # But let's be safe and use index alignment.
    
    # We need a context DataFrame aligned with X
    # X has same index as df?
    # pipeline.transform returns X.
    
    # Check if X and df are aligned
    if len(X) != len(df):
        logger.warning("X and df length mismatch. Aligning context...")
        df_context = df.loc[X.index].copy()
    else:
        df_context = df.copy()
        
    # Remove holdout set (last 180 days)
    holdout_size = config['cv'].get('final_holdout', 180)
    purge_gap = config['cv'].get('purge_gap', 5)
    
    X_tune = X.iloc[:-(holdout_size + purge_gap)]
    y_tune = y.iloc[:-(holdout_size + purge_gap)]
    df_context_tune = df_context.iloc[:-(holdout_size + purge_gap)]
    
    logger.info(f"Tuning data: {len(X_tune)} samples")
    
    # 2. Run Tuner
    tuner = HyperparameterTuner(n_trials=50, cv_splits=5)
    
    # 2.1 Tune Return Model
    logger.info("🔧 Tuning Return Model...")
    best_return_params = tuner.optimize(X_tune, y_tune, df_context_tune, model_type='return')
    
    # 2.2 Tune Risk Model
    logger.info("🔧 Tuning Risk Model...")
    # Risk target is abs(returns)
    y_risk = y_tune.abs()
    
    # Use risk features if selected (optional, but good for consistency)
    # For simplicity, we tune on all features or top K features.
    # Ideally we should use the same feature selection as in pipeline.
    # But here we use X_tune which already has features.
    best_risk_params = tuner.optimize(X_tune, y_risk, df_context_tune, model_type='risk')
    
    # 3. Save Results
    # Combine into config structure
    final_params = {
        'lgbm_return': best_return_params,
        'lgbm_risk': best_risk_params
    }
    
    output_path = project_root / "conf" / "best_params.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(final_params, f)
        
    logger.info(f"✅ Best params saved to: {output_path}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
