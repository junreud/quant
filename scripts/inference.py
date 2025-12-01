"""
Inference module for Kaggle submission.

This module is uploaded together with the model to Kaggle Dataset.
The notebook simply imports predict() from this file.

캐글 서버에서 run_pipeline.py와 동일한 전처리를 수행합니다.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory and parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 전처리 모듈 import (src 패키지 사용)
try:
    from src.preprocessing import DataPreprocessor
    from src.features import FeatureEngineer
    from src.pipeline import Pipeline
    from src.allocation import dynamic_risk_allocation
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Script location: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"   sys.path: {sys.path}")
    
    # Try to find src in likely locations
    possible_paths = [
        '/kaggle/input/quant-model-v1',
        '/kaggle/input/quant-model-v1/src',
        '../input/quant-model-v1',
    ]
    found = False
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'src')):
            print(f"   Found 'src' in: {path}. Adding to sys.path...")
            sys.path.insert(0, path)
            from src.preprocessing import DataPreprocessor
            from src.features import FeatureEngineer
            from src.pipeline import Pipeline
            from src.allocation import dynamic_risk_allocation
            found = True
            break
    
    if not found:
        print("   ⚠️ Could not find 'src' package. Please ensure dataset is attached and path is correct.")
        raise


# Global variables (loaded once)
MODEL = None
PIPELINE = None
FEATURE_COLS = None
RETURN_FEATURES = None
RISK_FEATURES = None
RISK2_FEATURES = None
HISTORY_BUFFER = None
PRED_BUFFER = None
CONFIG = None


def load_model(model_path: str = "triple_model.pkl"):
    """
    Load model and pipeline (called once at startup).
    """
    global MODEL, PIPELINE, FEATURE_COLS, RETURN_FEATURES, RISK_FEATURES, RISK2_FEATURES, CONFIG
    print("DEBUG: load_model called")
    
    if MODEL is not None:
        print("DEBUG: MODEL already loaded")
        return  # Already loaded
    
    # Determine model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of paths to check
    paths_to_check = [
        model_path, # As provided
        os.path.join(script_dir, "..", "models", "triple_model.pkl"), # Local dev
        os.path.join(script_dir, "triple_model.pkl"), # Same dir
        "/kaggle/input/quant-model-v1/models/triple_model.pkl", # Kaggle
        "/kaggle/input/quant-model-v1/triple_model.pkl",
    ]
    
    found_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path:
        model_path = found_path
        print(f"   Found model at: {model_path}")
    else:
        print(f"⚠️ Model file not found. Checked: {paths_to_check}")

    print(f"📦 Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    MODEL = model_data # Dictionary containing models
    PIPELINE = model_data['pipeline']
    FEATURE_COLS = model_data['feature_cols'] # All generated features
    
    # Load selected features for each model
    RETURN_FEATURES = model_data.get('return_features', FEATURE_COLS)
    RISK_FEATURES = model_data.get('risk_features', FEATURE_COLS)
    RISK2_FEATURES = model_data.get('risk2_features', FEATURE_COLS)
    CONFIG = model_data.get('config', {})
    
    print(f"✅ Model loaded")
    print(f"   Return Features: {len(RETURN_FEATURES)}")
    print(f"   Risk Features: {len(RISK_FEATURES)}")
    print(f"   Risk2 Features: {len(RISK2_FEATURES)}")
    print(f"✅ Pipeline loaded: {type(PIPELINE).__name__}")

    # Load History Buffer
    history_path = os.path.join(os.path.dirname(model_path), 'history.pkl')
    global HISTORY_BUFFER
    if os.path.exists(history_path):
        HISTORY_BUFFER = pd.read_pickle(history_path)
        print(f"✅ History buffer loaded: {len(HISTORY_BUFFER)} samples")
    else:
        print("⚠️ History buffer not found. Cold start.")
        HISTORY_BUFFER = pd.DataFrame()
        
    # Load Prediction History Buffer
    global PRED_BUFFER
    pred_history_path = os.path.join(os.path.dirname(model_path), 'pred_history.pkl')
    if os.path.exists(pred_history_path):
        with open(pred_history_path, 'rb') as f:
            PRED_BUFFER = pickle.load(f)
        print(f"✅ Prediction buffer loaded: {len(PRED_BUFFER)} samples")
    else:
        print("⚠️ Prediction buffer not found. Cold start.")
        PRED_BUFFER = []


def predict(test, model_path: str = "triple_model.pkl"):
    """
    Predict allocation for test data.
    """
    # Load model if not loaded
    load_model(model_path)
    
    # Convert to pandas
    import polars as pl
    if isinstance(test, pl.DataFrame):
        test_pd = test.to_pandas()
    else:
        test_pd = test
        
    # Update History Buffer
    global HISTORY_BUFFER
    
    # Append new data
    if HISTORY_BUFFER is None or HISTORY_BUFFER.empty:
        HISTORY_BUFFER = test_pd
    else:
        HISTORY_BUFFER = pd.concat([HISTORY_BUFFER, test_pd], axis=0)
        
    # Reset index to avoid duplicates (Critical for pipeline transform)
    HISTORY_BUFFER = HISTORY_BUFFER.reset_index(drop=True)
        
    # Keep max size (e.g., 200)
    if len(HISTORY_BUFFER) > 200:
        HISTORY_BUFFER = HISTORY_BUFFER.iloc[-200:]
    
    # ========================================
    # Feature Engineering on Buffer
    # ========================================
    try:
        # Transform the WHOLE buffer
        X_buffer = PIPELINE.transform(HISTORY_BUFFER, return_target=False)
        
        # Take the LAST row (current test sample)
        X_test = X_buffer.iloc[[-1]]
        
    except Exception as e:
        print(f"⚠️ Pipeline transform failed: {e}")
        print(f"   Fallback to manual preprocessing...")
        
        # Fallback: Use raw features if possible
        x_test_dict = {}
        for col in FEATURE_COLS:
            if col in test_pd.columns:
                val = test_pd[col].iloc[-1]
                x_test_dict[col] = val
            else:
                x_test_dict[col] = 0.0
        
        X_test = pd.DataFrame([x_test_dict])
        X_test = X_test.fillna(0)
    
    # ========================================
    # 모델 추론 (Triple Model)
    # ========================================
    if MODEL is None:
        raise ValueError("MODEL is None in predict()")
        
    # 1. Return Prediction
    # Use specific features
    pred_return = MODEL['model_return'].predict(X_test[RETURN_FEATURES])[0]
    
    # 2. Risk Prediction
    pred_risk = MODEL['model_risk'].predict(X_test[RISK_FEATURES])[0]
    
    # 3. Risk Model 2 Prediction (Market Regime)
    pred_risk2 = MODEL['model_risk2'].predict(X_test[RISK2_FEATURES])[0]
    
    # ========================================
    # Triple Model Allocation
    # ========================================
    
    # Calculate Trend Signal (Momentum Guard)
    # Use HISTORY_BUFFER to calculate MA20
    trend_signal = 1.0 # Default Bullish
    
    if HISTORY_BUFFER is not None and not HISTORY_BUFFER.empty:
        # Check if 'close' or 'Close' exists
        close_col = None
        for col in ['close', 'Close', 'adj_close', 'Adj Close']:
            if col in HISTORY_BUFFER.columns:
                close_col = col
                break
        
        if close_col:
            # Need at least 20 days
            if len(HISTORY_BUFFER) >= 20:
                # Calculate MA20
                ma20 = HISTORY_BUFFER[close_col].rolling(window=20).mean().iloc[-1]
                current_price = HISTORY_BUFFER[close_col].iloc[-1]
                
                # Signal: 1.0 if Price >= MA20, 0.0 if Price < MA20
                if current_price < ma20:
                    trend_signal = 0.0 # Bearish
                    print(f"📉 Momentum Guard Triggered: Price({current_price:.2f}) < MA20({ma20:.2f})")
                else:
                    print(f"📈 Momentum Guard: Bullish (Price {current_price:.2f} >= MA20 {ma20:.2f})")
            else:
                print(f"⚠️ Not enough history for MA20 ({len(HISTORY_BUFFER)} < 20)")
        else:
            print("⚠️ Close price column not found in history for Momentum Guard.")
    
    from src.allocation import triple_model_allocation
    
    # Use config values if available, else defaults
    k = CONFIG['allocation']['k'] if CONFIG else 3.5
    threshold = CONFIG['allocation']['threshold'] if CONFIG else -0.012
    
    allocation = triple_model_allocation(
        return_pred=pred_return,
        risk_vol_pred=pred_risk,
        risk_market_pred=pred_risk2,
        market_threshold=threshold,
        k=k,
        trend_signal=trend_signal
    )
    
    return float(allocation)


if __name__ == "__main__":
    # Test locally
    import polars as pl
    
    print("=" * 80)
    print("🧪 Local Testing")
    print("=" * 80)
    
    # Load model to get feature cols
    try:
        load_model()
        if FEATURE_COLS:
            # Create dummy data with correct columns
            dummy_data = {col: [0.0] for col in FEATURE_COLS}
            # Add some raw columns if needed by pipeline (e.g. 'open', 'close'...) 
            # But pipeline transform might need specific raw columns.
            # For inference test, we can just bypass pipeline transform if we provide features directly?
            # No, predict() calls pipeline.transform.
            
            # Let's just create a dummy dataframe with some common raw columns
            # and hope pipeline handles it or we hit the fallback.
            # Actually, to hit fallback, we can force an error or just rely on the fact that 
            # pipeline.transform will likely fail with missing raw columns.
            
            test = pl.DataFrame(dummy_data)
            # Add raw cols just in case
            test = test.with_columns([
                pl.lit(100.0).alias('open'),
                pl.lit(101.0).alias('close'),
                pl.lit(1000).alias('volume'),
            ])
        else:
             test = pl.DataFrame({'dummy': [1.0]})
    except Exception as e:
        print(f"⚠️ Failed to load model for testing: {e}")
        test = pl.DataFrame({'dummy': [1.0]})
    
    try:
        result = predict(test)
        print(f"\n✅ Test prediction: {result}")
        print(f"   (Expected: 0.5 or 1.5)")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
