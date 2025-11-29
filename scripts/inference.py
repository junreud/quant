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
            found = True
            break
    
    if not found:
        print("   ⚠️ Could not find 'src' package. Please ensure dataset is attached and path is correct.")
        raise


# Global variables (loaded once)
MODEL = None
PIPELINE = None
FEATURE_COLS = None
HISTORY_BUFFER = None


def load_model(model_path: str = "simple_model.pkl"):
    """
    Load model and pipeline (called once at startup).
    
    Parameters
    ----------
    model_path : str
        Path to model pickle file
    """
    global MODEL, PIPELINE, FEATURE_COLS
    
    if MODEL is not None:
        return  # Already loaded
    
    # Determine model path
    # 1. Try provided path (relative to CWD)
    if os.path.exists(model_path):
        pass
    # 2. Try relative to this script (e.g. ../models/simple_model.pkl)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming script is in scripts/ and model is in models/
        alt_path = os.path.join(script_dir, "..", "models", "simple_model.pkl")
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"   Found model at: {model_path}")
        else:
            # 3. Try Kaggle input structure
            kaggle_path = "/kaggle/input/quant-model-v1/models/simple_model.pkl"
            if os.path.exists(kaggle_path):
                model_path = kaggle_path
                print(f"   Found model at: {model_path}")

    print(f"📦 Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    MODEL = model_data # Dictionary containing 'model_return' and 'model_risk'
    PIPELINE = model_data['pipeline']  # 저장된 파이프라인 사용
    FEATURE_COLS = model_data['feature_cols']
    
    print(f"✅ Model loaded: {len(FEATURE_COLS)} features")
    print(f"✅ Pipeline loaded: {type(PIPELINE).__name__}")

    # Load History Buffer
    history_path = os.path.join(os.path.dirname(model_path), 'history.pkl')
    if os.path.exists(history_path):
        HISTORY_BUFFER = pd.read_pickle(history_path)
        print(f"✅ History buffer loaded: {len(HISTORY_BUFFER)} samples")
    else:
        print("⚠️ History buffer not found. Cold start (first few predictions may be inaccurate).")
        HISTORY_BUFFER = pd.DataFrame()


def predict(test, model_path: str = "simple_model.pkl"):
    """
    Predict allocation for test data.
    
    Uses a HISTORY_BUFFER to maintain past data for rolling feature calculation.
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
        # Ensure columns match (handling potential missing columns in test)
        # For simplicity, we assume test has same raw columns as history
        HISTORY_BUFFER = pd.concat([HISTORY_BUFFER, test_pd], axis=0)
        
    # Keep max size (e.g., 200) to prevent memory issues
    if len(HISTORY_BUFFER) > 200:
        HISTORY_BUFFER = HISTORY_BUFFER.iloc[-200:]
    
    # ========================================
    # Feature Engineering on Buffer
    # ========================================
    try:
        # Transform the WHOLE buffer
        # This calculates rolling features correctly using past data
        X_buffer = PIPELINE.transform(HISTORY_BUFFER, return_target=False)
        
        # Take the LAST row (current test sample)
        X_test = X_buffer.iloc[[-1]]
        
    except Exception as e:
        print(f"⚠️ Pipeline transform failed: {e}")
        print(f"   Fallback to manual preprocessing...")
        
        # Fallback: Use raw features if possible (ignoring rolling)
        X_test = pd.DataFrame()
        for col in FEATURE_COLS:
            if col in test_pd.columns:
                X_test[col] = test_pd[col].iloc[-1]
            else:
                X_test[col] = 0.0
        
        # Fill NaN with 0
        X_test = X_test.fillna(0)
    
    # ========================================
    # 모델 추론 (Dual Model)
    # ========================================
    # 1. Return Prediction
    pred_return = MODEL['model_return'].predict(X_test)[0]
    
    # 2. Risk Prediction
    # Risk model might use different features, but we assume X_test has all features
    # Pipeline usually returns all generated features.
    # We need to select features if the model expects specific ones.
    # But LGBM is robust to extra columns if feature names match.
    # However, if we used feature selection, we should be careful.
    # In run_pipeline.py, we saved 'feature_cols' which are ALL generated features.
    # The models were trained on subsets (top_k).
    # LightGBM selects features by name, so passing a superset DataFrame is fine.
    
    pred_risk = MODEL['model_risk'].predict(X_test)[0]
    
    # ========================================
    # Allocation Logic (Risk-Adjusted)
    # ========================================
    # k * (Return / Risk)
    k = 0.5
    vol_floor = 0.005 # Minimum volatility to prevent division by zero
    
    risk_val = max(pred_risk, vol_floor)
    allocation = k * (pred_return / risk_val)
    
    # Clip to [0, 2]
    allocation = max(0.0, min(2.0, float(allocation)))
    
    return allocation


if __name__ == "__main__":
    # Test locally
    import polars as pl
    
    print("=" * 80)
    print("🧪 Local Testing")
    print("=" * 80)
    
    # Create dummy test data
    test = pl.DataFrame({
        'D1': [0.5],
        'D2': [0.3],
        'E1': [0.1]
    })
    
    try:
        result = predict(test)
        print(f"\n✅ Test prediction: {result}")
        print(f"   (Expected: 0.5 or 1.5)")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
