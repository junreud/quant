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
HISTORY_BUFFER = None
PRED_BUFFER = None


def load_model(model_path: str = "dual_model.pkl"):
    """
    Load model and pipeline (called once at startup).
    
    Parameters
    ----------
    model_path : str
        Path to model pickle file
    """
    global MODEL, PIPELINE, FEATURE_COLS
    print("DEBUG: load_model called")
    
    if MODEL is not None:
        print("DEBUG: MODEL already loaded")
        return  # Already loaded
    
    # Determine model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of paths to check
    paths_to_check = [
        model_path, # As provided
        os.path.join(script_dir, "..", "models", "dual_model.pkl"), # Local dev
        os.path.join(script_dir, "dual_model.pkl"), # Same dir
        "/kaggle/input/quant-model-v1/models/dual_model.pkl", # Kaggle
        "/kaggle/input/quant-model-v1/dual_model.pkl",
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
    
    MODEL = model_data # Dictionary containing 'model_return' and 'model_risk'
    print(f"   Model type: {type(MODEL)}")
    if isinstance(MODEL, dict):
        print(f"   Model keys: {list(MODEL.keys())}")
    else:
        print(f"   Model content: {MODEL}")
    
    # Check if we loaded the wrapper dict or just the model dict
    if 'model_return' not in MODEL and 'model' in MODEL:
         # Maybe it's nested?
         print("   ⚠️ 'model_return' not found, checking 'model' key...")
         if isinstance(MODEL['model'], dict):
             print(f"   Inner keys: {list(MODEL['model'].keys())}")
             MODEL = MODEL['model'] # Unwrap if needed
    
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


def predict(test, model_path: str = "dual_model.pkl"):
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
        # Construct dictionary first to ensure correct DataFrame shape
        x_test_dict = {}
        for col in FEATURE_COLS:
            if col in test_pd.columns:
                # Take the last value
                val = test_pd[col].iloc[-1]
                x_test_dict[col] = val
            else:
                x_test_dict[col] = 0.0
        
        # Create 1-row DataFrame
        X_test = pd.DataFrame([x_test_dict])
        
        # Fill NaN with 0
        X_test = X_test.fillna(0)
    
    # ========================================
    # 모델 추론 (Dual Model)
    # ========================================
    # 1. Return Prediction
    if isinstance(MODEL, dict):
        # Debug print removed for production
        pass
    
    if MODEL is None:
        raise ValueError("MODEL is None in predict()")
        
    pred_return = MODEL['model_return'].predict(X_test)[0]
    
    # 2. Risk Prediction
    pred_risk = MODEL['model_risk'].predict(X_test)[0]
    
    # ========================================
    # Dynamic Allocation (Relative Strength)
    # ========================================
    global PRED_BUFFER
    
    # Update Prediction Buffer
    if PRED_BUFFER is None:
        PRED_BUFFER = []
        
    PRED_BUFFER.append(pred_return)
    
    # Keep max size (e.g., 200)
    if len(PRED_BUFFER) > 200:
        PRED_BUFFER.pop(0)
        
    # Calculate Rolling Stats
    if len(PRED_BUFFER) < 2:
        roll_mean = pred_return # Not enough data
        roll_std = 1.0
    else:
        # Use last N (e.g., 20)
        window = 20
        recent_preds = PRED_BUFFER[-window:]
        roll_mean = np.mean(recent_preds)
        roll_std = np.std(recent_preds)
        
    # Dynamic Allocation Logic (Using src.allocation)
    # This ensures consistency with the optimized strategy
    allocation = dynamic_risk_allocation(
        return_pred=pred_return,
        risk_pred=pred_risk,
        rolling_mean=roll_mean,
        rolling_std=roll_std,
        k=0.7,  # Optimized parameter
        max_position=2.0
    )
    
    # Ensure float
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
