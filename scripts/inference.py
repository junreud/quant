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

# Add current directory to path so src can be imported if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    MODEL = model_data['model']
    PIPELINE = model_data['pipeline']  # 저장된 파이프라인 사용
    FEATURE_COLS = model_data['feature_cols']
    
    print(f"✅ Model loaded: {len(FEATURE_COLS)} features")
    print(f"✅ Pipeline loaded: {type(PIPELINE).__name__}")


def predict(test, model_path: str = "simple_model.pkl"):
    """
    Predict allocation for test data.
    
    이 함수는 run_pipeline.py와 동일한 전처리 과정을 수행합니다:
    1. Pipeline을 통한 전처리 (fillna 등)
    2. 모델 추론
    3. Allocation 변환
    
    Parameters
    ----------
    test : polars.DataFrame
        Test data (Polars DataFrame)
    model_path : str
        Path to model file (relative to working directory)
        
    Returns
    -------
    float
        Allocation value (0.0 ~ 2.0)
    """
    # Load model if not loaded
    load_model(model_path)
    
    # Convert to pandas
    import polars as pl
    if isinstance(test, pl.DataFrame):
        test_pd = test.to_pandas()
    else:
        test_pd = test
    
    # ========================================
    # run_pipeline.py와 동일한 전처리 과정
    # ========================================
    # Pipeline을 통해 전처리 (fillna, feature engineering)
    try:
        X_test = PIPELINE.transform(test_pd, return_target=False)
    except Exception as e:
        print(f"⚠️ Pipeline transform failed: {e}")
        print(f"   Fallback to manual preprocessing...")
        
        # Fallback: 수동 전처리
        X_test = pd.DataFrame()
        for col in FEATURE_COLS:
            if col in test_pd.columns:
                X_test[col] = test_pd[col]
            else:
                X_test[col] = 0.0
        
        # Fill NaN with 0
        X_test = X_test.fillna(0)
    
    # ========================================
    # 모델 추론
    # ========================================
    y_pred = MODEL.predict(X_test)[0]
    
    # ========================================
    # Allocation 변환 (run_pipeline.py의 smart_allocation과 동일)
    # ========================================
    # Simple strategy: > 0 → 1.5, <= 0 → 0.5
    if y_pred > 0:
        allocation = 1.5
    else:
        allocation = 0.5
    
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
