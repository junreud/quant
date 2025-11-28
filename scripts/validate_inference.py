"""
Local validation script using inference.py

이 스크립트는:
1. kaggle_upload/inference.py의 predict() 사용
2. Validation 데이터로 예측
3. CompetitionMetric으로 최종 점수 계산

Kaggle 제출 전에 로컬에서 점수를 미리 확인!
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "kaggle_upload"))  # inference.py 경로 추가

import pandas as pd
import numpy as np
import polars as pl
from src.metric import CompetitionMetric
from src.utils import get_logger, load_config
from inference import predict  # kaggle_upload/inference.py에서 import

logger = get_logger(name="validate_inference", level="INFO")


def validate_inference(config: dict):
    """
    inference.py의 predict()로 validation 점수 계산.
    """
    logger.info("=" * 80)
    logger.info("🔍 Validating Inference with Competition Metric")
    logger.info("=" * 80)
    
    # Load validation data (last 180)
    train_path = project_root / config['data']['train']
    df = pd.read_csv(train_path)
    
    holdout_size = 180
    val_df = df.iloc[-holdout_size:]
    
    logger.info(f"\n📊 Validation data: {len(val_df)} samples (last {holdout_size})")
    
    # Predict using inference.py
    logger.info(f"\n🤖 Running predictions with inference.py...")
    
    allocations = []
    for idx, row in val_df.iterrows():
        # Convert row to Polars DataFrame (same as Kaggle)
        test_pl = pl.DataFrame([row.to_dict()])
        
        # Call predict() from inference.py with full model path
        model_path = str(project_root / "kaggle_upload" / "simple_model.pkl")
        allocation = predict(test_pl, model_path=model_path)
        allocations.append(allocation)
    
    allocations = np.array(allocations)
    
    logger.info(f"✅ Predictions complete")
    logger.info(f"   Min allocation: {allocations.min():.4f}")
    logger.info(f"   Max allocation: {allocations.max():.4f}")
    logger.info(f"   Mean allocation: {allocations.mean():.4f}")
    
    # Calculate competition score
    logger.info(f"\n📈 Calculating competition score...")
    
    metric_calculator = CompetitionMetric(
        vol_threshold=config['metric']['vol_threshold'],
        use_return_penalty=config['metric']['use_return_penalty'],
        min_periods=config['metric']['min_periods']
    )
    
    results = metric_calculator.calculate_score(
        allocations=allocations,
        forward_returns=val_df['forward_returns'].values,
        market_returns=val_df['forward_returns'].values,
        risk_free_rate=val_df['risk_free_rate'].values
    )
    
    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FINAL VALIDATION SCORE")
    logger.info(f"{'='*80}")
    logger.info(f"🎯 Adjusted Sharpe: {results['score']:.6f}")
    logger.info(f"📊 Sharpe Ratio: {results['sharpe']:.6f}")
    logger.info(f"⚠️  Vol Penalty: {results['vol_penalty']:.6f}")
    logger.info(f"⚠️  Return Penalty: {results['return_penalty']:.6f}")
    logger.info(f"")
    logger.info(f"📉 Strategy Vol: {results['strategy_vol']:.2f}%")
    logger.info(f"📉 Market Vol: {results['market_vol']:.2f}%")
    logger.info(f"📊 Vol Ratio: {results['vol_ratio']:.4f}")
    logger.info(f"")
    logger.info(f"💰 Strategy Return: {results['strategy_mean_excess_return']:.6f}")
    logger.info(f"💰 Market Return: {results['market_mean_excess_return']:.6f}")
    logger.info(f"📊 Return Gap: {results['return_gap']:.4f}%")
    logger.info(f"{'='*80}")
    
    # Compare with baseline
    baseline_score = 0.465  # Full train baseline
    improvement = (results['score'] - baseline_score) / baseline_score * 100
    
    logger.info(f"\n📊 vs Baseline:")
    logger.info(f"   Baseline: {baseline_score:.6f}")
    logger.info(f"   Current:  {results['score']:.6f}")
    logger.info(f"   Change:   {improvement:+.2f}%")
    
    # Expected Public LB
    logger.info(f"\n🎯 Expected Public LB: ~{results['score']:.3f}")
    logger.info(f"   (based on Local CV = 0.465 ≈ Public LB = 0.461)")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Validation complete!")
    logger.info(f"{'='*80}")
    
    return results


def main():
    config = load_config()
    validate_inference(config)


if __name__ == "__main__":
    main()
