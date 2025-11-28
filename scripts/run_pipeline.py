"""
전체 파이프라인 실행 스크립트 (Purged Walk-Forward CV)

한 번의 실행으로:
1. 데이터 전처리
2. 모델 학습 (Purged Walk-Forward CV)
3. OOF 평가
4. 최종 테스트 (마지막 180일)
5. Kaggle 패키징

사용법:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --cv-splits 5
    python scripts/run_pipeline.py --skip-package
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import subprocess
from src.pipeline import create_pipeline
from src.allocation import smart_allocation
from src.metric import CompetitionMetric
from src.cv_strategy import get_cv_strategy
from src.experiment_tracker import ExperimentTracker
from src.utils import get_logger, load_config, ensure_dir

logger = get_logger(name="pipeline", level="INFO")


def run_full_pipeline(
    cv_splits: int = 5,
    skip_package: bool = False,
    cv_strategy: str = 'purged_walkforward'
):
    """
    전체 파이프라인 실행.
    
    Parameters
    ----------
    cv_splits : int
        CV split 개수
    skip_package : bool
        Kaggle 패키징 건너뛰기
    cv_strategy : str
        CV 전략 ('timeseries', 'purged_walkforward')
    """
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("🚀 FULL PIPELINE EXECUTION (Purged Walk-Forward)")
    logger.info("=" * 80)
    
    # ========================================
    # Step 1: 데이터 로드 및 전처리
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 Step 1: Data Loading & Preprocessing")
    logger.info("=" * 80)
    
    train_path = project_root / config['data']['train']
    logger.info(f"Loading from: {train_path}")
    
    # 전처리 Pipeline 사용
    pipeline = create_pipeline(
        # 전처리
        fillna_strategy=config['features']['fill_missing_strategy'],
        # Feature Engineering
        add_interactions=config['features']['add_interactions'],
        use_time_series_features=config['features']['use_time_series_features'],
        use_advanced_features=config['features']['use_advanced_features'],
        use_market_regime_features=config['features'].get('use_market_regime_features'),
        # Feature Selection
        use_feature_selection=config['features']['feature_selection']['enabled'],
        feature_selection_method=config['features']['feature_selection']['method'],
        top_k_features=config['features']['feature_selection']['top_k']
    )
    
    df = pd.read_csv(train_path)
    X, y = pipeline.fit_transform(df)
    feature_cols = pipeline.get_feature_names()
    
    logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 마지막 N일 분리 (최종 테스트용) - config에서 읽기
    holdout_size = config['cv'].get('final_holdout', 180)
    purge_gap = config['cv'].get('purge_gap', 5)
    
    # CV 데이터와 최종 테스트 사이에 purge_gap만큼 간격을 둠 (data leakage 방지)
    X_cv = X.iloc[:-(holdout_size + purge_gap)]  # CV용 (마지막 180+5=185개 제외)
    y_cv = y.iloc[:-(holdout_size + purge_gap)]
    df_cv = df.iloc[:-(holdout_size + purge_gap)]
    
    X_test = X.iloc[-holdout_size:]  # 최종 테스트용 (마지막 180개)
    y_test = y.iloc[-holdout_size:]
    df_test = df.iloc[-holdout_size:]
    
    logger.info(f"CV data: {len(X_cv)} samples (excluding last {holdout_size + purge_gap} days)")
    logger.info(f"Purge gap: {purge_gap} days between CV and final test")
    logger.info(f"Final test: {len(X_test)} samples (last {holdout_size} days)")
    
    # ========================================
    # Step 2: 모델 학습 (Purged Walk-Forward CV)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"🤖 Step 2: Model Training ({cv_strategy}, {cv_splits} splits)")
    logger.info("=" * 80)
    
    # Purged Walk-Forward CV
    if cv_strategy == 'purged_walkforward':
        cv = get_cv_strategy(
            'purged_walkforward',
            n_splits=cv_splits,
            train_size=config['cv'].get('train_size', 2000),
            test_size=config['cv'].get('test_size', 500),
            purge_gap=config['cv'].get('purge_gap', 5)
        )
        logger.info("Using Purged Walk-Forward Split:")
        logger.info(f"  • Train size: {config['cv'].get('train_size', 2000)}")
        logger.info(f"  • Test size: {config['cv'].get('test_size', 500)}")
        logger.info(f"  • Purge gap: {config['cv'].get('purge_gap', 5)}")
    else:
        cv = get_cv_strategy('timeseries', n_splits=cv_splits)
        logger.info(f"Using TimeSeriesSplit (n_splits={cv_splits})")
    
    oof_predictions = np.zeros(len(X_cv))
    oof_mask = np.zeros(len(X_cv), dtype=bool)
    models = []
    
    lgbm_params = config['lgbm'].copy()
    
    # Metric calculator
    metric_calculator = CompetitionMetric(
        vol_threshold=config['metric']['vol_threshold'],
        use_return_penalty=config['metric']['use_return_penalty'],
        min_periods=config['metric']['min_periods']
    )
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_cv)):
        logger.info(f"\n  Fold {fold_idx + 1}/{cv_splits} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        
        X_train_fold = X_cv.iloc[train_idx]
        y_train_fold = y_cv.iloc[train_idx]
        X_val_fold = X_cv.iloc[val_idx]
        y_val_fold = y_cv.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='rmse')
        
        # OOF predictions
        fold_pred = model.predict(X_val_fold)
        oof_predictions[val_idx] = fold_pred
        oof_mask[val_idx] = True
        
        # Fold score
        fold_allocations = smart_allocation(fold_pred, center=1.0, sensitivity=20)
        val_df_fold = df_cv.iloc[val_idx]
        
        fold_result = metric_calculator.calculate_score(
            allocations=fold_allocations,
            forward_returns=val_df_fold['forward_returns'].values,
            market_returns=val_df_fold['forward_returns'].values,
            risk_free_rate=val_df_fold['risk_free_rate'].values
        )
        
        fold_scores.append(fold_result['score'])
        models.append(model)
        
        logger.info(f"  ✅ Fold {fold_idx + 1} Train_idx: {train_idx[0]} ~ {train_idx[-1]} | Val_idx: {val_idx[0]} ~ {val_idx[-1]} | Score: {fold_result['score']:.6f}")
    
    logger.info(f"\n✅ CV Training complete | OOF samples: {oof_mask.sum()}/{len(X_cv)}")
    logger.info(f"📊 Fold Scores: {', '.join([f'{s:.4f}' for s in fold_scores])}")
    logger.info(f"📊 Mean ± Std: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    # ========================================
    # Step 3: OOF 평가
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("📈 Step 3: OOF Evaluation")
    logger.info("=" * 80)
    
    oof_df = df_cv[oof_mask]
    oof_pred = oof_predictions[oof_mask]
    
    # Allocation
    allocations = smart_allocation(oof_pred, center=1.0, sensitivity=20)
    
    results = metric_calculator.calculate_score(
        allocations=allocations,
        forward_returns=oof_df['forward_returns'].values,
        market_returns=oof_df['forward_returns'].values,
        risk_free_rate=oof_df['risk_free_rate'].values
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 OOF RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Adjusted Sharpe: {results['score']:.6f}")
    logger.info(f"📊 Sharpe Ratio: {results['sharpe']:.6f}")
    logger.info(f"⚠️  Vol Penalty: {results['vol_penalty']:.6f}")
    logger.info(f"⚠️  Return Penalty: {results['return_penalty']:.6f}")
    logger.info(f"📉 Strategy Vol: {results['strategy_vol']:.2f}%")
    logger.info(f"📉 Market Vol: {results['market_vol']:.2f}%")
    logger.info(f"{'='*80}")
    
    # ========================================
    # Step 4: 최종 모델 학습 (전체 CV 데이터)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🏁 Step 4: Final Model Training (CV data only)")
    logger.info("=" * 80)
    
    final_model = lgb.LGBMRegressor(**lgbm_params)
    final_model.fit(X_cv, y_cv)
    
    logger.info(f"✅ Final model trained on {len(X_cv)} samples")
    
    # ========================================
    # Step 5: 최종 테스트 (마지막 180일)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🧪 Step 5: Final Test Evaluation (Last 180 days)")
    logger.info("=" * 80)
    
    test_pred = final_model.predict(X_test)
    test_allocations = smart_allocation(test_pred, center=1.0, sensitivity=12)
    
    test_results = metric_calculator.calculate_score(
        allocations=test_allocations,
        forward_returns=df_test['forward_returns'].values,
        market_returns=df_test['forward_returns'].values,
        risk_free_rate=df_test['risk_free_rate'].values
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 FINAL TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Adjusted Sharpe: {test_results['score']:.6f}")
    logger.info(f"📊 Sharpe Ratio: {test_results['sharpe']:.6f}")
    logger.info(f"⚠️  Vol Penalty: {test_results['vol_penalty']:.6f}")
    logger.info(f"⚠️  Return Penalty: {test_results['return_penalty']:.6f}")
    logger.info(f"{'='*80}")
    
    logger.info(f"\n💡 OOF vs Final Test:")
    logger.info(f"   OOF:  {results['score']:.6f}")
    logger.info(f"   Test: {test_results['score']:.6f}")
    logger.info(f"   Diff: {(test_results['score'] - results['score']):.6f}")
    
    # 최종 모델 재학습 (전체 데이터)
    logger.info(f"\n🔄 Retraining on ALL data for submission...")
    final_model_all = lgb.LGBMRegressor(**lgbm_params)
    final_model_all.fit(X, y)
    logger.info(f"✅ Final model retrained on {len(X)} samples")
    
    # ========================================
    # Step 6: 모델 저장
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("💾 Step 6: Saving Model")
    logger.info("=" * 80)
    
    model_dir = project_root / config['output']['model_dir']
    ensure_dir(model_dir)
    
    model_path = model_dir / "simple_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model_all,  # 전체 데이터로 학습한 모델
            'feature_cols': feature_cols,
            'config': config,
            'oof_score': results['score'],
            'test_score': test_results['score'],
            'cv_models': models,
            'pipeline': pipeline
        }, f)
    
    logger.info(f"✅ Model saved: {model_path}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model_all.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = project_root / config['output']['submission_dir'] / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"\nTop 5 features:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # ========================================
    # Step 7: Kaggle 패키징
    # ========================================
    if skip_package:
        logger.info("\n" + "=" * 80)
        logger.info("📦 Step 7: Kaggle Packaging")
        logger.info("=" * 80)
        
        package_script = project_root / "scripts" / "package_for_kaggle.sh"
        try:
            result = subprocess.run(
                ["bash", str(package_script)],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            logger.info(result.stdout)
            if result.returncode == 0:
                logger.info("✅ Kaggle package created: kaggle_submission.zip")
            else:
                logger.warning(f"⚠️  Packaging failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"⚠️  Packaging error: {e}")
    
    # ========================================
    # Final Summary & Experiment Tracking
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("=" * 80)
    
    # 실험 추적
    tracker = ExperimentTracker(tracking_dir=str(project_root / "experiments"))
    
    # 실험 이름 생성
    from datetime import datetime
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 실험 정보 저장
    exp_config = {
        "cv_strategy": cv_strategy,
        "cv_splits": cv_splits,
        "train_size": 2000 if cv_strategy == 'purged_walkforward' else len(X_cv),
        "test_size": 500 if cv_strategy == 'purged_walkforward' else 0,
        "lgbm_n_estimators": lgbm_params.get('n_estimators'),
        "lgbm_learning_rate": lgbm_params.get('learning_rate'),
    }
    
    exp_results = {
        "oof_score": results['score'],
        "oof_sharpe": results['sharpe'],
        "test_score": test_results['score'],
        "test_sharpe": test_results['sharpe'],
        "cv_mean": np.mean(fold_scores),
        "cv_std": np.std(fold_scores),
        "n_features": len(feature_cols)
    }
    
    exp_notes = f"Purged WF baseline - Top feature: {importance_df.iloc[0]['feature']}"
    
    exp_dir = tracker.log_experiment(
        name=exp_name,
        config=exp_config,
        results=exp_results,
        notes=exp_notes
    )
    
    logger.info(f"\n💾 Experiment logged: {exp_name}")
    logger.info(f"   Directory: {exp_dir}")
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  • Experiment: {exp_name}")
    logger.info(f"  • CV Strategy: {cv_strategy}")
    logger.info(f"  • CV Folds: {cv_splits}")
    logger.info(f"  • OOF Score: {results['score']:.6f}")
    logger.info(f"  • Test Score: {test_results['score']:.6f} (Last 180 days)")
    logger.info(f"  • Fold Scores: {', '.join([f'{s:.4f}' for s in fold_scores])}")
    logger.info(f"  • CV Mean ± Std: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    logger.info(f"  • Model: {model_path}")
    logger.info(f"  • Features: {len(feature_cols)}")
    logger.info(f"\n📌 Next Steps:")
    logger.info(f"  1. Review Test score: {test_results['score']:.6f}")
    logger.info(f"  2. View experiments: experiments/experiments.csv")
    logger.info(f"  3. Upload kaggle_submission.zip to Kaggle Dataset")
    logger.info(f"  4. Submit using kaggle_submission_universal.ipynb")
    logger.info(f"  5. Compare with expected Public LB: ~{test_results['score']:.3f}")
    logger.info("=" * 80)


def main():
    import argparse
    config = load_config()
    # parser = argparse.ArgumentParser(description="Run full training pipeline")
    # parser.add_argument('--cv-splits', type=int, default=config['cv']['n_splits'], help='Number of CV splits')
    # parser.add_argument('--skip-package', action='store_true', default=config['output']['skip_package'], help='Skip Kaggle packaging')
    # parser.add_argument('--cv-strategy', type=str, default=config['cv']['strategy'],
    #                    choices=['timeseries', 'purged_walkforward'],
    #                    help='CV strategy to use')
    # args = parser.parse_args()
    
    run_full_pipeline(
        cv_splits=config['cv']['n_splits'],
        skip_package=config['output']['skip_package'],
        cv_strategy=config['cv']['strategy']
    )


if __name__ == "__main__":
    main()
