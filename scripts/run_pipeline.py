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
        # Feature Selection (Explicitly handled later)
        # use_feature_selection=False 
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
    oof_risk_predictions = np.zeros(len(X_cv)) # Risk OOF
    oof_risk2_predictions = np.zeros(len(X_cv)) # Risk 2 OOF (Market Regime)
    oof_mask = np.zeros(len(X_cv), dtype=bool)
    
    models_return = []
    models_risk = []
    models_risk2 = []
    
    # Custom Objectives & Metrics
    from src.custom_objectives import correlation_metric
    
    # 1. Return Model Params
    return_lgbm_params = config['lgbm_return'].copy()
    
    # 2. Risk Model Params
    risk_lgbm_params = config['lgbm_risk'].copy()

    # 3. Risk Model 2 Params (Market Regime)
    risk2_lgbm_params = config.get('lgbm_risk2', config['lgbm_return'].copy())
    
    # FeatureSelector를 사용하여 Risk Target(abs returns)과 상관관계가 높은 Feature 선별
    from src.feature_selection import FeatureSelector
    feature_selector = FeatureSelector()
    
    # 1. Return Model Feature Selection (Importance)
    # We need to select features for Return Model explicitly now
    return_top_k = config['features']['feature_selection']['return_model']['top_k']
    logger.info(f"Selecting Top {return_top_k} features for Return Model...")
    
    return_features = feature_selector.select_by_importance(
        X, y, top_k=return_top_k,
        lgbm_params={'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'seed': 42}
    )
    logger.info(f"Return Model Features: {len(return_features)}")

    # 2. Risk Model Feature Selection (Correlation)
    # Risk Target 생성
    
    # Risk Target 생성
    y_risk_target = y.abs()
    
    # Risk 2 Target (Market Regime)
    # Use market_forward_excess_returns if available, else use forward_returns
    if 'market_forward_excess_returns' in df.columns:
        # Ensure alignment with X (in case pipeline dropped rows)
        y_risk2_target = df.loc[X.index, 'market_forward_excess_returns']
    else:
        logger.warning("⚠️ 'market_forward_excess_returns' not found. Using 'forward_returns' for Risk Model 2.")
        y_risk2_target = y
    
    risk_top_k = config['features']['feature_selection']['risk_model']['top_k']
    risk_features = feature_selector.select_by_correlation(X, y_risk_target, method='spearman', top_k=risk_top_k)
    logger.info(f"Risk Model Features (Top {risk_top_k}): {risk_features[:10]} ...")
    
    # Risk 2 Feature Selection (Correlation + Crash Divergence)
    risk2_top_k = config['features']['feature_selection']['risk2_model']['top_k']
    
    # 1. Correlation
    risk2_corr_features = feature_selector.select_by_correlation(X, y_risk2_target, method='spearman', top_k=risk2_top_k)
    
    # 2. Crash Divergence (Auto-detect crash features)
    # Use default 5% quantile and top 20 features (or read from config if added)
    risk2_crash_features = feature_selector.select_by_crash_divergence(
        X, y_risk2_target, 
        crash_threshold_quantile=config['features']['feature_selection']['risk2_model']['crash_threshold_quantile'], 
        top_k=config['features']['feature_selection']['risk2_model']['top_k'] # Select top 50 divergent features
    )
    
    # Combine and deduplicate
    risk2_features = list(set(risk2_corr_features + risk2_crash_features))
    logger.info(f"Risk Model 2 Features: {len(risk2_features)} (Corr: {len(risk2_corr_features)}, Crash: {len(risk2_crash_features)})")
    logger.info(f"Top Crash Features: {risk2_crash_features[:10]}")
    
    # Metric calculator
    metric_calculator = CompetitionMetric(
        vol_threshold=config['metric']['vol_threshold'],
        use_return_penalty=config['metric']['use_return_penalty'],
        min_periods=config['metric']['min_periods']
    )
    
    fold_scores = []
    
    from src.allocation import risk_adjusted_allocation, triple_model_allocation
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_cv)):
        logger.info(f"\n  Fold {fold_idx + 1}/{cv_splits} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        
        X_train_fold = X_cv.iloc[train_idx]
        y_train_fold = y_cv.iloc[train_idx]
        X_val_fold = X_cv.iloc[val_idx]
        y_val_fold = y_cv.iloc[val_idx]
        
        # 1. Return Model Training
        model_return = lgb.LGBMRegressor(**return_lgbm_params)
        model_return.fit(
            X_train_fold[return_features], y_train_fold,
            eval_set=[(X_val_fold[return_features], y_val_fold)],
            eval_metric=correlation_metric, # Use IC for Early Stopping
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 2. Risk Model Training (Target: abs(returns))
        y_train_risk = y_train_fold.abs()
        y_val_risk = y_val_fold.abs()
        
        model_risk = lgb.LGBMRegressor(**risk_lgbm_params)
        model_risk.fit(
            X_train_fold[risk_features], y_train_risk,
            eval_set=[(X_val_fold[risk_features], y_val_risk)],
            eval_metric='quantile',
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 3. Risk Model 2 Training (Target: Market Returns)
        y_train_risk2 = y_risk2_target.iloc[train_idx]
        y_val_risk2 = y_risk2_target.iloc[val_idx]
        
        model_risk2 = lgb.LGBMRegressor(**risk2_lgbm_params)
        model_risk2.fit(
            X_train_fold[risk2_features], y_train_risk2,
            eval_set=[(X_val_fold[risk2_features], y_val_risk2)],
            eval_metric=risk2_lgbm_params['metric'],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # OOF predictions
        pred_return = model_return.predict(X_val_fold[return_features])
        pred_risk = model_risk.predict(X_val_fold[risk_features])
        pred_risk2 = model_risk2.predict(X_val_fold[risk2_features])
        
        oof_predictions[val_idx] = pred_return
        oof_risk_predictions[val_idx] = pred_risk
        oof_risk2_predictions[val_idx] = pred_risk2
        oof_mask[val_idx] = True
        
        # Fold score (Triple Model Allocation)
        # k=1.0, threshold=0.0
        fold_allocations = triple_model_allocation(
            return_pred=pred_return,
            risk_vol_pred=pred_risk,
            risk_market_pred=pred_risk2,
            market_threshold=config['allocation']['threshold'],
            k=config['allocation']['k']
        )
        
        val_df_fold = df_cv.iloc[val_idx]
        
        fold_result = metric_calculator.calculate_score(
            allocations=fold_allocations,
            forward_returns=val_df_fold['forward_returns'].values,
            market_returns=val_df_fold['forward_returns'].values,
            risk_free_rate=val_df_fold['risk_free_rate'].values
        )
        
        fold_scores.append(fold_result['score'])
        models_return.append(model_return)
        models_risk.append(model_risk)
        models_risk2.append(model_risk2)
        
            # Log metrics (Handle different metric names safely)
        ret_score = model_return.best_score_['valid_0'].get('correlation', 0.0)
        risk_score = model_risk.best_score_['valid_0'].get('quantile', 0.0)
        risk2_score = model_risk2.best_score_['valid_0'].get(risk2_lgbm_params['metric'], 0.0)
        logger.info(f"  ✅ Fold {fold_idx + 1} Score: {fold_result['score']:.6f} | Ret IC: {ret_score:.4f} | Risk Q: {risk_score:.4f} | Mkt {risk2_lgbm_params['metric']}: {risk2_score:.4f}")
    
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
    oof_pred_return = oof_predictions[oof_mask]
    oof_df = df_cv[oof_mask]
    oof_pred_return = oof_predictions[oof_mask]
    oof_pred_risk = oof_risk_predictions[oof_mask]
    oof_pred_risk2 = oof_risk2_predictions[oof_mask]
    
    # Allocation
    # Use Triple Model Allocation
    allocations = triple_model_allocation(
        return_pred=oof_pred_return,
        risk_vol_pred=oof_pred_risk,
        risk_market_pred=oof_pred_risk2,
        market_threshold=config['allocation']['threshold'],
        k=config['allocation']['k']
    )
    
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
    
    # Save OOF predictions for analysis
    oof_save_path = project_root / config['output']['submission_dir'] / 'oof_predictions.csv'
    oof_df_save = pd.DataFrame({
        'date_id': oof_df['date_id'] if 'date_id' in oof_df.columns else oof_df.index,
        'actual_return': oof_df['forward_returns'].values,
        'pred_return': oof_pred_return,
        'pred_return': oof_pred_return,
        'pred_risk': oof_pred_risk,
        'pred_risk2': oof_pred_risk2,
        'allocation': allocations
    })
    oof_df_save.to_csv(oof_save_path, index=False)
    logger.info(f"💾 OOF predictions saved to: {oof_save_path}")
    
    # ========================================
    # Step 4: 최종 모델 학습 (전체 CV 데이터)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🏁 Step 4: Final Model Training (CV data only)")
    logger.info("=" * 80)
    
    # Return Model
    final_model_return = lgb.LGBMRegressor(**return_lgbm_params)
    final_model_return.fit(X_cv[return_features], y_cv, eval_metric=correlation_metric)
    
    # Risk Model
    final_model_risk = lgb.LGBMRegressor(**risk_lgbm_params)
    final_model_risk.fit(X_cv[risk_features], y_cv.abs(), eval_metric='quantile')
    
    # Risk Model 2
    final_model_risk2 = lgb.LGBMRegressor(**risk2_lgbm_params)
    final_model_risk2.fit(X_cv[risk2_features], y_risk2_target.iloc[:len(X_cv)], eval_metric=risk2_lgbm_params['metric'])
    
    logger.info(f"✅ Final models trained on {len(X_cv)} samples")
    
    # ========================================
    # Step 5: 최종 테스트 (마지막 180일)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🧪 Step 5: Final Test Evaluation (Last 180 days)")
    logger.info("=" * 80)
    
    test_pred_ret = final_model_return.predict(X_test[return_features])
    test_pred_ret = final_model_return.predict(X_test[return_features])
    test_pred_risk = final_model_risk.predict(X_test[risk_features])
    test_pred_risk2 = final_model_risk2.predict(X_test[risk2_features])
    
    # Triple Allocation for Test
    test_allocations = triple_model_allocation(
        return_pred=test_pred_ret,
        risk_vol_pred=test_pred_risk,
        risk_market_pred=test_pred_risk2,
        market_threshold=config['allocation']['threshold'],
        k=config['allocation']['k']
    )
    
    test_results = metric_calculator.calculate_score(
        allocations=test_allocations,
        forward_returns=df_test['forward_returns'].values,
        market_returns=df_test['forward_returns'].values,
        risk_free_rate=df_test['risk_free_rate'].values
    )
    
    # Save Test predictions for analysis
    test_save_path = project_root / config['output']['submission_dir'] / 'test_predictions.csv'
    test_df_save = pd.DataFrame({
        'date_id': df_test['date_id'] if 'date_id' in df_test.columns else df_test.index,
        'actual_return': df_test['forward_returns'].values,
        'pred_return': test_pred_ret,
        'pred_return': test_pred_ret,
        'pred_risk': test_pred_risk,
        'pred_risk2': test_pred_risk2,
        'allocation': test_allocations,
    })
    test_df_save.to_csv(test_save_path, index=False)
    logger.info(f"💾 Test predictions saved to: {test_save_path}")
    
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
    
    final_model_return_all = lgb.LGBMRegressor(**return_lgbm_params)
    final_model_return_all.fit(X[return_features], y)
    
    final_model_risk_all = lgb.LGBMRegressor(**risk_lgbm_params)
    final_model_risk_all = lgb.LGBMRegressor(**risk_lgbm_params)
    final_model_risk_all.fit(X[risk_features], y.abs())
    
    final_model_risk2_all = lgb.LGBMRegressor(**risk2_lgbm_params)
    final_model_risk2_all.fit(X[risk2_features], y_risk2_target)
    
    logger.info(f"✅ Final models retrained on {len(X)} samples")
    
    # ========================================
    # Step 6: 모델 저장
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("💾 Step 6: Saving Model")
    logger.info("=" * 80)
    
    model_dir = project_root / config['output']['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model
    model_save_path = model_dir / 'dual_model.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model_return': final_model_return_all,
            'model_risk': final_model_risk_all,
            'model_risk2': final_model_risk2_all,
            'pipeline': pipeline,
            'feature_cols': pipeline.feature_engineer.feature_cols,
            'config': config
        }, f)
        
    logger.info(f"✅ Model saved: {model_save_path}")
    
    # Save History for Inference (Last 200 rows)
    # This is needed to calculate rolling features for the first few test samples
    history_save_path = model_dir / 'history.pkl'
    # Save only necessary columns for feature engineering
    # We need raw columns, so we take them from the original df (before transform)
    # But wait, 'df' variable in main() is already loaded.
    # We need to ensure we have the raw data. 
    # In run_full_pipeline, 'df' is the raw dataframe loaded from train.csv.
    # We should use that.
    history_df = df.tail(200).reset_index(drop=True)
    history_df.to_pickle(history_save_path)
    logger.info(f"✅ History buffer saved: {history_save_path} (200 samples)")
    
    # Save Prediction History for Dynamic Allocation
    # We use the OOF predictions (or we could predict on the history_df)
    # Since history_df is the last 200 rows of TRAIN, we should use predictions on THEM.
    # But we don't have predictions on history_df readily available unless they are in OOF.
    # If history_df is part of the last fold's validation, it's in OOF.
    # If we retrained on ALL data, we should ideally predict on history_df with the FINAL model.
    # Let's do that to be consistent with the model we ship.
    
    logger.info("🔮 Generating prediction history for inference buffer...")
    # Preprocess history_df
    X_history = pipeline.transform(history_df, return_target=False)
    
    # Predict with FINAL models
    pred_history_return = final_model_return_all.predict(X_history[return_features])
    
    pred_history_path = model_dir / 'pred_history.pkl'
    with open(pred_history_path, 'wb') as f:
        pickle.dump(list(pred_history_return), f)
        
    logger.info(f"✅ Prediction history saved: {pred_history_path} ({len(pred_history_return)} samples)")
    
    ensure_dir(model_dir)
    
    model_path = model_dir / "dual_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model_return': final_model_return_all,
            'model_risk': final_model_risk_all,
            'model_risk2': final_model_risk2_all,
            'feature_cols': feature_cols, # All generated features
            'return_features': return_features,
            'risk_features': risk_features,
            'risk2_features': risk2_features,
            'config': config,
            'oof_score': results['score'],
            'test_score': test_results['score'],
            'pipeline': pipeline
        }, f)
    
    logger.info(f"✅ Model saved: {model_path}")
    
    # Feature importance (Return Model)
    importance_df = pd.DataFrame({
        'feature': return_features,
        'importance': final_model_return_all.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = project_root / config['output']['submission_dir'] / 'feature_importance_return.csv'
    importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"\nTop 5 features (Return Model):")
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
        "return_n_estimators": return_lgbm_params.get('n_estimators'),
        "return_learning_rate": return_lgbm_params.get('learning_rate'),
        "risk_n_estimators": risk_lgbm_params.get('n_estimators'),
        "risk_learning_rate": risk_lgbm_params.get('learning_rate'),
        "k": config['allocation']['k'],
        "threshold": config['allocation']['threshold'],
        "return_top_k": config['feature_selection']['return_top_k'],
        "risk_top_k": config['feature_selection']['risk_top_k'],
        "risk2_top_k": config['feature_selection']['risk2_top_k'],
    }
    
    exp_results = {
        "oof_score": results['score'],
        "oof_sharpe": results['sharpe'],
        "oof_vol_penalty": results['vol_penalty'],
        "oof_return_penalty": results['return_penalty'],
        "test_score": test_results['score'],
        "test_sharpe": test_results['sharpe'],
        "test_vol_penalty": test_results['vol_penalty'],
        "test_return_penalty": test_results['return_penalty'],
        "cv_mean": np.mean(fold_scores),
        "cv_std": np.std(fold_scores),
        "n_features": len(feature_cols),
        # Model Performance Metrics (Approximate from last fold or mean if tracked)
        # We didn't track mean IC/Quantile across folds in variables, but we can add them.
        # For now, let's just log the OOF score which is the main one.
        # But user asked for "individual model performance".
        # Let's calculate them on OOF predictions.
        "return_ic": np.corrcoef(oof_pred_return, oof_df['forward_returns'])[0, 1],
        "risk_quantile": np.mean(np.maximum(0.75 * (oof_df['forward_returns'].abs() - oof_pred_risk), (0.75 - 1) * (oof_df['forward_returns'].abs() - oof_pred_risk))), # Negative of score
        # Risk 2 RMSE (Market Regime)
        # We need y_risk2_target aligned with OOF.
        # It's tricky because y_risk2_target might be different from forward_returns.
        # But we have oof_pred_risk2.
        # Let's skip complex calc here and rely on CV logs or add properly later.
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
