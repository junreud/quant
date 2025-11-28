"""
Phase 1: 베이스라인 모델 (Position = 1.0)

이 스크립트는 가장 단순한 전략을 구현합니다:
- 모든 날짜에 대해 Position = 1.0 (시장 추종)
- 평가 함수로 점수 계산
- 제출 파일 생성

목적: 
1. 평가 지표 함수가 올바르게 작동하는지 확인
2. 베이스라인 점수 확인
3. 데이터 파이프라인 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.metric import CompetitionMetric
from src.utils import get_logger, load_config, ensure_dir

logger = get_logger(name="baseline", level="INFO")


def load_data(config: dict) -> tuple:
    """
    데이터 로드.
    
    Returns
    -------
    tuple
        (train_df, test_df)
    """
    logger.info("📁 데이터 로드 중...")
    
    train_path = project_root / config['data']['train']
    test_path = project_root / config['data']['test']
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    logger.info(f"   Train shape: {train.shape}")
    logger.info(f"   Test shape: {test.shape}")
    
    return train, test


def create_baseline_submission(test: pd.DataFrame) -> pd.DataFrame:
    """
    베이스라인 제출 파일 생성 (Position = 1.0).
    
    Parameters
    ----------
    test : pd.DataFrame
        테스트 데이터
        
    Returns
    -------
    pd.DataFrame
        제출 파일 (date_id, allocation)
    """
    submission = pd.DataFrame({
        'date_id': test['date_id'],
        'allocation': 1.0  # 모든 날짜에 대해 시장 추종
    })
    
    return submission


def evaluate_baseline(train: pd.DataFrame, config: dict) -> dict:
    """
    Train 데이터로 베이스라인 전략 평가.
    
    Parameters
    ----------
    train : pd.DataFrame
        학습 데이터
    config : dict
        설정
        
    Returns
    -------
    dict
        평가 지표
    """
    logger.info("📊 베이스라인 전략 평가 중...")
    
    # Position = 1.0 (시장 추종)
    allocations = np.ones(len(train))
    forward_returns = train['forward_returns'].values
    risk_free_rate = train['risk_free_rate'].values
    
    # 평가 함수 초기화
    metric_config = config['metric']
    metric_calculator = CompetitionMetric(
        vol_threshold=metric_config['vol_threshold'],
        use_return_penalty=metric_config['use_return_penalty'],
        min_periods=metric_config['min_periods']
    )
    
    # 점수 계산
    results = metric_calculator.calculate_score(
        allocations=allocations,
        forward_returns=forward_returns,
        market_returns=forward_returns,  # 시장 수익률 = forward_returns
        risk_free_rate=risk_free_rate
    )
    
    return results


def print_evaluation_results(results: dict) -> None:
    """
    평가 결과 출력.
    
    Parameters
    ----------
    results : dict
        평가 지표 딕셔너리
    """
    logger.info("=" * 80)
    logger.info("📈 베이스라인 평가 결과 (Position = 1.0)")
    logger.info("=" * 80)
    logger.info(f"🎯 최종 점수 (Adjusted Sharpe): {results['score']:.6f}")
    logger.info("")
    logger.info(f"📊 Sharpe Ratio (before penalty): {results['sharpe']:.6f}")
    logger.info(f"⚠️  Volatility Penalty: {results['vol_penalty']:.6f}")
    logger.info(f"⚠️  Return Penalty: {results['return_penalty']:.6f}")
    logger.info("")
    logger.info(f"📉 전략 변동성: {results['strategy_vol']:.2f}%")
    logger.info(f"📉 시장 변동성: {results['market_vol']:.2f}%")
    logger.info(f"📊 변동성 비율: {results['vol_ratio']:.4f}")
    logger.info("")
    logger.info(f"💰 전략 평균 수익률: {results['strategy_mean_excess_return']:.6f}")
    logger.info(f"💰 시장 평균 수익률: {results['market_mean_excess_return']:.6f}")
    logger.info(f"📊 수익률 갭: {results['return_gap']:.4f}%")
    logger.info("")
    logger.info(f"✅ 유효 데이터 개수: {results['n_valid']}")
    logger.info("=" * 80)


def main():
    """메인 실행 함수."""
    logger.info("🚀 Phase 1: 베이스라인 모델 (Position = 1.0) 시작")
    logger.info("")
    
    # 설정 로드
    config = load_config()
    
    # 데이터 로드
    train, test = load_data(config)
    
    # 베이스라인 평가 (Train 데이터)
    results = evaluate_baseline(train, config)
    print_evaluation_results(results)
    
    # 제출 파일 생성
    logger.info("")
    logger.info("📝 제출 파일 생성 중...")
    submission = create_baseline_submission(test)
    
    # 결과 저장
    output_dir = project_root / config['output']['submission_dir']
    ensure_dir(output_dir)
    
    submission_path = output_dir / "baseline_submission.csv"
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"✅ 제출 파일 저장 완료: {submission_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Phase 1 완료!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📌 다음 단계:")
    logger.info("   1. 베이스라인 점수 확인")
    logger.info("   2. Phase 2: 알파 예측 모델 개발")
    logger.info("")


if __name__ == "__main__":
    main()
