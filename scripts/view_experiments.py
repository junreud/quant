"""
실험 결과 조회 스크립트

experiments.csv의 결과를 예쁘게 출력합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiment_tracker import ExperimentTracker
import pandas as pd


def main():
    tracker = ExperimentTracker()
    
    print("=" * 80)
    print("📊 EXPERIMENT TRACKING")
    print("=" * 80)
    
    # 전체 실험 summary
    df = tracker.get_summary()
    
    if df.empty:
        print("\n⚠️  No experiments found!")
        return
    
    print(f"\n📋 Total Experiments: {len(df)}")
    
    # 주요 컬럼만 출력
    key_cols = ['name', 'timestamp']
    result_cols = [col for col in df.columns if col.startswith('result_')]
    display_cols = key_cols + result_cols
    
    print("\n" + "=" * 80)
    print("Recent Experiments:")
    print("=" * 80)
    print(df[display_cols].tail(10).to_string(index=False))
    
    # 최고 점수 실험
    if 'result_test_score' in df.columns:
        print("\n" + "=" * 80)
        print("🏆 Best Experiment (Test Score):")
        print("=" * 80)
        best = tracker.get_best_experiment('result_test_score')
        for key, value in best.items():
            if key.startswith('result_') or key == 'name':
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print(f"📁 Full data: {tracker.summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
