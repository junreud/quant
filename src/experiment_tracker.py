"""
실험 추적 모듈

실험 결과를 자동으로 저장하고 관리합니다.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ExperimentTracker:
    """
    실험 결과 추적기.
    
    사용법:
        tracker = ExperimentTracker()
        tracker.log_experiment(
            name="exp001_baseline",
            config={"cv_splits": 5},
            results={"oof_score": 0.52}
        )
    """
    
    def __init__(self, tracking_dir: str = "experiments"):
        """
        Parameters
        ----------
        tracking_dir : str
            실험 추적 디렉토리
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(exist_ok=True)
        
        self.summary_file = self.tracking_dir / "experiments.csv"
        
    def log_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        results: Dict[str, float],
        notes: str = ""
    ) -> str:
        """
        실험 결과 저장.
        
        Parameters
        ----------
        name : str
            실험 이름 (예: "exp001_baseline")
        config : dict
            실험 설정
        results : dict
            실험 결과 (점수 등)
        notes : str
            메모
            
        Returns
        -------
        str
            실험 디렉토리 경로
        """
        # 타임스탬프
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # 실험 디렉토리 생성
        exp_dir = self.tracking_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # 상세 정보 저장 (JSON)
        exp_data = {
            "name": name,
            "timestamp": timestamp.isoformat(),
            "config": config,
            "results": results,
            "notes": notes
        }
        
        detail_file = exp_dir / f"results_{timestamp_str}.json"
        with open(detail_file, "w") as f:
            json.dump(exp_data, f, indent=2)
        
        # Summary CSV에 추가
        self._update_summary(name, timestamp, config, results, notes)
        
        return str(exp_dir)
    
    def _update_summary(
        self,
        name: str,
        timestamp: datetime,
        config: Dict[str, Any],
        results: Dict[str, float],
        notes: str
    ):
        """Summary CSV 업데이트."""
        # 기존 데이터 로드 or 새로 생성
        if self.summary_file.exists():
            df = pd.read_csv(self.summary_file)
        else:
            df = pd.DataFrame()
        
        # 새 row
        new_row = {
            "name": name,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes
        }
        
        # Config 추가
        for key, value in config.items():
            new_row[f"config_{key}"] = value
        
        # Results 추가
        for key, value in results.items():
            new_row[f"result_{key}"] = value
        
        # 추가
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 저장
        df.to_csv(self.summary_file, index=False)
    
    def get_summary(self) -> pd.DataFrame:
        """Summary 테이블 반환."""
        if self.summary_file.exists():
            return pd.read_csv(self.summary_file)
        else:
            return pd.DataFrame()
    
    def get_best_experiment(self, metric: str = "result_test_score") -> Dict:
        """
        최고 점수 실험 반환.
        
        Parameters
        ----------
        metric : str
            비교할 메트릭 (컬럼명)
            
        Returns
        -------
        dict
            최고 실험 정보
        """
        df = self.get_summary()
        if df.empty or metric not in df.columns:
            return {}
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()
    
    def compare_experiments(self, exp_names: list = None) -> pd.DataFrame:
        """
        실험 비교 테이블 생성.
        
        Parameters
        ----------
        exp_names : list, optional
            비교할 실험 이름 리스트 (None이면 전체)
            
        Returns
        -------
        pd.DataFrame
            비교 테이블
        """
        df = self.get_summary()
        
        if exp_names:
            df = df[df['name'].isin(exp_names)]
        
        # 주요 컬럼만
        key_cols = ['name', 'timestamp']
        result_cols = [col for col in df.columns if col.startswith('result_')]
        
        return df[key_cols + result_cols]
