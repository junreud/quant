"""
Utility functions for quant strategy project.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import yaml


def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    로거 설정 및 반환.
    
    Parameters
    ----------
    name : str
        로거 이름
    level : str
        로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns
    -------
    logging.Logger
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    YAML 설정 파일 로드.
    
    Parameters
    ----------
    config_path : str
        설정 파일 경로
        
    Returns
    -------
    dict
        설정 딕셔너리
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    # Check for best_params.yaml and merge if exists
    best_params_file = config_file.parent / "best_params.yaml"
    if best_params_file.exists():
        try:
            with open(best_params_file, 'r') as f:
                best_params = yaml.safe_load(f)
            
            if best_params:
                # Deep merge or simple update?
                # Simple update is enough for top-level keys like 'lgbm_return', 'lgbm_risk'
                # But we should be careful not to overwrite other things if best_params is partial.
                # The tuner saves full parameter dicts for these keys.
                
                # Update specific keys if they exist in best_params
                for key in ['lgbm_return', 'lgbm_risk']:
                    if key in best_params:
                        if key in config and isinstance(config[key], dict) and isinstance(best_params[key], dict):
                            config[key].update(best_params[key])
                        else:
                            config[key] = best_params[key]
                        # logger is not available inside function easily, but we can print
                        print(f"✅ Loaded best parameters for {key} from {best_params_file}")
                        
        except Exception as e:
            print(f"⚠️ Failed to load best_params.yaml: {e}")
    
    return config


def ensure_dir(path: Path) -> None:
    """
    디렉토리가 없으면 생성.
    
    Parameters
    ----------
    path : Path
        생성할 디렉토리 경로
    """
    path.mkdir(parents=True, exist_ok=True)


logger = get_logger()
