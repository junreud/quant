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
