
import sys
from pathlib import Path
project_root = Path('.').absolute()
sys.path.insert(0, str(project_root))

import pandas as pd
from src.utils import load_config
from src.pipeline import create_pipeline

config = load_config()
train_path = project_root / config['data']['train']
df = pd.read_csv(train_path)

print("Columns in raw df:", df.columns.tolist())
if 'market_forward_excess_returns' in df.columns:
    print("market_forward_excess_returns stats:")
    print(df['market_forward_excess_returns'].describe())
else:
    print("market_forward_excess_returns NOT found in raw df")

pipeline = create_pipeline(
    fillna_strategy=config['features']['fill_missing_strategy'],
    add_interactions=config['features']['add_interactions'],
    use_time_series_features=config['features']['use_time_series_features'],
    use_advanced_features=config['features']['use_advanced_features'],
    use_market_regime_features=config['features'].get('use_market_regime_features'),
)

X, y = pipeline.fit_transform(df)
print("\nColumns in X (first 10):", X.columns[:10].tolist())
