
import pandas as pd
import numpy as np
from pathlib import Path

# Load Data
project_root = Path(".")
oof_path = project_root / 'results' / 'oof_predictions.csv'

if not oof_path.exists():
    print(f"❌ File not found: {oof_path}")
    exit(1)

df = pd.read_csv(oof_path)

# Calculate Volatility Regime
df['market_vol_20d'] = df['actual_return'].rolling(20).std()
df = df.dropna(subset=['market_vol_20d']).copy()
df['vol_regime'] = pd.qcut(df['market_vol_20d'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Filter Q3 and Q4
q3 = df[df['vol_regime'] == 'Q3']
q4 = df[df['vol_regime'] == 'Q4']

def get_stats(data, name):
    ic = data['pred_return'].corr(data['actual_return'])
    win_rate = (data['actual_return'] > 0).mean()
    avg_return = data['actual_return'].mean() * 252
    avg_allocation = data['allocation'].mean()
    avg_pred_risk = data['pred_risk'].mean()
    avg_pred_risk2 = data['pred_risk2'].mean()
    
    # Directional Accuracy: Did model predict sign correctly?
    # (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
    directional_acc = ((np.sign(data['pred_return']) == np.sign(data['actual_return']))).mean()

    return {
        "Regime": name,
        "Count": len(data),
        "IC (Pred Accuracy)": f"{ic:.4f}",
        "Directional Acc": f"{directional_acc:.2%}",
        "Avg Allocation": f"{avg_allocation:.2f}",
        "Win Rate (Market)": f"{win_rate:.2%}",
        "Avg Mkt Return (Ann)": f"{avg_return:.2%}",
        "Avg Pred Risk (Model B)": f"{avg_pred_risk:.4f}",
        "Avg Pred Risk2 (Crash)": f"{avg_pred_risk2:.4f}"
    }

stats_q3 = get_stats(q3, "Q3 (Medium Vol)")
stats_q4 = get_stats(q4, "Q4 (High Vol)")

print(f"{'Metric':<25} | {'Q3 (Medium Vol)':<15} | {'Q4 (High Vol)':<15}")
print("-" * 60)
for key in stats_q3.keys():
    if key == "Regime": continue
    print(f"{key:<25} | {stats_q3[key]:<15} | {stats_q4[key]:<15}")
