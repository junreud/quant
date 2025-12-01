
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

# Calculate Strategy Return
df['strat_return'] = df['allocation'] * df['actual_return']

# 1. Big Losses Analysis
# Previous worst days were around -6% strategy return.
worst_days = df.nsmallest(5, 'strat_return')
print("\n📉 Top 5 Worst Days (New Model):")
print(worst_days[['date_id', 'actual_return', 'pred_return', 'pred_risk', 'pred_risk2', 'allocation', 'strat_return']].to_string(index=False))

# 2. Missed Opportunities Analysis
# High market return (> 99th percentile) but low allocation (< 0.5)
high_mkt_return = df['actual_return'].quantile(0.99)
missed_opps = df[(df['actual_return'] > high_mkt_return) & (df['allocation'] < 0.5)]
print(f"\n🤷 Missed Opportunities (Mkt > {high_mkt_return:.4f}, Alloc < 0.5): {len(missed_opps)} days")
if len(missed_opps) > 0:
    print(missed_opps[['date_id', 'actual_return', 'pred_return', 'pred_risk', 'pred_risk2', 'allocation']].head().to_string(index=False))

# 3. Market Regime Analysis (Q3 vs Q4)
df['market_vol_20d'] = df['actual_return'].rolling(20).std()
df = df.dropna(subset=['market_vol_20d']).copy()
df['vol_regime'] = pd.qcut(df['market_vol_20d'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

q3 = df[df['vol_regime'] == 'Q3']
q4 = df[df['vol_regime'] == 'Q4']

def get_stats(data):
    return {
        "IC": data['pred_return'].corr(data['actual_return']),
        "WinRate": (data['actual_return'] > 0).mean(),
        "AvgAlloc": data['allocation'].mean(),
        "AvgStratRet": data['strat_return'].mean() * 252
    }

s3 = get_stats(q3)
s4 = get_stats(q4)

print("\n📊 Regime Analysis (Q3 vs Q4):")
print(f"{'Metric':<15} | {'Q3':<10} | {'Q4':<10}")
print("-" * 40)
print(f"{'IC':<15} | {s3['IC']:.4f}     | {s4['IC']:.4f}")
print(f"{'Avg Alloc':<15} | {s3['AvgAlloc']:.4f}     | {s4['AvgAlloc']:.4f}")
print(f"{'Avg Strat Ret':<15} | {s3['AvgStratRet']:.2%}     | {s4['AvgStratRet']:.2%}")
