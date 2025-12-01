import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
# If running from notebooks dir, root is ..
# If running from project root, root is .
current_dir = Path.cwd()
if (current_dir / 'src').exists():
    project_root = current_dir
elif (current_dir.parent / 'src').exists():
    project_root = current_dir.parent
else:
    # Fallback to file location if running as script
    project_root = Path(__file__).parent.parent

sys.path.append(str(project_root))

from src.allocation import dynamic_risk_allocation, smart_allocation
from src.metric import CompetitionMetric

# Load OOF Predictions
oof_path = project_root / 'results' / 'oof_predictions.csv'
df_oof = pd.read_csv(oof_path)
print(f"Loaded {len(df_oof)} rows from {oof_path} (OOF)")

# Load Test Predictions
test_path = project_root / 'results' / 'test_predictions.csv'
if test_path.exists():
    df_test = pd.read_csv(test_path)
    print(f"Loaded {len(df_test)} rows from {test_path} (Test)")
else:
    df_test = None
    print("Test predictions not found.")

# Function to simulate strategy with new parameters
def simulate_strategy(df, k=0.5, rolling_window=20, risk_alpha=0.75):
    df_sim = df.copy()
    
    # 1. Re-calculate Rolling Stats (if needed, but OOF usually has them implicitly via time)
    # Since OOF is concatenated folds, rolling on the whole DF might be slightly inaccurate at fold boundaries,
    # but acceptable for quick simulation.
    df_sim['roll_mean'] = df_sim['pred_return'].rolling(window=rolling_window, min_periods=1).mean()
    df_sim['roll_std'] = df_sim['pred_return'].rolling(window=rolling_window, min_periods=1).std()
    
    # 2. Apply Dynamic Allocation
    # Note: We need to use the function from src.allocation or define it here
    # Let's define it here for quick iteration
    
    def local_dynamic_allocation(row):
        return_pred = row['pred_return']
        risk_pred = row['pred_risk']
        roll_mean = row['roll_mean']
        roll_std = row['roll_std']
        
        safe_std = max(roll_std, 1e-8)
        safe_risk = max(risk_pred, 1e-6)
        
        signal = return_pred - roll_mean
        z_score = signal / safe_std
        
        # New Logic: Allocation = k * Z_Score
        raw_alloc = k * z_score
        return np.clip(raw_alloc, 0.0, 2.0)
        
    df_sim['new_allocation'] = df_sim.apply(local_dynamic_allocation, axis=1)
    
    # 3. Calculate Returns
    df_sim['strategy_return'] = df_sim['new_allocation'] * df_sim['actual_return']
    df_sim['cum_return'] = (1 + df_sim['strategy_return']).cumprod()
    df_sim['market_cum'] = (1 + df_sim['actual_return']).cumprod()
    
    # 4. Metrics
    total_return = df_sim['cum_return'].iloc[-1] - 1
    sharpe = df_sim['strategy_return'].mean() / df_sim['strategy_return'].std() * np.sqrt(252)
    
    print(f"--- Simulation Results (k={k}) ---")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Zero Allocation: {(df_sim['new_allocation'] == 0).mean()*100:.1f}%")
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_sim['market_cum'], label='Market', alpha=0.5)
    plt.plot(df_sim['cum_return'], label=f'Strategy (k={k})', linewidth=2)
    plt.title(f"Cumulative Return (k={k})")
    plt.legend()
    plt.show()
    
    return df_sim

# Interactive Testing
k_values = [0.3, 0.5, 0.7]

print("\n=== OOF Analysis ===")
for k in k_values:
    simulate_strategy(df_oof, k=k)

if df_test is not None:
    print("\n=== Test Analysis ===")
    for k in k_values:
        sim_df = simulate_strategy(df_test, k=k)
        
        if k == 0.7:
            # Compare with pipeline allocation
            print("\n--- Comparison with Pipeline (k=0.7) ---")
            pipeline_alloc = df_test['allocation']
            sim_alloc = sim_df['new_allocation']
            
            corr = pipeline_alloc.corr(sim_alloc)
            print(f"Correlation between Pipeline and Sim Alloc: {corr:.4f}")
            print(f"Pipeline Mean Alloc: {pipeline_alloc.mean():.4f}")
            print(f"Sim Mean Alloc: {sim_alloc.mean():.4f}")
            
            # Check if pipeline used a different k or logic
            # Pipeline Sharpe
            pipeline_ret = pipeline_alloc * df_test['actual_return']
            pipeline_sharpe = pipeline_ret.mean() / pipeline_ret.std() * np.sqrt(252)
            print(f"Re-calculated Pipeline Sharpe: {pipeline_sharpe:.4f}")