import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_oof(oof_path):
    print(f"📊 Analyzing: {oof_path}")
    df = pd.read_csv(oof_path)
    
    # 1. Allocation Statistics
    zero_alloc = (df['allocation'] == 0).mean() * 100
    mean_alloc = df['allocation'].mean()
    print(f"\n1️⃣ Allocation Stats:")
    print(f"   - Zero Allocation: {zero_alloc:.2f}%")
    print(f"   - Mean Allocation: {mean_alloc:.4f}")
    
    # 2. Prediction Statistics
    mean_pred_return = df['pred_return'].mean()
    pos_pred_return = (df['pred_return'] > 0).mean() * 100
    print(f"\n2️⃣ Prediction Stats:")
    print(f"   - Mean Predicted Return: {mean_pred_return:.6f}")
    print(f"   - Positive Predictions: {pos_pred_return:.2f}%")
    print(f"   - Mean Predicted Risk: {df['pred_risk'].mean():.6f}")
    
    # 3. Market vs Strategy
    # Cumulative Returns
    df['market_cum'] = (1 + df['actual_return']).cumprod()
    df['strategy_return'] = df['allocation'] * df['actual_return']
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod()
    
    market_total = df['market_cum'].iloc[-1] - 1
    strategy_total = df['strategy_cum'].iloc[-1] - 1
    print(f"\n3️⃣ Performance:")
    print(f"   - Market Total Return: {market_total*100:.2f}%")
    print(f"   - Strategy Total Return: {strategy_total*100:.2f}%")
    
    # 4. Plots
    plt.figure(figsize=(15, 10))
    
    # Distribution of Predicted Returns
    plt.subplot(2, 2, 1)
    sns.histplot(df['pred_return'], kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Distribution of Predicted Returns')
    
    # Predicted Return vs Risk
    plt.subplot(2, 2, 2)
    plt.scatter(df['pred_risk'], df['pred_return'], alpha=0.1)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Risk')
    plt.ylabel('Predicted Return')
    plt.title('Risk vs Return')
    
    # Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(df['market_cum'], label='Market')
    plt.plot(df['strategy_cum'], label='Strategy')
    plt.legend()
    plt.title('Cumulative Returns')
    
    output_plot = Path(oof_path).parent / 'analysis_plot.png'
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"\n📈 Plot saved to: {output_plot}")

if __name__ == "__main__":
    analyze_oof("results/oof_predictions.csv")
