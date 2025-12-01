
import json
import nbformat
from pathlib import Path

notebook_path = Path("notebooks/04_model_performance_analysis.ipynb")

if not notebook_path.exists():
    print(f"❌ Notebook not found: {notebook_path}")
    exit(1)

print(f"📖 Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 4. OOF Deep Dive Analysis\n",
            "Analyzing the Out-of-Fold (OOF) predictions to understand model behavior across different market regimes and identify strengths/weaknesses."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load OOF Data\n",
            "oof_path = project_root / 'results' / 'oof_predictions.csv'\n",
            "if not oof_path.exists():\n",
            "    print(f\"File not found: {oof_path}\")\n",
            "else:\n",
            "    oof_df = pd.read_csv(oof_path)\n",
            "    print(f\"Loaded {len(oof_df)} rows from {oof_path}\")\n",
            "    display(oof_df.head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.1 Market Regime Analysis (Volatility Quintiles)\n",
            "How does the model perform in calm vs. volatile markets?\n",
            "- **Method**: Divide data into 5 regimes based on Rolling Volatility (20D Std).\n",
            "- **Metrics**: Sharpe Ratio, Win Rate, and Average Return per regime."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate Rolling Volatility (Market Regime Proxy)\n",
            "oof_df['market_vol_20d'] = oof_df['actual_return'].rolling(20).std()\n",
            "\n",
            "# Drop NaN from rolling\n",
            "analysis_df = oof_df.dropna(subset=['market_vol_20d']).copy()\n",
            "\n",
            "# Create Volatility Quintiles (Q1=Calm, Q5=Volatile)\n",
            "analysis_df['vol_regime'] = pd.qcut(analysis_df['market_vol_20d'], 5, labels=['Q1 (Calm)', 'Q2', 'Q3', 'Q4', 'Q5 (Volatile)'])\n",
            "\n",
            "# Calculate Performance by Regime\n",
            "regime_stats = analysis_df.groupby('vol_regime').apply(lambda x: pd.Series({\n",
            "    'Sharpe': x['allocation'].mul(x['actual_return']).mean() / x['allocation'].mul(x['actual_return']).std() * np.sqrt(252) if x['allocation'].mul(x['actual_return']).std() != 0 else 0,\n",
            "    'Avg_Return': x['allocation'].mul(x['actual_return']).mean() * 252,\n",
            "    'Win_Rate': (x['allocation'].mul(x['actual_return']) > 0).mean(),\n",
            "    'Avg_Allocation': x['allocation'].mean(),\n",
            "    'Count': len(x)\n",
            "}))\n",
            "\n",
            "print(\"Performance by Market Regime (Volatility):\")\n",
            "display(regime_stats)\n",
            "\n",
            "# Plot\n",
            "fig, ax = plt.subplots(1, 2, figsize=(15, 6))\n",
            "regime_stats['Sharpe'].plot(kind='bar', ax=ax[0], title='Sharpe Ratio by Volatility Regime', color='skyblue')\n",
            "ax[0].axhline(0, color='grey', linestyle='--')\n",
            "regime_stats['Avg_Allocation'].plot(kind='bar', ax=ax[1], title='Average Allocation by Regime', color='orange')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.2 Prediction Quality Analysis (Return Deciles)\n",
            "When the model predicts \"High Return\", does the market actually go up?\n",
            "- **Method**: Divide `pred_return` into 10 Deciles.\n",
            "- **Expectation**: Monotonic increase in Actual Return from D1 to D10."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create Prediction Deciles\n",
            "analysis_df['pred_decile'] = pd.qcut(analysis_df['pred_return'], 10, labels=False) + 1\n",
            "\n",
            "# Calculate Avg Actual Return per Decile\n",
            "decile_stats = analysis_df.groupby('pred_decile')['actual_return'].mean()\n",
            "\n",
            "# Plot\n",
            "plt.figure(figsize=(10, 6))\n",
            "decile_stats.plot(kind='bar', color='teal', alpha=0.7)\n",
            "plt.title(\"Average Actual Return by Prediction Decile (Monotonicity Check)\")\n",
            "plt.xlabel(\"Prediction Decile (1=Lowest Pred, 10=Highest Pred)\")\n",
            "plt.ylabel(\"Average Actual Return\")\n",
            "plt.axhline(0, color='grey', linestyle='--')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.3 Error Analysis (Big Losses & Missed Opportunities)\n",
            "Where did we lose money? Where did we miss out?"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate Strategy Returns\n",
            "analysis_df['strat_return'] = analysis_df['allocation'] * analysis_df['actual_return']\n",
            "\n",
            "# 1. Big Losses (Bottom 1% of Strategy Returns)\n",
            "worst_days = analysis_df.nsmallest(int(len(analysis_df)*0.01), 'strat_return')\n",
            "print(\"\\n📉 Top 5 Worst Days (Big Losses):\")\n",
            "display(worst_days[['date_id', 'actual_return', 'pred_return', 'pred_risk', 'pred_risk2', 'allocation', 'strat_return']].head())\n",
            "\n",
            "# 2. Missed Opportunities (Top 1% Market Returns where Allocation < 0.5)\n",
            "missed_opps = analysis_df[(analysis_df['actual_return'] > analysis_df['actual_return'].quantile(0.99)) & (analysis_df['allocation'] < 0.5)]\n",
            "print(\"\\n🤷 Missed Opportunities (High Market Return, Low Allocation):\")\n",
            "display(missed_opps[['date_id', 'actual_return', 'pred_return', 'pred_risk', 'pred_risk2', 'allocation']].head())"
        ]
    }
]

# Append cells
nb['cells'].extend(new_cells)

print(f"✅ Appending {len(new_cells)} cells...")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("🎉 Done! Notebook updated.")
