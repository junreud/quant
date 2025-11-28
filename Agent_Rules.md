# Project Context: S&P 500 Forecasting (v5.0 Phased Strategy & LB Sync)

## 1. Project Overview

- **Objective:** Predict daily S&P 500 returns to maximize the **Volatility-Adjusted Sharpe Ratio**.
- **Strategy:** **Phased Multi-Model Strategy** (Phase 1: Dual-Head -> Phase 2: Hybrid Defense).
- **Core Philosophy:** "Robustness over Leaderboard Performance."

## 2. Data Specifications

- **`train.csv`**: Historical market data.
- **`test.csv`**: Mock test set (for API testing).
- **Evaluation Phase:** Phase 1 Public LB scores are calculated using the **last 180 days** of `train.csv`.

## 3. Evaluation Metric (Volatility-Adjusted Sharpe)

- **Formula:** `Sharpe / (Vol_Penalty * Return_Penalty)`
- **Constraints:** `0 <= Position <= 2`

## 4. Validation Strategy (Strict LB Simulation)

To perfectly simulate the Public Leaderboard locally:

### A. Data Splitting Rule

- **`FINAL_HOLDOUT_DAYS`**: **180 days** (Corresponds to the Public LB period).
- **Training Set:** `train.csv` [:-180] (All data **EXCEPT** the last 180 days).
- **Local Hold-out Set:** `train.csv` [-180:] (The **LAST** 180 days only).

### B. Training Protocol

1.  **Development Phase:**
    - Train your models **ONLY** on the `Training Set`.
    - **NEVER** use the `Local Hold-out Set` for training or CV folds. It must remain unseen.
    - Perform Purged CV (with embargo) within the `Training Set`.
2.  **Sync Check:**
    - After training, infer predictions on the `Local Hold-out Set`.
    - Calculate the score. **This score must match the Public LB score.**
3.  **Submission Phase (Optional):**
    - Only after the model architecture is finalized and synced, you may refit on the full dataset (including the last 180 days) for the actual submission.

## 5. Modeling Guidelines (Phased Roadmap) ⭐ UPDATED

Develop the model pipeline in the following order. **Current Status: Execute Phase 1.**

### [Phase 1] Dual-Model Structure (Priority)

Implement the core logic first.

- **Model A (Return):**
  - Target: `forward_returns` (or Alpha: `forward_returns - market_returns`).
  - Role: Determine the direction and magnitude of investment (Accelerator).
- **Model B (Risk):**
  - Target: `abs(forward_returns)` (Proxy for volatility).
  - Role: Determine the risk level to scale down the position (Brake).
- **Allocation Logic:**
  - `Position = k * (Model_A_Pred / Model_B_Pred)`
  - Apply `np.clip(Position, 0, 2)`

### [Phase 2] Hybrid Defense (Future Upgrade)

Add this layer only after Phase 1 is stable and validated.

- **Model C (Market Regime):**
  - Target: `market_forward_excess_returns`.
  - Role: Detect market crash/bear regimes (Ignition Switch).
- **Advanced Logic:**
  - If `Model_C_Pred < Threshold` (e.g., 0 or negative), force `Position = 0`.
  - Otherwise, use the Phase 1 allocation logic.

## 6. Inference Rules

- Use `kaggle_evaluation` module loop.
- Use `lagged_` features provided by the API.
- Ensure the inference logic matches the Training Phase (Phase 1 vs Phase 2).
