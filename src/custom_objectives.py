
import numpy as np
from scipy.stats import spearmanr

def correlation_metric(y_true, y_pred):
    """
    Custom evaluation metric for LightGBM (sklearn API).
    Signature: (y_true, y_pred) -> (name, value, is_higher_better)
    """
    # Handle NaNs
    if len(y_pred) < 2 or len(y_true) < 2:
        return 'correlation', 0.0, True
        
    # Pearson Correlation
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    
    return 'correlation', corr, True

def spearman_metric(preds, train_data):
    """
    Custom evaluation metric for LightGBM: Spearman Rank Correlation.
    Robust to outliers.
    """
    actuals = train_data.get_label()
    
    if len(preds) < 2 or len(actuals) < 2:
        return 'spearman', 0.0, True
        
    corr, _ = spearmanr(preds, actuals)
    return 'spearman', corr, True

def asymmetric_mae_loss(preds, train_data):
    """
    Custom Objective Function for Asymmetric MAE.
    Penalizes under-prediction (y > pred) more heavily.
    
    Loss = mean(|y - pred| * weight)
    where weight = alpha if y > pred (under-estimation)
          weight = 1.0   if y <= pred (over-estimation)
          
    Gradient (dLoss/dPred):
        -alpha if y > pred
        +1.0   if y <= pred
        
    Hessian (d2Loss/dPred2):
        0 (constant gradient) -> approximated as small constant for stability
    """
    labels = train_data.get_label()
    alpha = 2.0 # Penalty weight for under-estimation
    
    residual = (labels - preds)
    grad = np.where(residual > 0, -alpha, 1.0)
    hess = np.ones_like(grad) # Constant hessian for MAE-like loss
    
    return grad, hess

def asymmetric_mae_metric(preds, train_data):
    """
    Metric corresponding to asymmetric loss for evaluation.
    """
    labels = train_data.get_label()
    alpha = 2.0
    
    residual = (labels - preds)
    loss = np.where(residual > 0, alpha * residual, -residual)
    
    return 'asymmetric_mae', np.mean(loss), False
