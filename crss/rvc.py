"""Version B HARD reliability-supported evidence for disputed corrections."""
import numpy as np

EPS=1e-12


def hard_rvc_delta(expert_predictions: np.ndarray, reliability: np.ndarray,
                   usc_prediction: np.ndarray, observed_labels: np.ndarray,
                   num_classes: int) -> np.ndarray:
    predictions=np.asarray(expert_predictions);r=np.asarray(reliability,dtype=np.float64)
    n,m=predictions.shape
    if m!=5 or r.shape!=(n,m) or np.any(r<0) or not np.isfinite(r).all():
        raise ValueError('Require five aligned nonnegative reliability scores')
    if np.any((predictions<0)|(predictions>=num_classes)):
        raise ValueError('Invalid expert class prediction')
    totals=r.sum(1,keepdims=True)
    weights=np.divide(r,totals+EPS,out=np.full_like(r,1/m),where=totals>EPS)
    weights/=weights.sum(1,keepdims=True)
    evidence=np.zeros((n,num_classes),dtype=np.float64)
    np.add.at(evidence,(np.repeat(np.arange(n),m),predictions.ravel()),weights.ravel())
    rows=np.arange(n)
    return evidence[rows,usc_prediction]-evidence[rows,observed_labels]
