"""Uniform Soft Consensus keeps all five prototype predictions equally weighted."""
import numpy as np


def uniform_soft_consensus(expert_probabilities: np.ndarray,
                           expert_predictions: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    p=np.asarray(expert_probabilities)
    if p.ndim!=3 or p.shape[1]!=5 or p.shape[:2]!=expert_predictions.shape or \
       not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(2),1):
        raise ValueError('Require five aligned expert soft class predictions')
    if not np.array_equal(p.argmax(2),expert_predictions):
        raise ValueError('Expert own argmax and soft prediction differ')
    q=p.mean(1)
    return q,q.argmax(1),q.max(1)
