"""Instance-wise expert reliability: IC/EC average-tie percentile ranks."""
import numpy as np
from scipy.stats import rankdata


def average_tie_percentile(values: np.ndarray) -> np.ndarray:
    values=np.asarray(values)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Rank normalization requires a finite vector of >=2 samples')
    return (rankdata(values,method='average')-1)/(len(values)-1)


def extrinsic_consensus(expert_predictions: np.ndarray) -> np.ndarray:
    predictions=np.asarray(expert_predictions)
    if predictions.ndim!=2 or predictions.shape[1]!=5:
        raise ValueError('CRSS requires five expert hard predictions')
    return ((predictions[:,:,None]==predictions[:,None,:]).sum(axis=2)-1)/4


def instance_reliability(intrinsic_confidence: np.ndarray,
                         expert_predictions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    ic=np.asarray(intrinsic_confidence)
    if ic.shape!=expert_predictions.shape or ic.ndim!=2 or ic.shape[1]!=5:
        raise ValueError('IC and expert predictions must be aligned N x 5 matrices')
    ec=extrinsic_consensus(expert_predictions)
    ic_rank=np.column_stack([average_tie_percentile(ic[:,j]) for j in range(5)])
    ec_rank=np.column_stack([average_tie_percentile(ec[:,j]) for j in range(5)])
    reliability=(ic_rank+ec_rank)/2
    return ec,reliability
