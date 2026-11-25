"""USC top-rho candidates and the frozen RC/CN/U stratification rule."""
import numpy as np

from crss.config import RHO,SupervisionGroup


def reliable_candidate_mask(usc_confidence: np.ndarray, indices: np.ndarray, rho: float = RHO) -> np.ndarray:
    if not np.isfinite(rho) or not 0 < rho <= 1:raise ValueError('Invalid rho')
    q=np.asarray(usc_confidence);idx=np.asarray(indices)
    n=len(q)
    if n<2 or idx.shape!=(n,) or not np.isfinite(q).all() or len(np.unique(idx))!=n:
        raise ValueError('Invalid USC confidence/sample indices')
    order=np.lexsort((idx,-q))
    mask=np.zeros(n,dtype=bool)
    mask[order[:int(np.floor(rho*n))]]=True
    return mask


def stratify(candidate: np.ndarray, observed: np.ndarray, usc_prediction: np.ndarray,
             rvc_delta: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    n=len(candidate)
    if any(np.asarray(v).shape!=(n,) for v in (observed,usc_prediction,rvc_delta)):
        raise ValueError('Partition inputs are misaligned')
    rc=candidate&(usc_prediction==observed)
    cn=candidate&(usc_prediction!=observed)&(rvc_delta>0)
    u=~(rc|cn)
    group=np.full(n,SupervisionGroup.U,dtype=np.int8)
    group[rc]=SupervisionGroup.RC;group[cn]=SupervisionGroup.CN
    reason=np.full(n,'',dtype='<U16')
    reason[~candidate]='LOW_CONSENSUS'
    reason[candidate&~(rc|cn)]='RVC_REJECTED'
    if np.any(rc&cn) or not np.all(rc|cn|u) or not np.array_equal(u,group==SupervisionGroup.U):
        raise AssertionError('CRSS RC/CN/U is not a complete disjoint partition')
    return rc,cn,u,group,reason
