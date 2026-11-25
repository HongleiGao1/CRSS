"""Observed-label prototypes and prototype-induced expert predictions."""
import numpy as np

from crss.config import TAU


def prototype_induced_prediction(features: np.ndarray, observed_labels: np.ndarray,
                                 num_classes: int, tau: float = TAU) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return own-expert argmax, top-two cosine gap, and soft class prediction.

    The two similarity passes preserve the validated Stage I floating-point order.
    Only observed labels form prototypes; no GT label enters this function.
    """
    if not np.isfinite(tau) or tau <= 0:raise ValueError('Invalid tau')
    z=np.asarray(features);observed=np.asarray(observed_labels)
    if z.ndim!=2 or observed.shape!=(len(z),) or not np.issubdtype(observed.dtype,np.integer):
        raise ValueError('Invalid aligned frozen features/observed labels')
    if not np.isfinite(z).all() or np.any((observed<0)|(observed>=num_classes)):
        raise ValueError('Invalid features or observed labels')
    if np.any(np.bincount(observed,minlength=num_classes)==0):
        raise ValueError('Each observed class needs a prototype')
    sums=np.zeros((num_classes,z.shape[1]),dtype=np.float64)
    np.add.at(sums,observed,z)
    norm=np.linalg.norm(sums,axis=1)
    if np.any(norm==0):raise ValueError('Undefined observed-label prototype')
    prototypes=sums/norm[:,None]
    prediction=np.empty(len(z),dtype=np.int64)
    margin=np.empty(len(z))
    for start in range(0,len(z),4096):
        stop=min(start+4096,len(z))
        similarities=z[start:stop]@prototypes.T
        prediction[start:stop]=similarities.argmax(axis=1)
        top=np.partition(similarities,-2,axis=1)[:,-2:]
        margin[start:stop]=top.max(axis=1)-top.min(axis=1)
    # The established soft prediction evaluates the same prototypes in a second pass.
    sums=np.zeros((num_classes,z.shape[1]),dtype=np.float64)
    np.add.at(sums,observed,z)
    prototypes=sums/np.linalg.norm(sums,axis=1)[:,None]
    probabilities=np.empty((len(z),num_classes),dtype=np.float64)
    for start in range(0,len(z),4096):
        stop=min(start+4096,len(z))
        similarity=z[start:stop]@prototypes.T
        if not np.array_equal(similarity.argmax(1),prediction[start:stop]):
            raise AssertionError('Hard/soft prototype prediction mismatch')
        logits=similarity/tau
        logits-=logits.max(1,keepdims=True)
        exponential=np.exp(logits)
        probabilities[start:stop]=exponential/exponential.sum(1,keepdims=True)
    return prediction,margin,probabilities
