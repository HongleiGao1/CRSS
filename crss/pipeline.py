"""Frozen CRSS Stage I pipeline, independent of experiment and GT modules."""
from collections.abc import Mapping
import numpy as np

from crss.config import CRSSConfig,EXPERTS
from crss.prototypes import prototype_induced_prediction
from crss.reliability import instance_reliability
from crss.consensus import uniform_soft_consensus
from crss.rvc import hard_rvc_delta
from crss.stratification import reliable_candidate_mask,stratify
from crss.types import CRSSResult


def compute_crss_core_evidence(features_by_expert: Mapping[str,np.ndarray], observed_labels: np.ndarray,
             sample_indices: np.ndarray, config: CRSSConfig) -> dict:
    """Run frozen Stage I; clean labels are neither accepted nor accessed."""
    if not isinstance(config,CRSSConfig) or tuple(features_by_expert)!=EXPERTS:
        raise ValueError('Require CRSSConfig and five ordered frozen expert features')
    observed=np.asarray(observed_labels);indices=np.asarray(sample_indices)
    n=len(observed);classes=config.num_classes
    if n<2 or indices.shape!=(n,) or not np.issubdtype(observed.dtype,np.integer) or \
       np.any((observed<0)|(observed>=classes)) or len(np.unique(indices))!=n:
        raise ValueError('Invalid observed labels/sample indices')
    hard=[];ic=[];soft=[]
    for expert in EXPERTS:
        features=np.asarray(features_by_expert[expert])
        if features.ndim!=2 or features.shape[0]!=n:
            raise ValueError('Frozen feature alignment mismatch: '+expert)
        prediction,margin,probability=prototype_induced_prediction(features,observed,classes,config.tau)
        hard.append(prediction);ic.append(margin);soft.append(probability)
    expert_pred=np.stack(hard,axis=1)
    intrinsic=np.stack(ic,axis=1)
    expert_prob=np.stack(soft,axis=1)
    ec,reliability=instance_reliability(intrinsic,expert_pred)
    q,usc_pred,usc_conf=uniform_soft_consensus(expert_prob,expert_pred)
    delta=hard_rvc_delta(expert_pred,reliability,usc_pred,observed,classes)
    return dict(indices=indices,observed_labels=observed,
        expert_predictions=expert_pred,expert_probabilities=expert_prob,
        intrinsic_confidence=intrinsic,extrinsic_consensus=ec,expert_reliability=reliability,
        usc_distribution=q,usc_prediction=usc_pred,usc_confidence=usc_conf,rvc_delta=delta)


def apply_crss_partition(core: dict, rho: float) -> CRSSResult:
    """Only routing depends on rho. Hard RVC evidence is valid for all rows."""
    candidate=reliable_candidate_mask(core['usc_confidence'],core['indices'],rho)
    rc,cn,u,group,reason=stratify(candidate,core['observed_labels'],core['usc_prediction'],core['rvc_delta'])
    return CRSSResult(**core,reliable_candidate_mask=candidate,rc_mask=rc,cn_mask=cn,
                      u_mask=u,group_id=group,u_reason=reason)


def run_crss(features_by_expert, observed_labels, sample_indices, config):
    return apply_crss_partition(compute_crss_core_evidence(features_by_expert,observed_labels,
                               sample_indices,config),config.rho)
