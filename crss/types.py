"""Stable Stage I result and Stage II consumption contract."""
from dataclasses import dataclass
import numpy as np

from crss.config import SupervisionGroup


@dataclass(frozen=True)
class Stage2SupervisionBundle:
    indices: np.ndarray
    observed_labels: np.ndarray
    group_id: np.ndarray
    rc_mask: np.ndarray
    cn_mask: np.ndarray
    u_mask: np.ndarray
    usc_distribution: np.ndarray
    usc_prediction: np.ndarray
    usc_confidence: np.ndarray
    expert_predictions: np.ndarray
    expert_probabilities: np.ndarray
    expert_reliability: np.ndarray
    rvc_delta: np.ndarray

    @property
    def rc_hard_targets(self) -> np.ndarray:
        return self.observed_labels[self.rc_mask]

    @property
    def cn_hard_targets(self) -> np.ndarray:
        """The formal Stage II uses hard USC argmax labels, never soft targets."""
        return self.usc_prediction[self.cn_mask]

    @property
    def u_auxiliary_evidence(self) -> dict[str, np.ndarray]:
        mask=self.u_mask
        return dict(indices=self.indices[mask],observed_labels=self.observed_labels[mask],
            usc_distribution=self.usc_distribution[mask],usc_prediction=self.usc_prediction[mask],
            usc_confidence=self.usc_confidence[mask],expert_predictions=self.expert_predictions[mask],
            expert_probabilities=self.expert_probabilities[mask],expert_reliability=self.expert_reliability[mask],
            rvc_delta=self.rvc_delta[mask])


@dataclass(frozen=True)
class CRSSResult:
    indices: np.ndarray
    observed_labels: np.ndarray
    expert_predictions: np.ndarray
    expert_probabilities: np.ndarray
    intrinsic_confidence: np.ndarray
    extrinsic_consensus: np.ndarray
    expert_reliability: np.ndarray
    usc_distribution: np.ndarray
    usc_prediction: np.ndarray
    usc_confidence: np.ndarray
    reliable_candidate_mask: np.ndarray
    rvc_delta: np.ndarray
    rc_mask: np.ndarray
    cn_mask: np.ndarray
    u_mask: np.ndarray
    group_id: np.ndarray
    u_reason: np.ndarray

    def to_stage2_bundle(self) -> Stage2SupervisionBundle:
        if not np.array_equal(self.rc_mask,self.group_id==SupervisionGroup.RC) or \
           not np.array_equal(self.cn_mask,self.group_id==SupervisionGroup.CN) or \
           not np.array_equal(self.u_mask,self.group_id==SupervisionGroup.U):
            raise ValueError('Corrupt CRSS supervision groups')
        return Stage2SupervisionBundle(
            indices=self.indices,observed_labels=self.observed_labels,group_id=self.group_id,
            rc_mask=self.rc_mask,cn_mask=self.cn_mask,u_mask=self.u_mask,
            usc_distribution=self.usc_distribution,usc_prediction=self.usc_prediction,
            usc_confidence=self.usc_confidence,expert_predictions=self.expert_predictions,
            expert_probabilities=self.expert_probabilities,expert_reliability=self.expert_reliability,
            rvc_delta=self.rvc_delta)
