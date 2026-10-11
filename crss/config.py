"""Frozen CRSS method definition and task identity."""
from dataclasses import dataclass
from enum import IntEnum
import math

METHOD_NAME = 'CRSS'
FULL_NAME = 'Cross-Foundation Reliability-Guided Supervision Stratification'
METHOD_VERSION = '1.0.0'
EXPERTS = ('vit', 'clip', 'dino', 'deit3', 'convnextv2')
TAU = .1
RHO = .6


class SupervisionGroup(IntEnum):
    RC = 0
    CN = 1
    U = 2


@dataclass(frozen=True)
class CRSSConfig:
    dataset: str
    model_scale: str = 'base'
    noise_setting: str = 'human'
    seed: int = 1
    feature_cache_path: str = 'cache'
    tau: float = TAU
    rho: float = RHO

    def __post_init__(self):
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('tau must be finite and positive')
        if not math.isfinite(self.rho) or not 0 < self.rho <= 1:
            raise ValueError('rho must be finite and in (0, 1]')
        if self.dataset not in ('cifar10', 'cifar100'):
            raise ValueError('CRSS supports CIFAR-10/CIFAR-100')
        if self.model_scale not in ('weak', 'base', 'strong'):
            raise ValueError('CRSS supports weak/base/strong expert scales')
        if self.noise_setting not in ('human', 'sym', 'pair', 'idn'):
            raise ValueError('Unsupported observed-label setting')
        if not isinstance(self.seed, int) or self.seed not in (1,2,3,4,5):
            raise ValueError('Invalid task seed')
        if self.noise_setting == 'human' and self.seed != 1:
            raise ValueError('Human annotations have one fixed task (seed 1)')

    @property
    def num_classes(self) -> int:
        return 10 if self.dataset == 'cifar10' else 100
