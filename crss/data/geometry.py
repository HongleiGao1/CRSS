"""Locate prepared feature caches using their original model identities."""
from pathlib import Path

from crss.models.model_registry import legacy_base_allowed
from crss.utils.io import file_hash, object_hash


def model_identity(spec):
    return object_hash({**spec, 'local_sha256':file_hash(spec['checkpoint_path']) if spec['checkpoint_path'] else None})[:20]

def geometry_path(cfg, name):
    return Path(cfg['feature']['root']) / 'features' / cfg['dataset']['name'] / cfg.get('model_scale', 'base') / name / model_identity(cfg['models'][name]) / 'train_l2.npz'


def cached_geometry_path(cfg, name):
    """Only the original Base model identity may read the legacy family directory."""
    path = geometry_path(cfg, name)
    legacy = path.parents[2].parent / name / path.parent.name / path.name
    if not path.exists() and legacy_base_allowed(cfg, name) and legacy.is_file():
        return legacy
    return path
