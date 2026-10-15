from pathlib import Path
import json
import numpy as np
from crss.utils.io import array_hash, object_hash, read_npz, save_npz, file_hash

def model_identity(spec):
    return object_hash({**spec, 'local_sha256':file_hash(spec['checkpoint_path']) if spec['checkpoint_path'] else None})[:20]

def geometry_path(cfg, name):
    return Path(cfg['feature']['root']) / 'features' / cfg['dataset']['name'] / cfg.get('model_scale', 'base') / name / model_identity(cfg['models'][name]) / 'train_l2.npz'


def cached_geometry_path(cfg, name):
    """Only the original Base model identity may read the legacy family directory."""
    from crss.models.model_registry import legacy_base_allowed
    path = geometry_path(cfg, name)
    legacy = path.parents[2].parent / name / path.parent.name / path.name
    if not path.exists() and legacy_base_allowed(cfg, name) and legacy.is_file():
        return legacy
    return path

def binding_path(cfg, name, feature_sha256):
    from crss.data.noise import noise_name
    return cached_geometry_path(cfg, name).parent / f"{noise_name(cfg)}_seed{cfg['seed']}_{str(feature_sha256)[:16]}.npz"

def validate_features(payload, expected=None):
    features = payload['features']
    n = len(payload['indices'])
    if features.ndim != 2 or len(features) != n or not np.isfinite(features).all(): raise ValueError('Invalid feature matrix')
    if not np.array_equal(payload['indices'], np.arange(n)): raise ValueError('Feature indices out of order')
    if payload['clean_labels'].shape != (n,): raise ValueError('Invalid clean label vector')
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=2e-4): raise ValueError('Features are not L2 normalized (or contain zero vectors)')
    if str(payload['feature_sha256']) != array_hash(features): raise ValueError('Feature cache checksum mismatch')
    if expected is not None:
        for key in ['indices','clean_labels']:
            if not np.array_equal(payload[key], expected[key]): raise AssertionError(f'Cross-model {key} mismatch')
        if 'observed_labels' in payload and not np.array_equal(payload['observed_labels'], expected['observed_labels']):
            raise AssertionError('Cross-model observed_labels mismatch')
    return payload

def bind_noise(cfg, name, geometry, noise):
    validate_features(geometry, noise)
    path = binding_path(cfg, name, geometry['feature_sha256'])
    # A small annotation sidecar avoids copying 50k x d clean features across noises.
    payload = dict(indices=noise['indices'], clean_labels=noise['clean_labels'],
                   observed_labels=noise['observed_labels'], noisy_labels=noise['observed_labels'],
                   feature_sha256=geometry['feature_sha256'], noise_identity=noise['identity'])
    if path.exists():
        old = read_npz(path)
        if any(not np.array_equal(old[k], v) for k,v in payload.items()): raise ValueError(f'Stale annotation sidecar: {path}')
    else: save_npz(path, **payload)

def load_cached(cfg, name, noise, require_binding=True):
    path = cached_geometry_path(cfg,name)
    if not path.exists(): raise FileNotFoundError(f'Feature cache missing: {path}. Run run_extract.py; analysis never invokes a backbone.')
    payload = validate_features(read_npz(path), noise)
    meta = json.loads(str(payload['metadata']))
    scale = cfg.get('model_scale', 'base')
    if meta.get('scale', 'base') != scale: raise ValueError('Feature cache scale mismatch')
    if scale != 'base' and 'scale' not in meta: raise ValueError('Non-Base cache lacks scale metadata')
    from crss.models.model_registry import MODEL_SCALE_REGISTRY
    version = MODEL_SCALE_REGISTRY[scale][name]
    for key, expected in [('family', name), ('display_name', version['display_name']), ('actual_model_name', version['model_name'])]:
        if key in meta and meta[key] != expected: raise ValueError(f'Feature cache {key} mismatch')
    if not meta.get('frozen', False): raise ValueError('Cache does not identify a completely frozen backbone')
    if meta.get('pretrained') != cfg['models'][name]['pretrained']: raise ValueError('Cache pretraining flag mismatch')
    if not meta['pretrained'] and meta.get('seed') != cfg['seed']: raise ValueError('Untrained feature cache seed mismatch')
    if meta['model_identity'] != model_identity(cfg['models'][name]): raise ValueError('Model identity mismatch')
    if meta['dataset_root'] != cfg['dataset']['root'] or meta.get('manifest') != cfg['dataset']['manifest']: raise ValueError('Dataset source changed')
    if cfg['dataset']['manifest'] and meta.get('manifest_sha256') != file_hash(cfg['dataset']['manifest']): raise ValueError('Dataset manifest changed')
    if require_binding:
        bind_noise(cfg, name, payload, noise)
        annotation = read_npz(binding_path(cfg,name,payload['feature_sha256']))
        payload.update(observed_labels=annotation['observed_labels'], noisy_labels=annotation['noisy_labels'])
        validate_features(payload, noise)
    return payload
