from pathlib import Path
import numpy as np
from utils.io import read_npz, save_npz, object_hash, file_hash
from utils.reference import activate

def noise_name(cfg):
    spec = cfg['noise']
    if spec['type'] == 'symmetric': return f"sym_{spec['rate']:g}"
    if spec['type'] == 'human': return 'human_' + spec['variant']
    return spec['type']

def noise_path(cfg):
    return Path(cfg['feature']['root']) / 'noise' / f"{cfg['dataset']['name']}_{noise_name(cfg)}_seed{cfg['seed']}.npz"

def validate_noise(payload, clean_labels=None):
    clean, observed = payload['clean_labels'], payload['observed_labels']
    if clean.ndim != 1 or observed.shape != clean.shape: raise ValueError('Invalid noise shapes')
    if not np.array_equal(payload['indices'], np.arange(len(clean))): raise ValueError('Noise order mismatch')
    if clean_labels is not None and not np.array_equal(clean, clean_labels): raise ValueError('Noise GT mismatch')
    if not np.array_equal(payload['noise_mask'], clean != observed): raise ValueError('Invalid noise mask')
    if not np.array_equal(payload['clean_mask'], clean == observed): raise ValueError('Invalid clean mask')
    if not np.array_equal(payload['noisy_labels'], observed): raise ValueError('Observed/noisy alias mismatch')
    if np.any((observed < 0) | (observed >= int(payload['num_classes']))): raise ValueError('Invalid observed labels')
    return payload

def get_noise(cfg, dataset=None):
    path = noise_path(cfg)
    source = None
    name = cfg['dataset']['name'].removesuffix('n')
    if cfg['noise']['type'] == 'human':
        source = Path(cfg['noise']['human_root'] or cfg['dataset']['root']) / 'noise_label_human' / ('CIFAR-100_human.pt' if name == 'cifar100' else 'CIFAR-10_human.pt')
        if name not in {'cifar10','cifar100'}: raise ValueError('Human CIFAR-N loader only supports CIFAR')
    identity = object_hash({'noise':cfg['noise'], 'dataset':cfg['dataset']['name'], 'seed':cfg['seed'],
                            'source_sha256':file_hash(source) if source and source.exists() else None})
    if path.exists():
        payload = validate_noise(read_npz(path), None if dataset is None else dataset.clean_labels)
        if str(payload['identity']) != identity: raise ValueError(f'Noise cache identity changed: {path}. Use a different cache root.')
        return payload
    if dataset is None: raise FileNotFoundError(f'Noise cache missing: {path}; run run_extract.py first.')
    clean = dataset.clean_labels
    kind = cfg['noise']['type']
    if kind in {'human','symmetric'}:
        activate()
        from lnl_foundation.data.noise import make_noisy_labels
        variant = cfg['noise']['variant']
        if name == 'cifar100':
            if kind == 'human' and variant != 'fine': raise ValueError('CIFAR-100N requires variant fine')
            human_type = 'human_noisy_label'
        else:
            human_type = 'human_' + ({'worse':'worse_label','aggre':'aggre_label'}.get(variant, variant))
        observed = make_noisy_labels(name, cfg['noise']['human_root'] or cfg['dataset']['root'],
                                     None, clean, dataset.num_classes, kind, cfg['noise']['rate'], cfg['seed'],
                                     human_noise_type=human_type)
    elif kind == 'none': observed = clean.copy()
    elif kind == 'observed':
        if dataset.observed_labels is None: raise ValueError('Manifest observed GT annotations required')
        observed = dataset.observed_labels
    else: raise ValueError(kind)
    payload = dict(clean_labels=clean, observed_labels=observed, noisy_labels=observed,
                   clean_mask=clean == observed, noise_mask=clean != observed, indices=dataset.indices,
                   num_classes=dataset.num_classes, class_names=np.asarray(dataset.class_names), identity=identity,
                   source=str(source) if source else ('FM-SRM symmetric transition' if kind == 'symmetric' else kind))
    if kind == 'human': payload['human_noisy_labels'] = observed
    validate_noise(payload, clean); save_npz(path, **payload)
    return payload
