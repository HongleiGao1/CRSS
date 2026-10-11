"""Validated cached features and immutable observed-label inputs.

These helpers never instantiate an encoder or generate noisy labels.
"""
import json
from pathlib import Path
import numpy as np
from crss.config import EXPERTS
from features.cache import cached_geometry_path, model_identity
from models.model_registry import MODEL_SCALE_REGISTRY
from utils.config import load_config as load_feature_config
from utils.io import array_hash, read_npz

ROOT = Path(__file__).resolve().parents[1]

def absolute(value):
    value=Path(value).expanduser()
    return (value if value.is_absolute() else ROOT/value).resolve()

def feature_group(dataset,scale,args):
    cfg=load_feature_config(['--config',str(absolute(args.config)),'--dataset',dataset,
        '--model-scale',scale,'--data-root',str(absolute(args.data_root)),
        '--noise-type','none','--seed','1','--set','feature.root='+str(absolute(args.cache_root)),
        '--models',*EXPERTS])
    features={};hashes={};paths={};indices=None
    for expert in EXPERTS:
        path=cached_geometry_path(cfg,expert)
        if not path.is_file():raise FileNotFoundError('Frozen feature cache missing: '+str(path))
        payload=read_npz(path)
        matrix=payload['features'];sample_ids=payload['indices']
        if matrix.ndim!=2 or matrix.shape[0]!=len(sample_ids) or not np.isfinite(matrix).all() or \
           not np.array_equal(sample_ids,np.arange(len(sample_ids))) or \
           not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-4) or \
           str(payload['feature_sha256'])!=array_hash(matrix):
            raise ValueError('Invalid frozen feature geometry/checksum: '+str(path))
        metadata=json.loads(str(payload['metadata']))
        spec=MODEL_SCALE_REGISTRY[scale][expert]
        if metadata.get('model_identity')!=model_identity(cfg['models'][expert]) or \
           metadata.get('scale','base')!=scale or metadata.get('frozen') is not True or \
           metadata.get('pretrained') is not True or \
           metadata.get('actual_model_name',spec['model_name'])!=spec['model_name']:
            raise ValueError('Frozen expert feature metadata mismatch: '+str(path))
        if indices is None:indices=payload['indices']
        elif not np.array_equal(indices,payload['indices']):
            raise ValueError('Expert feature sample indices differ')
        features[expert]=payload['features']
        hashes[expert]=str(payload['feature_sha256'])
        paths[expert]=str(path)
    return features,indices,hashes,paths

def observed_label_cache(dataset,noise,seed,args):
    if args.observed_labels:
        path=absolute(args.observed_labels)
    else:
        directory=absolute(args.noise_cache_root)/dataset
        matches=sorted(directory.glob(f'{noise}_seed{seed}_*.npz'))
        if len(matches)!=1:
            raise FileNotFoundError(f'Expected exactly one immutable observed-label cache for {dataset}/{noise}/seed{seed} '
                                    f'in {directory}; found {len(matches)}. Use --observed-labels explicitly.')
        path=matches[0]
    with np.load(path,allow_pickle=False) as archive:
        if 'indices' not in archive or 'observed_labels' not in archive:
            raise ValueError('Observed-label cache needs indices and observed_labels')
        indices=archive['indices'];observed=archive['observed_labels']
    audit=path.with_suffix('.json')
    if audit.is_file():
        metadata=json.loads(audit.read_text(encoding='utf-8'))
        if metadata.get('dataset')!=dataset or metadata.get('noise_type')!=noise or \
           int(metadata.get('seed',-1))!=seed or metadata.get('noisy_label_sha')!=array_hash(observed):
            raise ValueError('Observed-label cache task/provenance mismatch: '+str(path))
    return indices,observed,path

def _feature_path(meta, family, dataset, scale, feature_root):
    original = Path(meta['feature_paths'][family])
    if original.is_file(): return original
    candidates = list((feature_root/'features'/dataset/scale/family).glob('*/train_l2.npz'))
    if scale=='base': candidates += list((feature_root/'features'/dataset/family).glob('*/train_l2.npz'))
    matched=[]
    for path in candidates:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['feature_sha256']) == meta['feature_source_hashes'][family]: matched.append(path)
    if len(matched)!=1: raise FileNotFoundError(f'Expected one frozen {family} feature matching CRSS hash under {feature_root}; found {len(matched)}')
    return matched[0]
