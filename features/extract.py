import json
import os
from pathlib import Path
import numpy as np
from utils.reference import activate
from utils.io import array_hash, file_hash, read_npz, save_npz
from utils.config import enabled_models
from datasets.build_dataset import build_dataset
from datasets.noise import get_noise
from models.model_registry import cache_version_metadata
from .cache import cached_geometry_path, geometry_path, model_identity, load_cached, bind_noise, validate_features

def run_extract(cfg, logger, split='train'):
    if split == 'test':
        return run_extract_test(cfg, logger)
    if split != 'train':raise ValueError('split must be train or test')
    activate()
    from lnl_foundation.utils import get_device, set_seed
    from lnl_foundation.features import extract_features
    from models.build_model import build_model
    if cfg['hf_endpoint']: os.environ['HF_ENDPOINT'] = cfg['hf_endpoint'].rstrip('/')
    set_seed(cfg['seed'])
    dataset = build_dataset(cfg)
    noise = get_noise(cfg, dataset)  # Exactly one shared generation/load, outside model loop.
    for name in enabled_models(cfg):
        path = geometry_path(cfg,name)
        if cached_geometry_path(cfg, name).exists() and not cfg['force_extract']:
            load_cached(cfg,name,noise)
            logger.info('Reused %s: %s', name, cached_geometry_path(cfg, name)); continue
        set_seed(cfg['seed'])
        backbone = build_model(cfg['models'][name])
        dataset.transform = backbone.preprocess
        result = extract_features(backbone, dataset, cfg['dataset']['batch_size'], cfg['dataset']['num_workers'], get_device(cfg['device']))
        features = result['features'].numpy()
        metadata = dict(model_identity=model_identity(cfg['models'][name]), dataset_root=cfg['dataset']['root'],
                        manifest=cfg['dataset']['manifest'], manifest_sha256=file_hash(cfg['dataset']['manifest']) if cfg['dataset']['manifest'] else None,
                        checkpoint=result['checkpoint'], preprocessing=result['preprocess_config'], preprocessing_source=result['preprocess_source'],
                        feature_dim=features.shape[1], frozen=not any(p.requires_grad for p in backbone.parameters()),
                        pretrained=cfg['models'][name]['pretrained'], seed=cfg['seed'])
        metadata.update(cache_version_metadata(cfg, name, metadata))
        payload = dict(features=features, indices=result['indices'].numpy(), clean_labels=result['clean_labels'].numpy(),
                       feature_sha256=array_hash(features), metadata=json.dumps(metadata))
        validate_features(payload,noise); save_npz(path,**payload); bind_noise(cfg,name,payload,noise)
        logger.info('Saved %s: %s %s', name, features.shape, path)
        del backbone, result, features, payload


def run_extract_test(cfg, logger):
    """Extract CIFAR test features with frozen train-cache provenance."""
    if cfg['dataset']['name'] not in ('cifar10','cifar100'):
        raise ValueError('Test extraction currently supports CIFAR-10/CIFAR-100 only')
    activate()
    from lnl_foundation.utils import get_device, set_seed
    from lnl_foundation.data.datasets import load_cifar_base
    from lnl_foundation.features import extract_features
    from datasets.build_dataset import IndexedDataset
    from models.build_model import build_model
    if cfg['hf_endpoint']:os.environ['HF_ENDPOINT']=cfg['hf_endpoint'].rstrip('/')
    set_seed(cfg['seed'])
    base=load_cifar_base(cfg['dataset']['name'],cfg['dataset']['root'],train=False,transform=None,download=cfg['dataset']['download'])
    dataset=IndexedDataset(base,base.targets,base.classes)
    for name in enabled_models(cfg):
        train_path=cached_geometry_path(cfg,name)
        if not train_path.is_file():
            raise FileNotFoundError(f'Frozen training feature cache is required before test extraction: {train_path}')
        train=validate_features(read_npz(train_path))
        train_meta=json.loads(str(train['metadata']))
        if train_meta.get('model_identity')!=model_identity(cfg['models'][name]) or train_meta.get('scale','base')!=cfg['model_scale']:
            raise ValueError(f'Train cache model identity/scale mismatch: {train_path}')
        path=train_path.with_name('test_l2.npz')
        if path.is_file() and not cfg['force_extract']:
            cached=validate_features(read_npz(path))
            meta=json.loads(str(cached['metadata']))
            if (meta.get('source_train_sha256')!=str(train['feature_sha256']) or
                meta.get('model_identity')!=train_meta.get('model_identity') or
                meta.get('checkpoint')!=train_meta.get('checkpoint') or
                meta.get('preprocessing')!=train_meta.get('preprocessing') or
                meta.get('preprocessing_source')!=train_meta.get('preprocessing_source') or
                len(cached['features'])!=10000 or not np.array_equal(cached['clean_labels'],np.asarray(base.targets))):
                raise ValueError(f'Stale or mismatched test feature cache: {path}')
            logger.info('Reused test features %s: %s',name,path)
            continue
        set_seed(cfg['seed'])
        backbone=build_model(cfg['models'][name])
        dataset.transform=backbone.preprocess
        result=extract_features(backbone,dataset,cfg['dataset']['batch_size'],cfg['dataset']['num_workers'],get_device(cfg['device']))
        features=result['features'].numpy()
        preprocessing=json.loads(json.dumps(result['preprocess_config']))
        if (features.shape[1]!=train['features'].shape[1] or
            preprocessing!=train_meta['preprocessing'] or
            result['preprocess_source']!=train_meta['preprocessing_source'] or
            result['checkpoint']!=train_meta['checkpoint']):
            raise ValueError(f'Test backbone/preprocessing differs from frozen train cache: {name}')
        metadata=dict(train_meta,source_train_sha256=str(train['feature_sha256']),split='test',
                      preprocessing=preprocessing,checkpoint=result['checkpoint'])
        payload=dict(features=features,indices=result['indices'].numpy(),clean_labels=result['clean_labels'].numpy(),
                     feature_sha256=array_hash(features),metadata=json.dumps(metadata))
        validate_features(payload)
        if len(features)!=10000:raise ValueError('Expected 10000 CIFAR test samples')
        save_npz(path,**payload)
        logger.info('Saved test features %s: %s %s',name,features.shape,path)
        del backbone,result,features,payload
