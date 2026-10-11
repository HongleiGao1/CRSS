from pathlib import Path
import argparse
import copy
import yaml
from models.model_registry import apply_model_scale, SCALES

ROOT = Path(__file__).resolve().parents[1]

def load_config(argv=None):
    parser = argparse.ArgumentParser(description='Frozen feature extraction and cache configuration', epilog='run_extract.py also accepts --split train or test.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    parser.add_argument('--models', nargs='+', choices=('vit','clip','dino','deit3','convnextv2'))
    parser.add_argument('--model-scale', choices=SCALES)
    for flag in ['dataset', 'data-root', 'noise-type', 'human-variant', 'device', 'output-root']:
        parser.add_argument('--' + flag)
    parser.add_argument('--noise', dest='noise_type')
    parser.add_argument('--noise-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--force-extract', action='store_true')
    args = parser.parse_args(argv)
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('model_scale', 'base')
    for override in args.set:
        key, value = override.split('=', 1)
        parts, node = key.split('.'), cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError('Unknown config key: ' + key)
            node = node[part]
        if parts[-1] not in node:
            raise ValueError('Unknown config key: ' + key)
        node[parts[-1]] = yaml.safe_load(value)
    mapping = {'dataset':'dataset.name', 'data_root':'dataset.root', 'noise_type':'noise.type',
               'noise_rate':'noise.rate', 'human_variant':'noise.variant', 'device':'device',
               'output_root':'output.root', 'seed':'seed'}
    for arg, key in mapping.items():
        value = getattr(args, arg)
        if value is not None:
            parts = key.split('.')
            if len(parts) == 1: cfg[key] = value
            else: cfg[parts[0]][parts[1]] = value
    apply_model_scale(cfg, args.model_scale)
    if args.models:
        unknown = set(args.models) - cfg['models'].keys()
        if unknown: raise ValueError(f'Unknown models: {unknown}')
        for name, spec in cfg['models'].items():
            if 'enabled' in spec or name not in args.models:
                spec['enabled'] = name in args.models
    for section, key in [('dataset','root'), ('dataset','manifest'), ('noise','human_root'),
                         ('feature','root'), ('output','root')]:
        value = cfg[section][key]
        if value:
            p = Path(value).expanduser()
            cfg[section][key] = str((p if p.is_absolute() else ROOT / p).resolve())
    for spec in cfg['models'].values():
        if spec['checkpoint_path']:
            p = Path(spec['checkpoint_path']).expanduser()
            spec['checkpoint_path'] = str((p if p.is_absolute() else ROOT / p).resolve())
    if cfg['dataset']['name'] not in ('cifar10','cifar100'):
        raise ValueError('The released protocol supports CIFAR-10 and CIFAR-100')
    if not enabled_models(cfg): raise ValueError('At least one feature extractor must be enabled')
    if cfg['dataset']['batch_size'] < 1 or cfg['dataset']['num_workers'] < 0: raise ValueError('Invalid loader batch size or worker count')
    if cfg['noise']['type'] not in {'symmetric','human','none','observed'}: raise ValueError('Unsupported noise type')
    if not 0 <= cfg['noise']['rate'] <= 1: raise ValueError('Noise rate must be in [0,1]')
    if not cfg['feature']['normalize'] or not cfg['feature']['cache']: raise ValueError('This experiment requires normalized feature caches')
    cfg['force_extract'] = args.force_extract
    return cfg

def enabled_models(cfg):
    return [name for name, spec in cfg['models'].items() if spec.get('enabled', True)]
