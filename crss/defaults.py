"""Validated feature presets, embedded so a fresh clone needs no YAML files."""
from copy import deepcopy

FEATURE_DEFAULTS = {'model_scale': 'base',
 'seed': 1,
 'device': 'auto',
 'hf_endpoint': '',
 'dataset': {'name': 'cifar100',
             'root': '../DATASETS/',
             'manifest': None,
             'download': False,
             'batch_size': 256,
             'num_workers': 8},
 'noise': {'type': 'none', 'rate': 0.6, 'variant': 'fine', 'human_root': None},
 'models': {'vit': {'enabled': True,
                    'architecture': 'vit_b16_imagenet',
                    'pretrained': True,
                    'checkpoint_path': None},
            'clip': {'enabled': True,
                     'architecture': 'clip_vit_b16',
                     'pretrained': True,
                     'checkpoint_path': None},
            'dino': {'enabled': True,
                     'architecture': 'dinov2_vit_b14',
                     'pretrained': True,
                     'checkpoint_path': None},
            'deit3': {'provider': 'timm',
                      'architecture': 'deit3_base_patch16_224.fb_in22k_ft_in1k',
                      'pretrained': True,
                      'checkpoint_path': None},
            'convnextv2': {'provider': 'timm',
                           'architecture': 'convnextv2_base.fcmae_ft_in22k_in1k',
                           'pretrained': True,
                           'checkpoint_path': None}},
 'feature': {'normalize': True, 'cache': True, 'root': './cache'},
 'output': {'root': 'outputs/feature_extraction'}}


def feature_defaults(preset='default'):
    cfg = deepcopy(FEATURE_DEFAULTS)
    if preset == 'server':
        cfg['dataset'].update(batch_size=32, num_workers=4)
    elif preset != 'default':
        raise ValueError('Unknown built-in feature preset: ' + str(preset))
    return cfg
