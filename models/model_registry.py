"""Five fixed families; scale selection leaves the original Base specs unchanged."""
from pathlib import Path

FAMILIES = ('vit', 'clip', 'dino', 'deit3', 'convnextv2')
SCALES = ('weak', 'base', 'strong')


def entry(display, architecture, model_name, provider=None):
    return dict(display_name=display, architecture=architecture,
                model_name=model_name, provider=provider)


MODEL_SCALE_REGISTRY = {
    'weak': {
        'vit': entry('ViT-B/32', 'vit_b32_imagenet', 'vit_base_patch32_224.augreg_in21k_ft_in1k'),
        'clip': entry('CLIP-ViT-B/32', 'clip_vit_b32', 'ViT-B-32 / openai'),
        'dino': entry('DINOv2-S/14', 'dinov2_vit_s14', 'vit_small_patch14_dinov2.lvd142m'),
        'deit3': entry('DeiTIII-S/16', 'deit3_small_patch16_224.fb_in22k_ft_in1k', 'deit3_small_patch16_224.fb_in22k_ft_in1k', 'timm'),
        'convnextv2': entry('ConvNeXtV2-Tiny', 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'timm'),
    },
    'base': {
        'vit': entry('ViT-B/16', 'vit_b16_imagenet', 'vit_base_patch16_224.augreg2_in21k_ft_in1k'),
        'clip': entry('CLIP-ViT-B/16', 'clip_vit_b16', 'ViT-B-16 / openai'),
        'dino': entry('DINOv2-B/14', 'dinov2_vit_b14', 'vit_base_patch14_dinov2.lvd142m'),
        'deit3': entry('DeiTIII-B/16', 'deit3_base_patch16_224.fb_in22k_ft_in1k', 'deit3_base_patch16_224.fb_in22k_ft_in1k', 'timm'),
        'convnextv2': entry('ConvNeXtV2-Base', 'convnextv2_base.fcmae_ft_in22k_in1k', 'convnextv2_base.fcmae_ft_in22k_in1k', 'timm'),
    },
    'strong': {
        'vit': entry('ViT-L/16', 'vit_l16_imagenet', 'vit_large_patch16_224.augreg_in21k_ft_in1k'),
        'clip': entry('CLIP-ViT-L/14', 'clip_vit_l14', 'ViT-L-14 / openai'),
        'dino': entry('DINOv2-L/14', 'dinov2_vit_l14', 'vit_large_patch14_dinov2.lvd142m'),
        'deit3': entry('DeiTIII-L/16', 'deit3_large_patch16_224.fb_in22k_ft_in1k', 'deit3_large_patch16_224.fb_in22k_ft_in1k', 'timm'),
        'convnextv2': entry('ConvNeXtV2-Large', 'convnextv2_large.fcmae_ft_in22k_in1k', 'convnextv2_large.fcmae_ft_in22k_in1k', 'timm'),
    },
}


def apply_model_scale(cfg, override=None):
    scale = override if override is not None else cfg.get('model_scale', 'base')
    if scale not in SCALES:
        raise ValueError(f'Invalid model_scale {scale!r}; choose {SCALES}')
    if set(cfg['models']) - set(FAMILIES):
        raise ValueError('Only ViT, CLIP, DINOv2, DeiTIII and ConvNeXtV2 are supported')
    cfg['model_scale'] = scale
    for family, spec in cfg['models'].items():
        version = MODEL_SCALE_REGISTRY[scale][family]
        # No descriptive keys are added to specs: Base cache identity stays identical.
        if spec['architecture'] not in {MODEL_SCALE_REGISTRY[s][family]['architecture'] for s in SCALES}:
            raise ValueError(f'{family}: architecture is outside the fixed three-scale registry')
        spec['architecture'] = version['architecture']
        if version['provider']:
            spec['provider'] = version['provider']
        else:
            spec.pop('provider', None)
    return cfg


def scale_output(root, scale):
    path = Path(root)
    if path.name in SCALES:
        if path.name != scale:
            raise ValueError(f'Output directory scale differs from request: {path}')
        return path
    return path / scale


def cache_version_metadata(cfg, family, metadata):
    version = MODEL_SCALE_REGISTRY[cfg.get('model_scale', 'base')][family]
    return dict(family=family, scale=cfg.get('model_scale', 'base'),
                display_name=version['display_name'], actual_model_name=version['model_name'],
                pretrained_checkpoint=metadata.get('checkpoint', cfg['models'][family].get('checkpoint_path') or version['model_name']),
                input_size=metadata.get('input_size', metadata.get('preprocessing', {}).get('input_size')),
                feature_dim=metadata['feature_dim'])


def legacy_base_allowed(cfg, family):
    spec = cfg['models'][family]; base = MODEL_SCALE_REGISTRY['base'][family]
    return (cfg.get('model_scale', 'base') == 'base' and spec['architecture'] == base['architecture']
            and spec.get('provider') == base['provider'])
