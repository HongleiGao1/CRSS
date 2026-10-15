from crss.utils.reference import activate

def build_model(spec):
    activate()
    from lnl_foundation.backbones.frozen import FrozenBackbone, BackboneSpec
    import torch
    if spec.get('provider') == 'timm':
        from .timm_candidate import FrozenTimmCandidate
        model = FrozenTimmCandidate(spec['architecture'], spec['pretrained'])
    else:
        model = FrozenBackbone(spec['architecture'], pretrained=spec['pretrained'] and not spec['checkpoint_path'])
    if spec['checkpoint_path']:
        payload = torch.load(spec['checkpoint_path'], map_location='cpu', weights_only=True)
        state = payload.get('state_dict', payload.get('model', payload))
        model.model.load_state_dict(state, strict=True)
        model.spec = BackboneSpec(model.spec.provider, model.spec.architecture, spec['checkpoint_path'], model.spec.pretraining)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()): raise AssertionError('Backbone must be fully frozen')
    return model
