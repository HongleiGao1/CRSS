"""Official timm global features, with no random-weight fallback."""
from crss.utils.reference import activate
activate()
from torch import nn
from lnl_foundation.backbones.frozen import FrozenBackbone, BackboneSpec


class FrozenTimmCandidate(FrozenBackbone):
    def __init__(self, checkpoint, pretrained=True):
        import timm
        if pretrained is not True:
            raise ValueError('Candidate screening requires pretrained=True')
        if checkpoint not in timm.list_models(pretrained=True):
            raise ValueError(f'No registered pretrained timm model: {checkpoint}')
        nn.Module.__init__(self)
        self.requested_name = self.model_name = checkpoint
        self.spec = BackboneSpec('timm', checkpoint, checkpoint, 'Official timm pretrained weights')
        # Any download or state-dict loading exception propagates to the caller.
        self._load_timm(pretrained=True)
        if self.model.num_classes != 0:
            raise ValueError('Classifier must be removed for global features')
        self.pretrained_load_success = True
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()
        self._print_summary()
