# Registered checkpoints

These identifiers are taken directly from the current scale registry and
loaders, not substituted with similarly named checkpoints.

| Scale | Family | Actual checkpoint / provider identifier |
| --- | --- | --- |
| weak | vit | `vit_base_patch32_224.augreg_in21k_ft_in1k` |
| weak | clip | `ViT-B-32 / openai` |
| weak | dino | `vit_small_patch14_dinov2.lvd142m` |
| weak | deit3 | `deit3_small_patch16_224.fb_in22k_ft_in1k` |
| weak | convnextv2 | `convnextv2_tiny.fcmae_ft_in22k_in1k` |
| base | vit | `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| base | clip | `ViT-B-16 / openai` |
| base | dino | `vit_base_patch14_dinov2.lvd142m` |
| base | deit3 | `deit3_base_patch16_224.fb_in22k_ft_in1k` |
| base | convnextv2 | `convnextv2_base.fcmae_ft_in22k_in1k` |
| strong | vit | `vit_large_patch16_224.augreg_in21k_ft_in1k` |
| strong | clip | `ViT-L-14 / openai` |
| strong | dino | `vit_large_patch14_dinov2.lvd142m` |
| strong | deit3 | `deit3_large_patch16_224.fb_in22k_ft_in1k` |
| strong | convnextv2 | `convnextv2_large.fcmae_ft_in22k_in1k` |

CLIP uses open_clip with the OpenAI checkpoint/preprocessing. All other
families use timm pretrained configurations and `num_classes=0`. The frozen
adapter checks that each encoder returns a global `[batch, feature_dim]`
matrix; extraction L2-normalizes these features.

The checkpoint name `fcmae` is part of the ConvNeXtV2 checkpoint identifier.
It does not add a sixth expert family or a separate masked-autoencoder loader.
Only the five registered expert families are available.

For local weights, set `models.<family>.checkpoint_path` to a repository-relative
or absolute path in a copied YAML configuration. Paths and checkpoint hashes
are part of the cache identity. A newly configured local checkpoint path does
not silently reuse an old identity even when its architecture matches.

Feature caches record the actual loader checkpoint, preprocessing source and
configuration, input size, feature dimension, family, scale and hashes. Test
extraction checks the recorded train checkpoint and preprocessing before saving.
