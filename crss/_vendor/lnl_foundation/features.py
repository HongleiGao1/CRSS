"""Extract normalized features in dataset order."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(backbone, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone.to(device).eval()
    all_features, labels, indices = [], [], []
    for batch in tqdm(loader, desc="Extracting features"):
        images, clean = batch[:2]
        index = batch[2] if len(batch) >= 3 else None
        raw = backbone(images.to(device, non_blocking=True))
        if not torch.isfinite(raw).all():
            raise ValueError("Backbone output contains NaN or Inf values.")
        all_features.append(F.normalize(raw, dim=1).cpu())
        labels.append(clean)
        if index is not None:
            indices.append(index)
    features = torch.cat(all_features)
    ordered_indices = torch.cat(indices).long() if indices else torch.arange(len(dataset))
    if not torch.equal(ordered_indices, torch.arange(len(dataset))):
        raise ValueError("Feature extraction sample indices are not in manifest order.")
    return {
        "features": features,
        "clean_labels": torch.cat(labels).long(),
        "indices": ordered_indices,
        "normalized": True,
        "backbone": backbone.model_name,
        "feature_dim": int(features.shape[-1]),
        "checkpoint": backbone.spec.checkpoint,
        "preprocess_source": backbone.preprocessing_source,
        "preprocess_config": backbone.preprocess_config,
    }
