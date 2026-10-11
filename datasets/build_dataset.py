"""Stable indexed datasets. Clean labels are retained for offline diagnostics only."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
from utils.reference import activate

REGISTRY = {}

def register_dataset(name, loader):
    REGISTRY[name] = loader

class IndexedDataset:
    def __init__(self, base, clean_labels, class_names, transform=None, observed_labels=None):
        self.base = base
        self.clean_labels = np.asarray(clean_labels, dtype=np.int64)
        self.class_names = list(class_names)
        self.observed_labels = observed_labels
        self.indices = np.arange(len(self.clean_labels), dtype=np.int64)
        self.num_classes = len(class_names)
        self.transform = transform
        if len(base) != len(self.clean_labels): raise ValueError('Dataset/GT length mismatch')
        if np.any((self.clean_labels < 0) | (self.clean_labels >= self.num_classes)): raise ValueError('Invalid GT')

    def __len__(self): return len(self.clean_labels)

    def __getitem__(self, index):
        image = self.base[index][0]
        if self.transform is not None: image = self.transform(image)
        return image, int(self.clean_labels[index]), int(self.indices[index])

class ManifestImages:
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as source: return source.convert('RGB'), None

def build_dataset(cfg, transform=None):
    spec = cfg['dataset']; name = spec['name'].lower()
    if name in REGISTRY: return REGISTRY[name](cfg, transform)
    if name in {'cifar10','cifar100','cifar10n','cifar100n'}:
        activate()
        from lnl_foundation.data.datasets import load_cifar_base
        canonical = name.removesuffix('n')
        base = load_cifar_base(canonical, spec['root'], train=True, transform=None, download=spec['download'])
        return IndexedDataset(base, base.targets, base.classes, transform)
    if not spec['manifest']:
        raise ValueError(f'{name}: provide dataset.manifest CSV with path,clean_label,observed_label. '
                         'Animal-10N/WebVision annotations alone are NOT clean GT; Q1-Q4 cannot assume otherwise.')
    with Path(spec['manifest']).open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    if not rows or not {'path','clean_label','observed_label'} <= rows[0].keys(): raise ValueError('Invalid GT manifest')
    paths = [Path(spec['root']) / row['path'] for row in rows]
    clean = np.asarray([int(row['clean_label']) for row in rows])
    observed = np.asarray([int(row['observed_label']) for row in rows])
    names = [str(c) for c in range(int(max(clean.max(), observed.max())) + 1)]
    return IndexedDataset(ManifestImages(paths), clean, names, transform, observed)
