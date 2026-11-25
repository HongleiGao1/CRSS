"""Prepare immutable observed labels once, for CRSS."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from crss.data.cached_inputs import absolute, observed_label_cache
from crss.data.stage12_noise import make_noise, NOISES, RATES
from crss.utils.config import load_config
from crss.utils.io import read_npz
from crss.utils.reference import activate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('cifar10', 'cifar100', 'all'))
    parser.add_argument('--noise', choices=(*NOISES, 'all'), default='all')
    parser.add_argument('--seed', choices=('1', '2', '3', '4', '5', 'all'), default='all')
    parser.add_argument('--config', default='server')
    parser.add_argument('--data-root', default='../DATASETS/')
    parser.add_argument('--human-root', help='Parent of noise_label_human; defaults to data-root')
    parser.add_argument('--cache-root', default='cache')
    parser.add_argument('--noise-device', default='cpu', help='cpu or cuda:N; recorded in IDN provenance')
    parser.add_argument('--reference-results', help='Optional archived FM-SRM partitions for exact historical labels')
    parser.add_argument('--download', action='store_true', help='Download CIFAR images; CIFAR-N files are supplied separately')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    datasets = ('cifar10', 'cifar100') if args.dataset == 'all' else (args.dataset,)
    noises = NOISES if args.noise == 'all' else (args.noise,)
    tasks = [(d, n, s) for d in datasets for n in noises
             for s in ((1,) if n == 'human' else range(1, 6) if args.seed == 'all' else (int(args.seed),))]
    if args.dry_run:
        for d, n, s in tasks:
            print(d, n, RATES[n], s)
        print(f'{len(tasks)} immutable label realization(s); scales share the same labels.')
        return 0
    activate()
    from lnl_foundation.data.datasets import load_cifar_base
    for dataset in datasets:
        cfg = load_config(['--config', args.config, '--dataset', dataset,
                           '--data-root', args.data_root, '--noise-type', 'none',
                           '--set', 'feature.root=' + str(absolute(args.cache_root))])
        cfg['stage12'] = dict(human_root=str(absolute(args.human_root or args.data_root)),
            reference_results=str(absolute(args.reference_results or 'cache/reference_labels')),
            noise_device=args.noise_device)
        base = load_cifar_base(dataset, cfg['dataset']['root'], train=True,
                               transform=None, download=args.download or cfg['dataset']['download'])
        if len(base) != 50000:
            raise ValueError('Expected the complete CIFAR training set')
        for _, noise, seed in (task for task in tasks if task[0] == dataset):
            directory = absolute(args.cache_root) / 'noise/stage12' / dataset
            matches = sorted(directory.glob(f'{noise}_seed{seed}_*.npz'))
            if matches:
                # Existing labels are authoritative across machines/library versions.
                # Never regenerate or overwrite an existing realization.
                helper = SimpleNamespace(observed_labels=None,
                    noise_cache_root=str(absolute(args.cache_root) / 'noise/stage12'))
                indices, observed, path = observed_label_cache(dataset, noise, seed, helper)
                payload = read_npz(path)
                if (not np.array_equal(indices, np.arange(50000)) or
                    observed.shape != (50000,) or not np.issubdtype(observed.dtype, np.integer) or
                    np.any((observed < 0) | (observed >= len(base.classes))) or
                    ('clean_labels' in payload and not np.array_equal(payload['clean_labels'], base.targets))):
                    raise ValueError('Existing observed labels do not match the CIFAR sample order/range')
                print('REUSE', path, flush=True)
                continue
            _, audit = make_noise(cfg, base, noise, seed)
            print(json.dumps({k: audit[k] for k in ('dataset', 'noise_type', 'seed',
                  'actual_noise_rate', 'alignment_status', 'noise_cache')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
