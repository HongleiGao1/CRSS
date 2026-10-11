"""Extract train/test features for one or all scales using the original loader."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_extract import main as extract_one
from models.model_registry import FAMILIES, SCALES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('cifar10', 'cifar100', 'all'))
    parser.add_argument('--model-scale', choices=(*SCALES, 'all'), default='base')
    parser.add_argument('--split', choices=('train', 'test', 'all'), default='all')
    parser.add_argument('--models', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--config', default='configs/server.yaml')
    parser.add_argument('--data-root', default='../DATASETS/')
    parser.add_argument('--cache-root', default='cache')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--force-extract', action='store_true')
    args = parser.parse_args(argv)
    datasets = ('cifar10', 'cifar100') if args.dataset == 'all' else (args.dataset,)
    scales = SCALES if args.model_scale == 'all' else (args.model_scale,)
    splits = ('train', 'test') if args.split == 'all' else (args.split,)
    for dataset in datasets:
        for scale in scales:
            for split in splits:
                flags = ['--config', args.config, '--dataset', dataset, '--data-root', args.data_root,
                    '--model-scale', scale, '--split', split, '--noise-type', 'none',
                    '--device', args.device, '--models', *args.models,
                    '--set', 'feature.root=' + args.cache_root]
                if args.batch_size is not None:
                    flags += ['--set', 'dataset.batch_size=' + str(args.batch_size)]
                if args.num_workers is not None:
                    flags += ['--set', 'dataset.num_workers=' + str(args.num_workers)]
                if args.download:
                    flags += ['--set', 'dataset.download=true']
                if args.force_extract:
                    flags += ['--force-extract']
                extract_one(flags)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
