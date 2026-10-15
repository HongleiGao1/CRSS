"""Run the published task matrix through the single-task CRSS entry point."""
import argparse
from pathlib import Path
import sys
import traceback
from crss.stage2.config import parse_args, absolute
from run_crss import main as run_one
from crss.utils.io import object_hash, save_json


def task_matrix(dataset='all', scale='all', noise='all', seed='all'):
    """Enumerate CRSS tasks; Human has one fixed label realization per scale."""
    def selected(value, values):
        return values if value == 'all' else (value,)
    return [(d, s, n, k) for d in selected(dataset, ('cifar10', 'cifar100'))
            for s in selected(scale, ('weak', 'base', 'strong'))
            for n in selected(noise, ('human', 'sym', 'pair', 'idn'))
            for k in ((1,) if n == 'human' else range(1, 6) if seed == 'all' else (int(seed),))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Other options are forwarded to run_crss.py, including --stage, --cuda, --rho and --resume.')
    parser.add_argument('--dataset', required=True, choices=('cifar10', 'cifar100', 'all'))
    parser.add_argument('--model-scale', choices=('weak', 'base', 'strong', 'all'), default='all')
    parser.add_argument('--noise', choices=('human', 'sym', 'pair', 'idn', 'all'), default='all')
    parser.add_argument('--seed', choices=('1', '2', '3', '4', '5', 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    tasks = task_matrix(args.dataset, args.model_scale, args.noise, args.seed)
    parsed = []
    for dataset, scale, noise, seed in tasks:
        flags = [*remaining, '--dataset', dataset, '--model-scale', scale, '--noise', noise, '--seed', str(seed)]
        parsed.append((flags, parse_args(flags)))
    if parsed[0][1].observed_labels and len({(a.dataset, a.noise, a.seed) for _, a in parsed}) != 1:
        parser.error('--observed-labels requires one dataset/noise/seed realization')
    print(f'CRSS selected tasks: {len(parsed)}', flush=True)
    failures = []
    for flags, task in parsed:
        if args.dry_run:
            print('/'.join(map(str, (task.dataset, task.model_scale, task.noise, task.seed))))
            continue
        try:
            run_one(flags)
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            failures.append(dict(dataset=task.dataset, scale=task.model_scale,
                noise=task.noise, seed=task.seed, error=error))
    if not args.dry_run:
        scope = dict(dataset=args.dataset, scale=args.model_scale, noise=args.noise,
                     seed=args.seed, options=remaining)
        report = absolute(parsed[0][1].output_root) / 'runs' / object_hash(scope)[:20] / 'failed_tasks.json'
        save_json(report, dict(selected_tasks=len(parsed), failed_tasks=failures))
        print(f'Completed: {len(parsed)-len(failures)}/{len(parsed)}; manifest: {report}', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
