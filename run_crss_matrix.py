"""Run the published task matrix through the single-task CRSS entry point."""
import argparse
import sys
import traceback
from crss.stage2.config import parse_args, absolute
from run_crss import main as run_one
from crss.utils.io import object_hash, save_json

HUMAN_TRAINING_SEEDS = (0, 1, 2, 3, 4)


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
    parser.add_argument('--seed', choices=('1', '2', '3', '4', '5', 'all'), default='all',
                        help='Synthetic noisy-label seed. Human always uses the fixed annotations.')
    parser.add_argument('--training-seed', type=int,
                        help='Run one head seed per task. Human defaults to five head seeds (0–4).')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    if args.training_seed is not None:
        remaining += ['--training-seed', str(args.training_seed)]
    tasks = task_matrix(args.dataset, args.model_scale, args.noise, args.seed)
    parsed = []
    for dataset, scale, noise, seed in tasks:
        flags = [*remaining, '--dataset', dataset, '--model-scale', scale, '--noise', noise, '--seed', str(seed)]
        task = parse_args(flags)
        if noise != 'human' or task.stage == 'stage1' or args.training_seed is not None:
            parsed.append((flags, task))
            continue

        for index, training_seed in enumerate(HUMAN_TRAINING_SEEDS):
            seed_flags = flags + ['--training-seed', str(training_seed)]
            if index:
                seed_flags = [flag for flag in seed_flags
                              if flag not in ('--force-stage1', '--recompute-stage1-evidence')]
            parsed.append((seed_flags, parse_args(seed_flags)))
    if parsed[0][1].observed_labels and len({(a.dataset, a.noise, a.seed) for _, a in parsed}) != 1:
        parser.error('--observed-labels requires one dataset/noise/seed realization')
    print(f'CRSS selected tasks: {len(parsed)}', flush=True)
    failures = []
    for flags, task in parsed:
        if args.dry_run:
            name = '/'.join(map(str, (task.dataset, task.model_scale, task.noise, task.seed)))
            if task.noise == 'human' and task.stage != 'stage1':
                name += f'/training_seed_{task.training_seed}'
            print(name)
            continue
        try:
            run_one(flags)
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            failures.append(dict(dataset=task.dataset, scale=task.model_scale,
                noise=task.noise, seed=task.seed, training_seed=task.training_seed, error=error))
    if not args.dry_run:
        scope = dict(dataset=args.dataset, scale=args.model_scale, noise=args.noise,
                     seed=args.seed, options=remaining)
        report = absolute(parsed[0][1].output_root) / 'runs' / object_hash(scope)[:20] / 'failed_tasks.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        save_json(report, dict(selected_tasks=len(parsed), failed_tasks=failures))
        print(f'Completed: {len(parsed)-len(failures)}/{len(parsed)}; manifest: {report}', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
