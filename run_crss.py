"""Partition prepared CIFAR features and train the CRSS linear head."""
from crss.stage2.artifacts import obtain_partition
from crss.stage2.config import parse_args
from crss.stage2.training import train_stage2_linear_head


def main(argv=None):
    args = parse_args(argv)
    result, metadata, path = obtain_partition(args, stage1=args.stage != 'stage2')
    if args.stage != 'stage2':
        print(f'RC: {result.rc_mask.sum()}, CN: {result.cn_mask.sum()}, U: {result.u_mask.sum()}')
        print(f'Partition: {path}')
    if args.stage == 'stage1':
        return metadata

    metrics, directory = train_stage2_linear_head(args, result, metadata, path)
    print(f'Best test accuracy: {metrics["accuracy"]:.2%} (epoch {metrics["best_epoch"]})')
    print(f'Checkpoint: {directory / "best_head.pt"}')
    return metrics


if __name__ == '__main__':
    main()
