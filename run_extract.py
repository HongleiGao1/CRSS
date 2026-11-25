"""Extract frozen train/test features for one scale or all fifteen experts."""
import copy
from pathlib import Path

from crss.models.model_registry import SCALES
from crss.utils.config import config_parser, config_from_args, ROOT
from crss.utils.run import create_run

def main(argv=None):
    parser=config_parser(allow_all=True)
    parser.add_argument('--split',choices=('train','test','all'),default='train')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--num-workers',type=int)
    parser.add_argument('--cache-root')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size<1:parser.error('batch-size must be positive')
    if args.num_workers is not None and args.num_workers<0:parser.error('num-workers must be nonnegative')
    datasets=('cifar10','cifar100') if args.dataset=='all' else (args.dataset,)
    scales=SCALES if args.model_scale=='all' else (args.model_scale,)
    splits=('train','test') if args.split=='all' else (args.split,)
    from crss.features.extract import run_extract
    for dataset in datasets:
        for scale in scales:
            task=copy.deepcopy(args);task.dataset=dataset;task.model_scale=scale
            cfg=config_from_args(task)
            if args.batch_size is not None:cfg['dataset']['batch_size']=args.batch_size
            if args.num_workers is not None:cfg['dataset']['num_workers']=args.num_workers
            if args.cache_root:
                path=Path(args.cache_root).expanduser()
                cfg['feature']['root']=str((path if path.is_absolute() else ROOT/path).resolve())
            if args.download:cfg['dataset']['download']=True
            for split in splits:
                destination,logger=create_run(cfg,'extract_test' if split=='test' else 'extract')
                try:
                    run_extract(cfg,logger,split=split)
                    logger.info('Feature extraction complete: %s',destination)
                except Exception:
                    logger.exception('Extraction failed'); raise

if __name__=='__main__': main()
