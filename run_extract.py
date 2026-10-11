import argparse

from utils.config import load_config
from utils.run import create_run

def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--split',choices=('train','test'),default='train')
    args,remaining=parser.parse_known_args(argv)
    cfg=load_config(remaining); destination,logger=create_run(cfg,'extract_test' if args.split=='test' else 'extract')
    try:
        from features.extract import run_extract
        run_extract(cfg,logger,split=args.split)
        logger.info('Feature extraction complete: %s',destination)
    except Exception:
        logger.exception('Extraction failed'); raise

if __name__=='__main__': main()
