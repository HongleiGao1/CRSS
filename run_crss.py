"""Unified production entry: CRSS partition and full-sample robust linear training."""
from pathlib import Path
import json
import sys
from crss.stage2.config import parse_args
from crss.stage2.artifacts import obtain_partition
from crss.stage2.training import train_stage2_linear_head
from crss.diagnostics import evaluate_crss_partition


def main(argv=None):
    args=parse_args(argv)
    result,m,path=obtain_partition(args,stage1=args.stage!='stage2')
    if args.stage!='stage2':
        metrics_path=path.parent/'stage1_metrics.json'
        stats=json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.is_file() else evaluate_crss_partition(result,rho=args.rho)
        print('\nCRSS Stage I Result')
        print(f'Dataset: {args.dataset}; Scale: {args.model_scale}; Noise: {args.noise}; Seed: {args.seed}; rho: {args.rho}; tau: {args.tau}')
        print('Core evidence:',m.get('core_status'));print('Partition:',m.get('partition_status'))
        print(json.dumps(stats,indent=2,ensure_ascii=False));print('Artifact:',path);print('Partition hash:',m['partition_hash'])
    if args.stage=='stage1':return m
    metrics,out=train_stage2_linear_head(args,result,m,path)
    print('\nCRSS Stage II Result')
    print(f'Dataset: {args.dataset}; Scale: {args.model_scale}; Noise: {args.noise}; Seed: {args.seed}; rho: {args.rho}')
    print('Partition:',path);print('Partition hash:',m['partition_hash'])
    print(f'Training samples: {len(result.indices)} / {len(result.indices)}')
    print(f'RC: Observed CE; CN: Hard USC CE; U: Observed GCE; q_gce: {args.q_gce}')
    print(json.dumps(metrics,indent=2));print('Output:',out)
    if args.stage=='all':
        print('\nCRSS FULL PIPELINE RESULT')
        print(f'rho: {args.rho}; Stage I partition hash: {m["partition_hash"]}')
        print(f'RC/CN/U: {result.rc_mask.sum()}/{result.cn_mask.sum()}/{result.u_mask.sum()}')
        print(f'Stage II Accuracy: {metrics["accuracy"]}; Raw ECE: {metrics["raw_ece15"]}; Best epoch: {metrics["best_epoch"]}')
        print('Canonical run:',args.canonical_run)
    return metrics


if __name__=='__main__':main()
