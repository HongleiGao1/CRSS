"""Effective CLI configuration and stable identities."""
import argparse
import math
from pathlib import Path
import yaml
from crss.config import CRSSConfig
from utils.io import object_hash

ROOT=Path(__file__).resolve().parents[1]
METHOD_VERSION='crss-two-stage-v1'
DEFAULTS=dict(rho=.6,tau=.1,q_gce=.7,lr=.001,weight_decay=.0001,batch_size=256,
              max_epochs=100,patience=20,training_seed=0,target_expert='vit')


def absolute(value):
    p=Path(value).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()


def format_rho_tag(rho):
    """Four-place readable prefix, exact float hash for non-four-place values."""
    tag=f'rho_{rho:.4f}'.replace('.','p')
    if float(f'{rho:.4f}')!=rho:tag+='_'+object_hash(float(rho))[:16]
    return tag


def parse_args(argv=None):
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config',default='configs/crss_two_stage.yaml')
    prelim,_=pre.parse_known_args(argv)
    cfg=yaml.safe_load(absolute(prelim.config).read_text(encoding='utf-8')) or {}
    p=argparse.ArgumentParser(description='CRSS: supervision stratification and all-sample robust linear training.',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',default=prelim.config)
    p.add_argument('--stage',choices=('stage1','stage2','all'),default='all')
    p.add_argument('--dataset',choices=('cifar10','cifar100'),required=True)
    p.add_argument('--model-scale',choices=('weak','base','strong'),default='base')
    p.add_argument('--noise',choices=('human','sym','pair','idn'),default='human')
    p.add_argument('--seed',type=int,default=1)
    for key,default in DEFAULTS.items():
        typ=int if isinstance(default,int) else str if isinstance(default,str) else float
        help_text=dict(rho='Global top-rho candidate fraction, finite (0,1]. All rows still train.',
            tau='Positive prototype softmax temperature; changing it requires distinct evidence.',
            q_gce='Positive observed-label GCE exponent for U.',lr='Constant AdamW learning rate.',
            weight_decay='Nonnegative AdamW weight decay.',batch_size='Shuffled training minibatch size; never drop final batch.',
            max_epochs='Training epoch limit.',patience='Stop after this many epochs without strictly higher test Acc.',
            training_seed='Nonnegative head initialization and training shuffle seed.',target_expert='Frozen target family; formal method supports vit only.')[key]
        p.add_argument('--'+key.replace('_','-'),type=typ,default=cfg.get(key,default),help=help_text)
    device=p.add_mutually_exclusive_group();device.add_argument('--cuda',type=int);device.add_argument('--cpu',action='store_true')
    for key,default in dict(feature_config='configs/server.yaml',data_root='../DATASETS/',cache_root='cache',
                            noise_cache_root='cache/noise/stage12',stage1_root='outputs/crss_stage1',
                            output_root='outputs/crss_stage2').items():
        p.add_argument('--'+key.replace('_','-'),default=cfg.get(key,default))
    p.add_argument('--observed-labels');p.add_argument('--clean-labels',help='Optional diagnostic-only .npy labels, original order.')
    p.add_argument('--force-stage1',action='store_true',help='Rebuild partition only; retain core evidence.')
    p.add_argument('--recompute-stage1-evidence',action='store_true',help='Explicitly recompute prototypes and core evidence.')
    p.add_argument('--resume',action='store_true',help='Skip a complete Stage II task only after identity and checksum validation.')
    a=p.parse_args(argv)
    if a.noise=='human':a.seed=1
    CRSSConfig(a.dataset,a.model_scale,a.noise,a.seed,tau=a.tau,rho=a.rho)
    for key in ('q_gce','lr'):
        if not math.isfinite(getattr(a,key)) or getattr(a,key)<=0:p.error(key+' must be finite and positive')
    if not math.isfinite(a.weight_decay) or a.weight_decay<0:p.error('weight_decay must be finite and nonnegative')
    for key in ('batch_size','max_epochs','patience'):
        if getattr(a,key)<1:p.error(key+' must be positive')
    if a.training_seed<0:p.error('training_seed must be nonnegative')
    if a.target_expert!='vit':p.error('The formal target expert is vit')
    if a.cuda is not None and a.cuda<0:p.error('cuda must be nonnegative')
    if a.stage=='stage2' and (a.force_stage1 or a.recompute_stage1_evidence):p.error('Stage II never creates a partition; use --stage stage1 or all')
    a.device='cpu' if a.cpu else f'cuda:{a.cuda}' if a.cuda is not None else 'auto'
    a.changed_parameters={k:dict(default=v,actual=getattr(a,k)) for k,v in DEFAULTS.items() if getattr(a,k)!=v}
    a.canonical_run=not a.changed_parameters
    return a
