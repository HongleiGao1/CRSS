"""Exact FM-SRM noise wrapper and provenance. No foundation features used for IDN."""
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from crss.utils.io import array_hash,file_hash,object_hash,read_npz,save_npz
from crss.utils.reference import activate

ROOT=Path(__file__).resolve().parents[2]
AUDITED_PROTOCOL = {'reference': 'Learning with noisy labels in foundation model settings via sample reliability '
              'modeling',
 'audited_reference': 'FM-SRM source snapshot bundled under vendor/',
 'seeds': [1, 2, 3, 4, 5],
 'files': {'lnl_foundation/data/noise.py': {'sha256_lf': 'a41985ad7814c7bbb85f51d62e71d31b5f9c92d3ed1577f6f7e907c9c67ca917'},
           'lnl_foundation/data/datasets.py': {'sha256_lf': '9698bd20155a4807d44c4f7a9857505de7cff9df402aeb2ba244654f606cd7f2'},
           'lnl_foundation/utils.py': {'sha256_lf': '64cc849cdb97ac796db095243057ce6ce20b28a291c63136e00f1e17bce088d4'},
           'lnl_foundation/features.py': {'sha256_lf': '469ac2b938c29d85544939a2901ceb7971cc86e0cff3cb889480ed506aa7c874'},
           'configs/default.yaml': {'sha256_lf': 'ca4d987157595304c6b4dda17175749cbba013a9c91c4800f69a114e1b60f08d'},
           'scripts/run_partition_generalization.py': {'sha256_lf': 'c2d579016de5de614ce5f19d5461cd4ebf66be88dc66e91cdf29bf3e7a68b012'}},
 'findings': {'datasets': 'torchvision CIFAR10/CIFAR100 train=True; targets int64; native train '
                          'order',
              'human': 'CIFAR-10_human.pt worse_label; CIFAR-100_human.pt noisy_label; clean_label '
                       'order equality required',
              'sym': 'off-diagonal .6/(C-1), diagonal .4; per-row np.random.choice after '
                     'set_seed(seed)',
              'pair': 'cyclic c -> (c+1)%C with probability .3; diagonal .7; per-row '
                      'np.random.choice',
              'idn': 'original JYP matrix IDN; raw 32x32 images; ToTensor + CIFAR10 mean/std for '
                     'BOTH datasets; truncnorm tau=.4 std=.1; NumPy normal Cx3072xC weights; torch '
                     'softmax; per-row np.random.choice',
              'seed': 'reference configs/default.yaml explicitly seeds [1,2,3,4,5]; set_seed '
                      'resets Python/NumPy/Torch/CUDA',
              'storage': 'run_partition_generalization.py outputs partition.csv index/noisy_label; '
                         'metrics.json records noise_name/seed/device; no synthetic label archives '
                         'found in audited checkout',
              'ordering': 'features.py DataLoader shuffle=False; ordered indices=arange(N)',
              'alignment_limit': 'generator implementation and default seeds recovered; historical '
                                 'labels/environment require archived partition.csv for bitwise '
                                 'MATCHED'}}
SEEDS=(1,2,3,4,5)
NOISES=('human','sym','pair','idn')
RATES={'human':None,'sym':.6,'pair':.3,'idn':.4}
TYPES={'human':'human','sym':'symmetric','pair':'pairflip','idn':'instance'}
HUMAN={'cifar10':'human_worse_label','cifar100':'human_noisy_label'}


def protocol_manifest():
    data=AUDITED_PROTOCOL
    if tuple(data['seeds'])!=SEEDS:raise ValueError('Audited reference seed protocol mismatch')
    for f in ['lnl_foundation/data/noise.py','lnl_foundation/data/datasets.py','lnl_foundation/utils.py']:
        raw=(ROOT/'crss/_vendor'/f).read_bytes().replace(b'\r\n',b'\n')
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=data['files'][f]['sha256_lf']:raise ValueError('Audited reference source drift: '+f)
    return data


def noise_name(dataset,noise):
    return HUMAN[dataset] if noise=='human' else TYPES[noise]+'_'+str(RATES[noise])


def find_reference_labels(root,dataset,noise,seed,n):
    """Prefer recorded FM-SRM native partition labels; reject conflicting archives."""
    matches=[]
    for p in sorted(Path(root).glob(dataset+'/**/metrics.json')):
        m=json.loads(p.read_text(encoding='utf-8'))
        if m.get('dataset')!=dataset or m.get('noise_name')!=noise_name(dataset,noise) or int(m.get('seed',-1))!=seed:continue
        if noise!='human' and m.get('noise_ratio')!=RATES[noise]:continue
        if noise=='idn' and m.get('noise_generator')!='jyp_matrix_idn':raise ValueError('Reference IDN archive does not certify the audited JYP matrix protocol: '+str(p))
        path=p.with_name('partition.csv')
        if not path.is_file():continue
        import pandas as pd
        frame=pd.read_csv(path,usecols=['index','noisy_label']).sort_values('index')
        if len(frame)!=n or not np.array_equal(frame['index'].to_numpy(),np.arange(n)):raise ValueError('Reference sample order mismatch: '+str(path))
        labels=frame['noisy_label'].to_numpy()
        if not np.issubdtype(labels.dtype,np.integer):raise ValueError('Noninteger reference labels')
        if matches and not np.array_equal(matches[0][0],labels):raise ValueError('Conflicting reference noisy labels for same dataset/noise/seed; select one explicit --reference-results root')
        matches.append((labels,path,m))
    return matches[0] if matches else None


def make_noise(cfg,base,noise,seed):
    manifest=protocol_manifest();dataset=cfg['dataset']['name'];clean=np.asarray(base.targets,dtype=np.int64);indices=np.arange(len(clean),dtype=np.int64)
    classes=10 if dataset=='cifar10' else 100
    activate()
    from lnl_foundation.data.noise import make_noisy_labels
    generator=ROOT/'crss/_vendor/lnl_foundation/data/noise.py'
    human_source=Path(cfg['stage12']['human_root'])/'noise_label_human'/('CIFAR-10_human.pt' if dataset=='cifar10' else 'CIFAR-100_human.pt')
    source_sha=file_hash(human_source) if noise=='human' and human_source.is_file() else None
    archive=find_reference_labels(cfg['stage12']['reference_results'],dataset,noise,seed,len(clean))
    image_sha=array_hash(base.data) if noise=='idn' else None
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','torchvision']}
    identity=dict(dataset=dataset,noise=noise,rate=RATES[noise],seed=seed,device=cfg['stage12']['noise_device'],
        generator_sha=manifest['files']['lnl_foundation/data/noise.py']['sha256_lf'],clean_sha=array_hash(clean),indices_sha=array_hash(indices),image_sha=image_sha,
        human_source_sha=source_sha,reference_sha=file_hash(archive[1]) if archive else None,versions=versions)
    path=Path(cfg['feature']['root'])/'noise/stage12'/dataset/f'{noise}_seed{seed}_{object_hash(identity)[:20]}.npz'
    cache_audit=path.with_suffix('.json')
    if path.is_file() and cache_audit.is_file():
        stored=read_npz(path);audit=json.loads(cache_audit.read_text(encoding='utf-8'))
        if audit.get('cache_identity')!=identity or not np.array_equal(stored['clean_labels'],clean) or not np.array_equal(stored['indices'],indices) or audit['noisy_label_sha']!=array_hash(stored['observed_labels']):raise ValueError('Stale/corrupt stage12 noise cache')
        return stored['observed_labels'],audit
    if archive:
        observed=archive[0].astype(np.int64)
        status='MATCHED';reason='Exact archived FM-SRM partition.csv labels reused; no regenerated historical equivalence claimed.'
        # Human archives must still agree with the original immutable annotation.
        if noise=='human':
            official=make_noisy_labels(dataset,cfg['stage12']['human_root'],base.data,clean,classes,'human',None,seed,device=cfg['stage12']['noise_device'],human_noise_type=HUMAN[dataset])
            if not np.array_equal(official,observed):raise ValueError('Human reference archive differs from official annotation')
    else:
        observed=make_noisy_labels(dataset,cfg['stage12']['human_root'],base.data,clean,classes,TYPES[noise],RATES[noise],seed,
            device=cfg['stage12']['noise_device'],human_noise_type=HUMAN[dataset] if noise=='human' else None)
        status='MATCHED' if noise=='human' else 'NOT EXACTLY MATCHED'
        reason=('Exact fixed CIFAR-N annotation via unchanged FM-SRM loader; clean order checked.' if noise=='human' else
            'Exact audited FM-SRM generator/default seed list reused; no historical label archive available. Bitwise historical match cannot be certified (especially IDN device/library effects).')
    if observed.shape!=clean.shape or not np.issubdtype(observed.dtype,np.integer) or np.any((observed<0)|(observed>=classes)):raise ValueError('Invalid paper noise labels')
    audit=dict(dataset=dataset,noise_type=noise,noise_rate=RATES[noise],human_label_source=str(human_source) if noise=='human' else None,
        human_label_source_sha=source_sha,generator_source_file=str(generator),generator_source_sha=file_hash(generator),generator_canonical_sha=identity['generator_sha'],
        seed=seed,actual_noise_rate=float((observed!=clean).mean()),clean_label_sha=array_hash(clean),noisy_label_sha=array_hash(observed),
        sample_index_sha=array_hash(indices),image_sha=image_sha,alignment_status=status,alignment_reason=reason,
        reference_label_file=str(archive[1]) if archive else None,reference_label_sha=file_hash(archive[1]) if archive else None,
        noise_device=cfg['stage12']['noise_device'],reference_device=archive[2].get('device') if archive else None,
        versions=versions,noise_cache=str(path),cache_identity=identity,human_evaluation='deterministic fixed-label evaluation' if noise=='human' else None)
    save_npz(path,indices=indices,clean_labels=clean,observed_labels=observed)
    cache_audit.parent.mkdir(parents=True,exist_ok=True);cache_audit.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return observed,audit
