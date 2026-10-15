"""Immutable evidence and rho-specific partition contracts, with canonical import."""
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from crss.config import CRSSConfig,EXPERTS
from crss.io import RESULT_FIELDS,load_result,_atomic_json,git_version
from crss.pipeline import compute_crss_core_evidence,apply_crss_partition
from crss.types import CRSSResult
from crss.diagnostics import evaluate_crss_partition
from crss.features.cache import cached_geometry_path,model_identity
from crss.models.model_registry import MODEL_SCALE_REGISTRY
from crss.utils.config import load_config
from crss.utils.io import array_hash,file_hash,object_hash,read_npz,save_npz
from crss.data.cached_inputs import feature_group,observed_label_cache
from .config import METHOD_VERSION,absolute,format_rho_tag

CORE_FIELDS=tuple(k for k in RESULT_FIELDS if k not in ('reliable_candidate_mask','rc_mask','cn_mask','u_mask','group_id','u_reason'))


def sources(args):
    """Resolve existing five-model caches only. Never load a foundation model."""
    cfg=load_config(['--config',args.feature_config,'--dataset',args.dataset,
        '--model-scale',args.model_scale,'--data-root',str(absolute(args.data_root)),
        '--noise-type','none','--seed','1','--set','feature.root='+str(absolute(args.cache_root)),
        '--models',*EXPERTS])
    hashes={};paths={};indices=None
    for family in EXPERTS:
        path=cached_geometry_path(cfg,family)
        if not path.is_file():raise FileNotFoundError('Frozen feature cache missing: '+str(path))
        with np.load(path,allow_pickle=False) as f:
            hashes[family]=str(f['feature_sha256']);ids=f['indices'];m=json.loads(str(f['metadata']))
            matrix=f['features']
            if matrix.ndim!=2 or len(matrix)!=len(ids) or not np.isfinite(matrix).all() or array_hash(matrix)!=hashes[family] or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-4):
                raise ValueError('Frozen feature matrix/hash/normalization mismatch: '+str(path))
            del matrix
        if len(ids)!=50000 or not np.array_equal(ids,np.arange(50000)):
            raise ValueError('Expected all 50000 CIFAR training samples in original order')
        spec=MODEL_SCALE_REGISTRY[args.model_scale][family]
        if m.get('scale','base')!=args.model_scale or m.get('frozen') is not True or m.get('pretrained') is not True or m.get('model_identity')!=model_identity(cfg['models'][family]) or m.get('actual_model_name',spec['model_name'])!=spec['model_name']:
            raise ValueError('Frozen feature scale/provenance mismatch: '+str(path))
        if indices is None:indices=ids
        elif not np.array_equal(indices,ids):raise ValueError('Five-expert indices differ')
        paths[family]=str(path)
    helper=SimpleNamespace(**vars(args));helper.config=args.feature_config
    label_ids,observed,label_path=observed_label_cache(args.dataset,args.noise,args.seed,helper)
    if not np.array_equal(indices,label_ids):raise ValueError('Feature/observed-label order mismatch')
    identity=dict(dataset=args.dataset,model_scale=args.model_scale,noise_type=args.noise,seed=args.seed,
        tau=args.tau,expert_names=list(EXPERTS),feature_source_hashes=hashes,
        observed_label_hash=array_hash(observed),evidence_version='crss-evidence-v1')
    return identity,paths,indices,observed,label_path


def task_directory(args):
    return absolute(args.stage1_root)/args.dataset/args.model_scale/args.noise/f'seed_{args.seed}'


def validate_result(result,rho):
    from crss.training.linear import validate_partition
    validate_partition(result.to_stage2_bundle())
    expected=apply_crss_partition(core_from_result(result),rho)
    for key in ('reliable_candidate_mask','rc_mask','cn_mask','u_mask','group_id','u_reason'):
        if not np.array_equal(getattr(result,key),getattr(expected,key)):raise ValueError('Partition rule mismatch: '+key)
    n=len(result.indices);c=result.usc_distribution.shape[1]
    shapes=dict(expert_predictions=(n,5),intrinsic_confidence=(n,5),extrinsic_consensus=(n,5),
                expert_reliability=(n,5),expert_probabilities=(n,5,c),usc_distribution=(n,c))
    for key,shape in shapes.items():
        v=np.asarray(getattr(result,key))
        if v.shape!=shape or not np.isfinite(v).all():raise ValueError('Invalid evidence: '+key)
    if not np.allclose(result.usc_distribution.sum(1),1) or not np.array_equal(result.usc_confidence,result.usc_distribution.max(1)):
        raise ValueError('USC evidence inconsistency')
    if not np.allclose(result.expert_probabilities.sum(2),1) or np.any(result.expert_probabilities<0) or not np.array_equal(result.expert_probabilities.argmax(2),result.expert_predictions) or not np.array_equal(result.expert_probabilities.mean(1),result.usc_distribution):
        raise ValueError('Five-expert probabilities/USC mismatch')


def core_from_result(result):
    return {k:getattr(result,k) for k in (*CORE_FIELDS,'expert_probabilities')}


def core_hash(core,identity):
    return object_hash(dict(identity=identity,arrays={k:array_hash(v) for k,v in core.items()}))


def canonical_result(args,identity,indices,observed):
    directory=task_directory(args)
    if args.tau!=.1 or not all((directory/x).is_file() for x in ('metadata.json','crss_result.npz','expert_probs.npy')):return None
    m=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if m.get('method_version')!='1.0.0' or m.get('rho')!=.6 or m.get('tau')!=args.tau:return None
    for key in ('dataset','model_scale','noise_type','seed','expert_names','feature_source_hashes','observed_label_hash'):
        if m.get(key)!=identity[key]:return None
    result=load_result(directory)
    if not np.array_equal(result.indices,indices) or not np.array_equal(result.observed_labels,observed):raise ValueError('Canonical label/index mismatch')
    validate_result(result,.6)
    return result


def partition_hash(result,rho,tau,source_core_hash):
    return object_hash(dict(rho=rho,tau=tau,source_core_hash=source_core_hash,
                           arrays={k:array_hash(getattr(result,k)) for k in RESULT_FIELDS}))


def load_core(directory,identity):
    m=json.loads((directory/'core_metadata.json').read_text(encoding='utf-8'))
    if m['identity']!=identity or m['method_version']!=METHOD_VERSION:raise ValueError('Core identity mismatch')
    for name in ('core.npz','expert_probs.npy'):
        if file_hash(directory/name)!=m['file_hashes'][name]:raise ValueError('Core checksum mismatch: '+name)
    core=read_npz(directory/'core.npz');core['expert_probabilities']=np.load(directory/'expert_probs.npy',mmap_mode='r',allow_pickle=False)
    if core_hash(core,identity)!=m['source_core_hash']:raise ValueError('Core evidence hash mismatch')
    return core,m


def write_core(directory,core,identity,paths,canonical_source=None):
    directory.mkdir(parents=True,exist_ok=True)
    save_npz(directory/'core.npz',**{k:core[k] for k in CORE_FIELDS})
    if canonical_source:shutil.copy2(canonical_source/'expert_probs.npy',directory/'expert_probs.npy')
    else:np.save(directory/'expert_probs.npy',core['expert_probabilities'])
    m=dict(identity=identity,method_version=METHOD_VERSION,source_core_hash=core_hash(core,identity),
           feature_paths=paths,canonical_source=str(canonical_source) if canonical_source else None,
           git_commit=git_version(),file_hashes={x:file_hash(directory/x) for x in ('core.npz','expert_probs.npy')})
    _atomic_json(directory/'core_metadata.json',m)
    return m


def load_partition(directory,identity,rho,tau,indices,observed):
    m=json.loads((directory/'partition_metadata.json').read_text(encoding='utf-8'))
    if m.get('identity')!=identity or m.get('rho')!=rho or m.get('tau')!=tau or m.get('method_version')!=METHOD_VERSION:
        raise ValueError('Requested rho/task does not match partition metadata')
    if file_hash(directory/'partition.npz')!=m['file_sha256']:raise ValueError('Partition file checksum mismatch')
    core_dir=directory.parents[2]/'core'/m['core_key']
    core,cm=load_core(core_dir,identity)
    with np.load(directory/'partition.npz',allow_pickle=False) as f:
        result=CRSSResult(**{k:f[k] for k in RESULT_FIELDS},expert_probabilities=core['expert_probabilities'])
        if str(f['partition_hash'])!=m['partition_hash'] or str(f['source_core_hash'])!=cm['source_core_hash'] or float(f['rho'])!=rho or float(f['tau'])!=tau:
            raise ValueError('Embedded partition identity mismatch')
    if not np.array_equal(indices,result.indices) or not np.array_equal(observed,result.observed_labels):raise ValueError('Partition sample/label mismatch')
    if core_hash(core_from_result(result),identity)!=cm['source_core_hash']:raise ValueError('Partition/core evidence mismatch')
    if partition_hash(result,rho,tau,cm['source_core_hash'])!=m['partition_hash']:raise ValueError('Partition content hash mismatch')
    validate_result(result,rho)
    return result,m,directory/'partition.npz'


def write_partition(directory,result,args,identity,cm,core_key,clean_labels=None):
    directory.mkdir(parents=True,exist_ok=True)
    h=partition_hash(result,args.rho,args.tau,cm['source_core_hash'])
    save_npz(directory/'partition.npz',**{k:getattr(result,k) for k in RESULT_FIELDS},
        sample_indices=result.indices,usc_pred=result.usc_prediction,Q_USC=result.usc_confidence,
        candidate_mask=result.reliable_candidate_mask,Delta_R=result.rvc_delta,
        RC_mask=result.rc_mask,CN_mask=result.cn_mask,U_mask=result.u_mask,q_USC=result.usc_distribution,
        rho=np.array(args.rho),tau=np.array(args.tau),source_core_hash=np.array(cm['source_core_hash']),partition_hash=np.array(h))
    m=dict(identity=identity,rho=args.rho,tau=args.tau,method_version=METHOD_VERSION,partition_hash=h,
        source_core_hash=cm['source_core_hash'],core_key=core_key,feature_paths=cm['feature_paths'],
        canonical_run=args.canonical_run,changed_parameters=args.changed_parameters,git_commit=git_version(),
        file_sha256=file_hash(directory/'partition.npz'))
    _atomic_json(directory/'partition_metadata.json',m)
    metrics=evaluate_crss_partition(result,clean_labels,args.rho)
    _atomic_json(directory/'stage1_metrics.json',metrics)
    with (directory/'stage1_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics));w.writeheader();w.writerow(metrics)
    names=['sample_index','observed_label','usc_pred','Q_USC','candidate','Delta_R','group']
    if clean_labels is not None:names+=['clean_label','observed_is_clean','usc_is_correct']
    with (directory/'sample_assignments.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(names)
        for i in range(len(result.indices)):
            row=[int(result.indices[i]),int(result.observed_labels[i]),int(result.usc_prediction[i]),
                 float(result.usc_confidence[i]),bool(result.reliable_candidate_mask[i]),float(result.rvc_delta[i]),('RC','CN','U')[result.group_id[i]]]
            if clean_labels is not None:row += [int(clean_labels[i]),bool(result.observed_labels[i]==clean_labels[i]),bool(result.usc_prediction[i]==clean_labels[i])]
            w.writerow(row)
    return result,m,directory/'partition.npz'


def obtain_partition(args,stage1=True):
    identity,paths,indices,observed,label_path=sources(args)
    task=task_directory(args);key=object_hash(identity)
    canonical=None
    # Canonical 0.6 always has first priority unless explicit recomputation was requested.
    if not args.recompute_stage1_evidence:
        canonical=canonical_result(args,identity,indices,observed)
    # Keep imported canonical evidence distinct from an explicitly recomputed copy.
    if canonical is not None:key+='_'+core_hash(core_from_result(canonical),identity)[:16]
    elif not args.recompute_stage1_evidence and not (task/'core'/key/'core_metadata.json').is_file():
        # Imported artifacts remain usable after the legacy canonical copy is archived.
        matches=[]
        for cached in (task/'core').glob(key+'*'):
            mp=cached/'core_metadata.json'
            if mp.is_file() and json.loads(mp.read_text(encoding='utf-8')).get('identity')==identity:matches.append(cached.name)
        if len(matches)==1:key=matches[0]
        elif len(matches)>1:raise ValueError('Ambiguous matching core evidence; retain the canonical copy or select a separate --stage1-root')
    core_dir=task/'core'/key
    part_dir=task/'partitions'/format_rho_tag(args.rho)/key
    if not stage1:
        if (part_dir/'partition_metadata.json').is_file():
            result,m,path=load_partition(part_dir,identity,args.rho,args.tau,indices,observed)
        elif args.rho==.6 and canonical is not None:
            core=core_from_result(canonical);ch=core_hash(core,identity)
            result=canonical;path=task/'crss_result.npz'
            m=dict(identity=identity,rho=args.rho,tau=args.tau,method_version=METHOD_VERSION,
                   source_core_hash=ch,partition_hash=partition_hash(result,args.rho,args.tau,ch),feature_paths=paths,
                   file_sha256=file_hash(path),core_status='REUSED',partition_status='REUSED')
            print('REUSE_CANONICAL_CRSS_RHO_0.6',flush=True)
        else:
            raise FileNotFoundError('STAGE1_PARTITION_NOT_FOUND_FOR_REQUESTED_RHO\nRun: python run_crss.py --stage stage1 '
                f'--dataset {args.dataset} --model-scale {args.model_scale} --noise {args.noise} --seed {args.seed} '
                f'--rho {args.rho} --tau {args.tau} --config "{args.config}" --feature-config "{args.feature_config}" '
                f'--stage1-root "{args.stage1_root}" --cache-root "{args.cache_root}" --noise-cache-root "{args.noise_cache_root}"'
                +(f' --observed-labels "{args.observed_labels}"' if args.observed_labels else ''))
        return result,m,path
    if args.rho==.6 and canonical is not None and not args.force_stage1:
        print('REUSE_CANONICAL_CRSS_RHO_0.6',flush=True)
    core_status='REUSED'
    if args.recompute_stage1_evidence or not (core_dir/'core_metadata.json').is_file():
        if canonical is not None and not args.recompute_stage1_evidence:
            core=core_from_result(canonical)
            cm=write_core(core_dir,core,identity,paths,task)
        else:
            helper=SimpleNamespace(**vars(args));helper.config=args.feature_config
            features,fids,fhashes,fpaths=feature_group(args.dataset,args.model_scale,helper)
            if fhashes!=identity['feature_source_hashes'] or not np.array_equal(fids,indices):raise ValueError('Feature sources changed during evidence extraction')
            cfg=CRSSConfig(args.dataset,args.model_scale,args.noise,args.seed,tau=args.tau,rho=args.rho)
            core=compute_crss_core_evidence(features,observed,indices,cfg)
            del features
            cm=write_core(core_dir,core,identity,paths);core_status='RECOMPUTED'
    else:core,cm=load_core(core_dir,identity)
    print('REUSE_CRSS_CORE_EVIDENCE' if core_status=='REUSED' else 'RECOMPUTE_CRSS_CORE_EVIDENCE',flush=True)
    clean=None
    if args.clean_labels:clean=np.load(absolute(args.clean_labels),allow_pickle=False)
    else:
        # GT is loaded only after core computation, solely for offline diagnostics.
        with np.load(label_path,allow_pickle=False) as f:
            if 'clean_labels' in f:clean=f['clean_labels']
        if clean is None and Path(paths['vit']).is_file():
            with np.load(paths['vit'],allow_pickle=False) as f:
                if 'clean_labels' in f:clean=f['clean_labels']
    if (part_dir/'partition_metadata.json').is_file() and not args.force_stage1 and not args.recompute_stage1_evidence:
        result,m,path=load_partition(part_dir,identity,args.rho,args.tau,indices,observed);status='REUSED'
        if canonical is not None and args.rho==.6:
            if any(not np.array_equal(getattr(result,k),getattr(canonical,k)) for k in RESULT_FIELDS):raise ValueError('Canonical 0.6/partition mismatch')
    else:
        reuse_canonical=canonical is not None and args.rho==.6 and not args.force_stage1
        if not reuse_canonical:print(f'RECOMPUTE_PARTITION_FOR_RHO={args.rho}',flush=True)
        result=canonical if reuse_canonical else apply_crss_partition(core,args.rho)
        validate_result(result,args.rho)
        result,m,path=write_partition(part_dir,result,args,identity,cm,key,clean);status='REUSED' if reuse_canonical else 'RECOMPUTED'
    m.update(core_status=core_status,partition_status=status)
    return result,m,path
