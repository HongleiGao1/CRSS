"""CRSS result serialization and transparent Stage II loading."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np

from crss.config import CRSSConfig,EXPERTS,FULL_NAME,METHOD_NAME,METHOD_VERSION,RHO,TAU
from crss.types import CRSSResult,Stage2SupervisionBundle
from utils.io import array_hash,file_hash,object_hash

ROOT=Path(__file__).resolve().parents[1]
RESULT_FIELDS=('indices','observed_labels','expert_predictions','intrinsic_confidence',
    'extrinsic_consensus','expert_reliability','usc_distribution','usc_prediction',
    'usc_confidence','reliable_candidate_mask','rvc_delta','rc_mask','cn_mask',
    'u_mask','group_id','u_reason')


def code_hash() -> str:
    return object_hash({p.name:file_hash(p) for p in sorted((ROOT/'crss').glob('*.py'))})


def cache_identity(config:CRSSConfig,feature_hashes:dict[str,str],observed_labels:np.ndarray) -> dict:
    if tuple(feature_hashes)!=EXPERTS:raise ValueError('Require all five feature source hashes')
    return dict(dataset=config.dataset,model_scale=config.model_scale,noise_type=config.noise_setting,
        seed=config.seed,tau=config.tau,rho=config.rho,method_version=METHOD_VERSION,
        feature_source_hashes=feature_hashes,observed_label_hash=array_hash(observed_labels),
        code_hash=code_hash())


def git_version() -> str|None:
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=False)
    return result.stdout.strip() if result.returncode==0 else None


def _atomic_json(path:Path,value:dict) -> None:
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def save_result(directory:Path,result:CRSSResult,config:CRSSConfig,
                feature_hashes:dict[str,str],feature_paths:dict[str,str]) -> dict:
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=cache_identity(config,feature_hashes,result.observed_labels)
    probs_path=directory/'expert_probs.npy'
    temp=probs_path.with_name(probs_path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('wb') as stream:np.save(stream,result.expert_probabilities)
    os.replace(temp,probs_path)
    path=directory/'crss_result.npz';temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('wb') as stream:
        np.savez(stream,**{name:getattr(result,name) for name in RESULT_FIELDS})
    os.replace(temp,path)
    metadata=dict(method_name=METHOD_NAME,full_name=FULL_NAME,method_version=METHOD_VERSION,
        dataset=config.dataset,model_scale=config.model_scale,expert_names=list(EXPERTS),
        noise_type=config.noise_setting,seed=config.seed,tau=config.tau,rho=config.rho,M=len(EXPERTS),
        rank_normalization='average-tie percentile',rvc='hard evidence',
        rvc_rule='Delta_R > 0',feature_source_hashes=feature_hashes,feature_paths=feature_paths,
        observed_label_hash=identity['observed_label_hash'],code_hash=identity['code_hash'],
        git_version=git_version(),timestamp=datetime.now(timezone.utc).isoformat(),
        identity=identity,probability_file=probs_path.name)
    summary=dict(dataset=config.dataset,model_scale=config.model_scale,noise_type=config.noise_setting,
        seed=config.seed,sample_count=len(result.indices),candidate_count=int(result.reliable_candidate_mask.sum()),
        rc_count=int(result.rc_mask.sum()),cn_count=int(result.cn_mask.sum()),u_count=int(result.u_mask.sum()),
        reliable_count=int((result.rc_mask|result.cn_mask).sum()),
        rvc_rejected_count=int((result.u_reason=='RVC_REJECTED').sum()))
    _atomic_json(directory/'summary.json',summary)
    _atomic_json(directory/'metadata.json',metadata)
    return metadata


def result_cache_matches(directory:Path,identity:dict) -> bool:
    directory=Path(directory)
    required=('crss_result.npz','expert_probs.npy','metadata.json','summary.json')
    if any(not (directory/name).is_file() for name in required):return False
    try:
        metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        return metadata.get('identity')==identity and metadata.get('probability_file')=='expert_probs.npy'
    except (OSError,ValueError,TypeError):return False


def load_result(directory:Path,mmap_probabilities:bool=True) -> CRSSResult:
    directory=Path(directory)
    with np.load(directory/'crss_result.npz',allow_pickle=False) as archive:
        arrays={name:archive[name] for name in RESULT_FIELDS}
    probabilities=np.load(directory/'expert_probs.npy',allow_pickle=False,
                          mmap_mode='r' if mmap_probabilities else None)
    n=len(arrays['indices'])
    if probabilities.shape[:2]!=(n,5) or probabilities.shape[2]!=arrays['usc_distribution'].shape[1]:
        raise ValueError('CRSS result/probability sidecar mismatch')
    return CRSSResult(**arrays,expert_probabilities=probabilities)


def load_stage2_bundle(directory:Path,mmap_probabilities:bool=True) -> Stage2SupervisionBundle:
    """Stable loader: Stage II need not import prototype/reliability/RVC internals."""
    return load_result(directory,mmap_probabilities).to_stage2_bundle()
