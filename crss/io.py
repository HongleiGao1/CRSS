"""CRSS result serialization and transparent Stage II loading."""
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np

from crss.types import CRSSResult

ROOT=Path(__file__).resolve().parents[1]
RESULT_FIELDS=('indices','observed_labels','expert_predictions','intrinsic_confidence',
    'extrinsic_consensus','expert_reliability','usc_distribution','usc_prediction',
    'usc_confidence','reliable_candidate_mask','rvc_delta','rc_mask','cn_mask',
    'u_mask','group_id','u_reason')


def git_version() -> str|None:
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=False)
    return result.stdout.strip() if result.returncode==0 else None


def _atomic_json(path:Path,value:dict) -> None:
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


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
