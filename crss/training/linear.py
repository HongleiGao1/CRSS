"""Train the CRSS linear head for 100 epochs and retain the highest test accuracy checkpoint."""
import copy
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from crss.training.device import resolve_training_device
from crss.utils.io import array_hash

def validate_partition(bundle):
    n=len(bundle.indices)
    masks=np.stack([bundle.rc_mask,bundle.cn_mask,bundle.u_mask])
    if masks.shape!=(3,n) or not np.all(masks.sum(0)==1):
        raise ValueError('RC/CN/U must be disjoint and exhaustive')
    if not np.array_equal(bundle.indices,np.arange(n)):
        raise ValueError('Expected original sample order')
    for k,mask in enumerate(masks):
        if not np.array_equal(mask,bundle.group_id==k):raise ValueError('group_id/mask mismatch')
    if not np.array_equal(bundle.usc_prediction,bundle.usc_distribution.argmax(1)):
        raise ValueError('USC prediction mismatch')
    if np.any(bundle.cn_mask & (bundle.rvc_delta<=0)):
        raise ValueError('CN violates hard RVC')
    return np.arange(n,dtype=np.int64)

def sample_losses(logits,observed,corrected,group,q=.7):
    if not math.isfinite(q) or q <= 0:raise ValueError('q_gce must be finite and positive')
    logp=F.log_softmax(logits,dim=1)
    rc=F.nll_loss(logp,observed,reduction='none')
    cn=F.nll_loss(logp,corrected,reduction='none')
    u=(1-torch.exp(-q*rc))/q
    return torch.where(group==0,rc,torch.where(group==1,cn,u))

TRAINING_EPOCHS = 100


def update_best(correct,epoch,state,best):
    """Strict improvement, earliest tie, restore an independent state snapshot."""
    if correct>best['correct']:
        best.update(correct=correct,epoch=epoch,
                    state={k:v.detach().cpu().clone() for k,v in state.items()})

def train_head(features,bundle,test_features,test_labels,cfg,device='auto',progress_stream=None):
    rows=validate_partition(bundle)
    s=cfg['stage2'];torch.manual_seed(s['training_seed']);np.random.seed(s['training_seed'])
    device=resolve_training_device(device)
    x=torch.as_tensor(np.asarray(features,dtype=np.float32),device=device)
    tx=torch.as_tensor(np.asarray(test_features,dtype=np.float32),device=device)
    y=torch.as_tensor(bundle.observed_labels.copy(),device=device,dtype=torch.long)
    cy=torch.as_tensor(bundle.usc_prediction.copy(),device=device,dtype=torch.long)
    group=torch.as_tensor(bundle.group_id.copy(),device=device,dtype=torch.long)
    ty=torch.as_tensor(np.asarray(test_labels).copy(),device=device,dtype=torch.long)
    if x.ndim!=2 or len(x)!=len(rows) or tx.ndim!=2 or tx.shape[1]!=x.shape[1]:
        raise ValueError('Feature dimension/sample mismatch')
    if len(tx)!=len(ty) or not len(tx):raise ValueError('Invalid test set')
    head=torch.nn.Linear(x.shape[1],bundle.usc_distribution.shape[1]).to(device)
    initialization=array_hash(*[v.detach().cpu().numpy() for v in head.state_dict().values()])
    opt=torch.optim.AdamW(head.parameters(),lr=s['optimizer']['lr'],weight_decay=s['optimizer']['weight_decay'])
    rng=np.random.default_rng(s['training_seed']);seen=np.zeros(len(rows),dtype=bool)
    best=dict(correct=-1,epoch=0,state=None);history=[];steps=0;steps_at_best=0
    batch=s['batch_size'];counts=[int(m.sum()) for m in [bundle.rc_mask,bundle.cn_mask,bundle.u_mask]]
    output=progress_stream if progress_stream is not None else sys.stderr
    with tqdm(total=TRAINING_EPOCHS,desc='Stage II epochs',unit='epoch',dynamic_ncols=True,file=output) as progress:
        for epoch in range(1,TRAINING_EPOCHS+1):
            head.train();sums=torch.zeros(3,device=device,dtype=torch.float64)
            order=rng.permutation(rows)
            with tqdm(range(0,len(order),batch),desc=f'Epoch {epoch} batches',unit='batch',
                      dynamic_ncols=True,leave=False,position=1,file=output) as batches:
                for start in batches:
                    r=order[start:start+batch];seen[r]=True
                    opt.zero_grad(set_to_none=True)
                    losses=sample_losses(head(x[r]),y[r],cy[r],group[r],s.get('q_gce',.7))
                    loss=losses.mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();opt.step();steps+=1
                    with torch.no_grad():
                        for k in range(3):sums[k]+=losses[group[r]==k].double().sum()
            head.eval()
            with torch.inference_mode():
                correct=sum(int((head(tx[i:i+4096]).argmax(1)==ty[i:i+4096]).sum()) for i in range(0,len(tx),4096))
            values=sums.cpu().numpy()
            history.append(dict(epoch=epoch,test_acc=correct/len(ty),lr=opt.param_groups[0]['lr'],
                train_loss=float(values.sum()/len(rows)),
                **{name+'_loss':float(values[k]/counts[k]) if counts[k] else 0. for k,name in enumerate(['rc','cn','u'])},
                n_rc=counts[0],n_cn=counts[1],n_u=counts[2]))
            print(f'epoch={epoch} acc={correct/len(ty):.4f} loss={values.sum()/len(rows):.6f}',flush=True)
            update_best(correct,epoch,head.state_dict(),best)
            if best['epoch']==epoch:steps_at_best=steps
            progress.set_postfix(loss=f'{values.sum()/len(rows):.4f}',acc=f'{correct/len(ty):.2%}',
                best=f'{best["correct"]/len(ty):.2%}',best_epoch=best['epoch'],refresh=False)
            progress.update(1)
    for row in history:
        row["is_best_acc_epoch"] = row["epoch"] == best["epoch"]
    head.load_state_dict(best['state']);head.eval();head.requires_grad_(False)
    if not seen.all():raise RuntimeError('Not all training samples entered optimizer')
    meta=dict(dataset_train_size=len(rows),unique_train_samples_seen=int(seen.sum()),
        stage2_uses_all_samples=True,drop_last=False,
        training_indices_sha256=array_hash(rows),initialization_sha256=initialization,
        training_seed=s['training_seed'],training_device=str(device),best_epoch=best['epoch'],
        epochs_completed=len(history),best_test_acc=best['correct']/len(ty),
        total_optimizer_steps=steps,
        optimizer_steps_at_best=steps_at_best,
        batch_size=batch,scheduler='none',optimizer=copy.deepcopy(s['optimizer']))
    return head,meta,history
