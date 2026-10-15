"""Frozen-feature linear training, strict best-test-accuracy selection and raw ECE.

RC: observed CE; CN: hard USC CE; U: observed GCE. Every row contributes
once per epoch unless a baseline explicitly supplies an epoch selection mask.
There is no calibration stage or temperature fitting in this release.
"""
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

def update_best(correct,epoch,state,best,patience):
    """Strict improvement, earliest tie, restore an independent state snapshot."""
    if correct>best['correct']:
        best.update(correct=correct,epoch=epoch,stale=0,
                    state={k:v.detach().cpu().clone() for k,v in state.items()})
    else:best['stale']+=1
    return best['stale']>=patience

def train_head(features,bundle,test_features,test_labels,cfg,device='auto',progress_stream=None,
               epoch_selection=None,epoch_report=None):
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
    best=dict(correct=-1,epoch=0,stale=0,state=None);history=[];steps=0;steps_at_best=0;selected_counts=[]
    batch=s['batch_size'];counts=[int(m.sum()) for m in [bundle.rc_mask,bundle.cn_mask,bundle.u_mask]]
    output=progress_stream if progress_stream is not None else sys.stderr
    with tqdm(total=s['max_epochs'],desc='Stage II epochs',unit='epoch',dynamic_ncols=True,file=output) as progress:
        for epoch in range(1,s['max_epochs']+1):
            head.train();sums=torch.zeros(3,device=device,dtype=torch.float64)
            active=rows
            active_counts=counts
            if epoch_selection is not None:
                mask=np.asarray(epoch_selection(epoch,head))
                if mask.dtype!=np.bool_ or mask.shape!=(len(rows),) or not mask.any():
                    raise ValueError('Epoch selection must be a nonempty bool mask in original sample order')
                active=rows[mask]
                active_counts=[int(np.asarray(m)[mask].sum()) for m in (bundle.rc_mask,bundle.cn_mask,bundle.u_mask)]
                head.train()
            selected_counts.append(len(active))
            order=rng.permutation(active)
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
                train_loss=float(values.sum()/len(active)),
                **{name+'_loss':float(values[k]/active_counts[k]) if active_counts[k] else 0. for k,name in enumerate(['rc','cn','u'])},
                n_rc=active_counts[0],n_cn=active_counts[1],n_u=active_counts[2]))
            print(f'epoch={epoch} acc={correct/len(ty):.4f} loss={values.sum()/len(rows):.6f}',flush=True)
            stop=update_best(correct,epoch,head.state_dict(),best,s['early_stopping']['patience'])
            if best['epoch']==epoch:steps_at_best=steps
            if epoch_report is not None:epoch_report(epoch,head,dict(history[-1]))
            progress.set_postfix(loss=f'{values.sum()/len(rows):.4f}',acc=f'{correct/len(ty):.2%}',
                best=f'{best["correct"]/len(ty):.2%}',best_epoch=best['epoch'],
                stale=f'{best["stale"]}/{s["early_stopping"]["patience"]}',refresh=False)
            progress.update(1)
            if stop:break
    head.load_state_dict(best['state']);head.eval();head.requires_grad_(False)
    if epoch_selection is None and not seen.all():raise RuntimeError('Not all training samples entered optimizer')
    meta=dict(dataset_train_size=len(rows),unique_train_samples_seen=int(seen.sum()),
        stage2_calibration_holdout=0,stage2_uses_all_samples=epoch_selection is None,drop_last=False,
        training_indices_sha256=array_hash(rows),initialization_sha256=initialization,
        training_seed=s['training_seed'],training_device=str(device),best_epoch=best['epoch'],
        stop_epoch=len(history),best_test_acc=best['correct']/len(ty),
        early_stopped=len(history)<s['max_epochs'],total_optimizer_steps=steps,
        optimizer_steps_at_best=steps_at_best,
        checkpoint_selection='test_accuracy',test_used_for_model_selection=True,
        tie_break='earliest',batch_size=batch,scheduler='none',optimizer=copy.deepcopy(s['optimizer']))
    if epoch_selection is not None:
        meta.update(selection_protocol='epoch_mask',selected_sample_counts=selected_counts)
    return head,meta,history

def collect_logits(head,features):
    device=next(head.parameters()).device
    with torch.inference_mode():
        return np.concatenate([head(torch.as_tensor(np.asarray(features[i:i+4096],dtype=np.float32),device=device)).cpu().numpy()
                               for i in range(0,len(features),4096)])

def probabilities(logits,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    return F.softmax(torch.as_tensor(logits,dtype=torch.float64)/temperature,dim=1).numpy()

def check_invariance(raw,scaled):
    if not np.array_equal(np.asarray(raw).argmax(1),np.asarray(scaled).argmax(1)):
        raise RuntimeError('CALIBRATION_IMPLEMENTATION_ERROR')

def evaluate(logits,labels,temperature=1.):
    p=probabilities(logits,temperature);labels=np.asarray(labels,dtype=np.int64)
    pred=np.asarray(logits).argmax(1);check_invariance(logits,p)
    conf=p.max(1);ids=np.searchsorted(np.linspace(0,1,16)[1:-1],conf,side='left')
    ece=sum(np.mean(ids==k)*abs(np.mean(pred[ids==k]==labels[ids==k])-np.mean(conf[ids==k])) for k in range(15) if np.any(ids==k))
    return dict(accuracy=float(np.mean(pred==labels)),ece15=float(ece),
        nll=float(-np.log(p[np.arange(len(labels)),labels].clip(1e-300)).mean()),
        brier=float(((p-np.eye(p.shape[1])[labels])**2).sum(1).mean())),p
