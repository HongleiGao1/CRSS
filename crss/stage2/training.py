"""All-sample Stage II. No anchor selection or temperature fitting is invoked."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from crss.io import _atomic_json,git_version
from crss.training.linear import train_head,collect_logits,evaluate
from crss.utils.io import array_hash,file_hash,object_hash,read_npz,save_npz
from .config import METHOD_VERSION,DEFAULTS,absolute,format_rho_tag

REQUIRED=('best_head.pt','metrics.json','metrics.csv','metadata.json','history.csv','test_predictions.npz')


def load_training_features(args,result,partition_meta):
    path=Path(partition_meta['feature_paths']['vit'])
    if not path.is_file():
        from crss.data.cached_inputs import _feature_path
        path=_feature_path(dict(feature_paths=partition_meta['feature_paths'],
            feature_source_hashes=partition_meta['identity']['feature_source_hashes']),
            'vit',args.dataset,args.model_scale,absolute(args.cache_root))
    train=read_npz(path);tm=json.loads(str(train['metadata']))
    expected=partition_meta['identity']['feature_source_hashes']['vit']
    if array_hash(train['features'])!=expected or str(train['feature_sha256'])!=expected:
        raise ValueError('Training feature hash differs from Stage I evidence')
    if not np.array_equal(train['indices'],result.indices) or not np.allclose(np.linalg.norm(train['features'],axis=1),1,atol=2e-4):
        raise ValueError('Training sample order/normalization mismatch')
    test_path=path.with_name('test_l2.npz')
    if not test_path.is_file():raise FileNotFoundError('Missing frozen test cache: '+str(test_path)+'; use run_extract.py --split test')
    test=read_npz(test_path);sm=json.loads(str(test['metadata']))
    if sm.get('source_train_sha256')!=expected or any(sm.get(k)!=tm.get(k) for k in ('model_identity','preprocessing','preprocessing_source','checkpoint')):
        raise ValueError('Test backbone/checkpoint/preprocessing differs from train')
    x=test['features'];labels=test['clean_labels']
    if x.ndim!=2 or x.shape[1]!=train['features'].shape[1] or not np.isfinite(x).all() or array_hash(x)!=str(test['feature_sha256']) or not np.allclose(np.linalg.norm(x,axis=1),1,atol=2e-4):raise ValueError('Test feature integrity/dimension mismatch')
    if len(x)!=10000 or labels.shape!=(len(x),) or not np.array_equal(test['indices'],np.arange(len(x))) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>=result.usc_distribution.shape[1])):
        raise ValueError('CIFAR test order/count/label range mismatch')
    return train['features'],x,labels,dict(train_feature_path=str(path),test_feature_path=str(test_path),
        train_feature_sha256=expected,test_feature_sha256=str(test['feature_sha256']),train_metadata=tm,test_metadata=sm)


def training_config(args):
    return dict(stage2=dict(training_seed=args.training_seed,batch_size=args.batch_size,max_epochs=args.max_epochs,
        early_stopping=dict(patience=args.patience),q_gce=args.q_gce,
        optimizer=dict(name='AdamW',lr=args.lr,weight_decay=args.weight_decay)))


def complete(directory,identity):
    done=directory/'DONE.json'
    if not done.is_file():return False
    try:
        m=json.loads(done.read_text(encoding='utf-8'))
        return m.get('identity')==identity and all((directory/k).is_file() and file_hash(directory/k)==m.get('file_hashes',{}).get(k) for k in REQUIRED)
    except (OSError,ValueError,TypeError,AttributeError):return False


def train_stage2_linear_head(args,result,partition_meta,partition_path):
    x,tx,ty,provenance=load_training_features(args,result,partition_meta)
    expected=partition_meta['partition_hash'];core=partition_meta['source_core_hash']
    from .artifacts import partition_hash
    loaded=partition_hash(result,args.rho,args.tau,core)
    if loaded!=expected:raise ValueError('Stage I/II partition hash mismatch')
    partition_file_hash=file_hash(partition_path)
    if partition_file_hash!=partition_meta['file_sha256']:raise ValueError('Partition file changed before training')
    cfg=training_config(args)
    identity=dict(task=partition_meta['identity'],rho=args.rho,tau=args.tau,partition_hash=expected,
        source_core_hash=core,method_version=METHOD_VERSION,config=cfg,target_expert='vit',device=args.device,
        test_feature_sha256=provenance['test_feature_sha256'],test_labels_sha256=array_hash(ty))
    directory=absolute(args.output_root)/args.dataset/args.model_scale/args.noise/f'seed_{args.seed}'/format_rho_tag(args.rho)/('run_'+object_hash(identity)[:20])
    if args.resume and complete(directory,identity):
        print('REUSE_STAGE2_COMPLETE',flush=True)
        return json.loads((directory/'metrics.json').read_text(encoding='utf-8')),directory
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'DONE.json').unlink(missing_ok=True)
    t=time.perf_counter()
    head,meta,history=train_head(x,result.to_stage2_bundle(),tx,ty,cfg,args.device)
    seconds=time.perf_counter()-t
    if file_hash(partition_path)!=partition_file_hash:raise ValueError('Partition file changed during training')
    logits=collect_logits(head,tx);metrics,p=evaluate(logits,ty)
    if not np.isclose(metrics['accuracy'],meta['best_test_acc'],rtol=0,atol=1e-12):raise RuntimeError('Restored checkpoint does not match best test accuracy')
    metrics.update(raw_ece15=metrics['ece15'],best_epoch=meta['best_epoch'],stop_epoch=meta['stop_epoch'],
        epochs_completed=len(history),training_loss=history[meta['best_epoch']-1]['train_loss'],
        optimizer_steps_at_best=meta['optimizer_steps_at_best'],training_time_seconds=seconds)
    meta.update(identity=identity,method_version=METHOD_VERSION,rho=args.rho,tau=args.tau,target_expert='vit',
        config_hash=object_hash(cfg),effective_config=cfg,
        partition_path=str(partition_path),partition_hash=expected,expected_partition_hash=expected,
        loaded_partition_hash=loaded,training_partition_hash=loaded,partition_hash_used=loaded,source_core_hash=core,
        partition_file_sha256=partition_file_hash,q_gce=args.q_gce,lr=args.lr,weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,patience=args.patience,canonical_run=args.canonical_run,
        changed_parameters=args.changed_parameters,mandatory_stage3=False,git_commit=git_version(),
        dataset=args.dataset,scale=args.model_scale,noise=args.noise,seed=args.seed,
        best_test_accuracy=meta['best_test_acc'],**provenance)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},metadata=meta),directory/'best_head.pt')
    _atomic_json(directory/'metrics.json',metrics);_atomic_json(directory/'metadata.json',meta)
    with (directory/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics));w.writeheader();w.writerow(metrics)
    with (directory/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    save_npz(directory/'test_predictions.npz',indices=np.arange(len(ty)),clean_labels=ty,
             logits=logits,probabilities=p,predictions=logits.argmax(1))
    _atomic_json(directory/'DONE.json',dict(identity=identity,file_hashes={k:file_hash(directory/k) for k in REQUIRED}))
    return metrics,directory
