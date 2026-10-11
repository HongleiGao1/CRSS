"""Offline GT diagnostics. Never called by evidence or training objectives."""
import numpy as np


def evaluate_crss_partition(result, clean_labels=None, rho=.6):
    n=len(result.indices);rc,cn,u=result.rc_mask,result.cn_mask,result.u_mask
    out=dict(sample_count=n,rho=rho,candidate_count=int(result.reliable_candidate_mask.sum()),
             candidate_fraction=float(result.reliable_candidate_mask.mean()),decided_coverage=float((rc|cn).mean()))
    out.update(rho_requested=rho,rho_actual_candidate_fraction=out['candidate_fraction'],decided_fraction=out['decided_coverage'])
    for name,mask in [('rc',rc),('cn',cn),('u',u)]:
        out[name+'_count']=int(mask.sum());out[name+'_fraction']=float(mask.mean())
    keys=('rc_observed_accuracy','cn_correction_accuracy','cn_noisy_precision','u_observed_accuracy',
          'u_usc_accuracy','clean_precision','clean_recall','clean_f1','noisy_precision','noisy_recall',
          'noisy_f1','selective_macro_f1','selective_balanced_accuracy')
    out.update({k:None for k in keys});out['gt_diagnostics_available']=clean_labels is not None
    if clean_labels is None:return out
    gt=np.asarray(clean_labels)
    if gt.shape!=(n,):raise ValueError('Diagnostic GT alignment mismatch')
    clean=result.observed_labels==gt;noisy=~clean
    def average(values,mask):return float(values[mask].mean()) if mask.any() else None
    out.update(rc_observed_accuracy=average(clean,rc),cn_correction_accuracy=average(result.usc_prediction==gt,cn),
        cn_noisy_precision=average(noisy,cn),u_observed_accuracy=average(clean,u),u_usc_accuracy=average(result.usc_prediction==gt,u))
    def scores(pred,truth,prefix):
        tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
        p=tp/(tp+fp) if tp+fp else None;r=tp/(tp+fn) if tp+fn else None
        f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        out.update({prefix+'_tp':tp,prefix+'_fp':fp,prefix+'_fn':fn,prefix+'_precision':p,
                    prefix+'_recall':r,prefix+'_f1':f})
        return f,r
    scores(rc,clean,'clean');scores(cn,noisy,'noisy')
    decided=rc|cn
    if decided.any():
        f1,rec1=scores(rc[decided],clean[decided],'selective_clean')
        f2,rec2=scores(cn[decided],noisy[decided],'selective_noisy')
        out['selective_macro_f1']=float(np.mean([x for x in (f1,f2) if x is not None])) if any(x is not None for x in (f1,f2)) else None
        out['selective_balanced_accuracy']=float(np.mean([x for x in (rec1,rec2) if x is not None])) if any(x is not None for x in (rec1,rec2)) else None
    return out
