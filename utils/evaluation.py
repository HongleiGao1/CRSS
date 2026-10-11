"""Offline partition diagnostics. Never imported by the trainers."""
import numpy as np


def partition_metrics(clean_mask, noisy_mask, observed, clean_labels):
    clean_mask = np.asarray(clean_mask, bool)
    noisy_mask = np.asarray(noisy_mask, bool)
    truth = np.asarray(observed) == np.asarray(clean_labels)
    if clean_mask.shape != truth.shape or noisy_mask.shape != truth.shape or np.any(clean_mask & noisy_mask):
        raise ValueError('Invalid diagnostic masks')
    result = {}
    for name, mask, actual in [('clean', clean_mask, truth), ('noisy', noisy_mask, ~truth)]:
        tp = int((mask & actual).sum())
        precision = tp / mask.sum() if mask.any() else 0.
        recall = tp / actual.sum() if actual.any() else 0.
        result.update({name+'_precision':float(precision), name+'_recall':float(recall),
                       name+'_f1':float(2*precision*recall/(precision+recall)) if precision+recall else 0.})
    result.update(macro_f1=(result['clean_f1']+result['noisy_f1'])/2,
                  coverage=float((clean_mask | noisy_mask).mean()))
    return result
