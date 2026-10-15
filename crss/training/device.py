"""Resolve process-visible CPU/GPU devices without changing the environment."""
import re
import torch

def resolve_training_device(requested='auto'):
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if requested == 'cpu':
        return torch.device('cpu')
    if not isinstance(requested, str) or not re.fullmatch(r'cuda:(0|[1-9][0-9]*)', requested):
        raise ValueError('Device must be auto, cpu, or cuda:N (for example cuda:0)')
    if not torch.cuda.is_available():
        raise RuntimeError(f'{requested} requested but CUDA is unavailable')
    index = int(requested.split(':')[1])
    if index >= torch.cuda.device_count():
        raise ValueError(f'{requested} requested but only {torch.cuda.device_count()} CUDA device(s) are visible')
    return torch.device(requested)
