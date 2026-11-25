from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import yaml
from .io import file_hash, save_json
from .config import ROOT
from .logger import make_logger
from crss.models.model_registry import scale_output

def create_run(cfg,stage):
    from crss.data.noise import noise_name
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:6]
    destination=scale_output(cfg['output']['root'], cfg.get('model_scale', 'base'))/cfg['dataset']['name']/noise_name(cfg)/f"seed_{cfg['seed']}"/(stage+'_'+stamp)
    for directory in ['logs']: (destination/directory).mkdir(parents=True,exist_ok=True)
    (destination/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    packages={}
    for name in ['torch','torchvision','timm','open_clip_torch','numpy','pandas','scikit-learn','PyYAML']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=5).stdout.strip() or None
    except (OSError,subprocess.TimeoutExpired): git=None
    source={str(p.relative_to(ROOT)):file_hash(p) for p in ROOT.rglob('*.py') if not any(part in {'cache','outputs','__pycache__'} for part in p.relative_to(ROOT).parts)}
    save_json(destination/'provenance.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
              packages=packages,git_commit=git,source_sha256=source,seed=cfg['seed'],models=cfg['models']))
    return destination,make_logger(destination/'logs'/'run.log')
