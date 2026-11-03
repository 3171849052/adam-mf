import os
from pathlib import Path
from adam_mf.audit import atomic_json,read_json,json_hash,code_hashes,verify_assets
from adam_mf.paths import ROOT,local_path

from adam_mf.runtime.management import process_token,alive

def environment():
    import sys,platform,importlib.metadata as m
    names=('torch','torchvision','timm','transformers','opacus','numpy','scipy','scikit-learn','pandas','pyarrow','safetensors','PyYAML','matplotlib')
    return dict(python=platform.python_version(),packages={n:m.version(n) for n in names})

def identity(config,hashes=None):
    if hashes is None:hashes=verify_assets(config)
    required=[];p=config['paths']
    required += [str(local_path(p['data'])/f'cifar100/cifar-100-python/{name}') for name in ('train','test','meta')]
    required += [str(local_path(p['data'])/f'sst2/{name}.parquet') for name in ('train','validation')]
    required += [str(local_path(p['preprocessed'])/name) for name in ('cv_split.npz','nlp_split.npz','nlp_tokens_128.pt','nlp_official_tokens_128.pt')]
    required += [str(local_path(p['cv_model'])/'model.safetensors')]
    required += [str(local_path(p['nlp_model'])/name) for name in ('pytorch_model.bin','config.json','vocab.txt','tokenizer.json','tokenizer_config.json')]
    required += [str(local_path(p['matrices'])/f'{w}_{m}_T{n}{suffix}') for n in (250,310) for w,m in [('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')] for suffix in ('.npz','.json')]
    for name in required:
        if str(Path(name).relative_to(ROOT)) not in hashes:raise ValueError(f'Configured input is outside the asset manifests: {name}')
    models={task:json_hash({p:h for p,h in hashes.items() if p.startswith(config['paths'][task+'_model']+'/')}) for task in ('cv','nlp')}
    if any(h==json_hash({}) for h in models.values()):raise ValueError('Model paths do not correspond to manifested assets')
    from adam_mf.runtime.scheduler import selected_gpus
    devices=[{k:g[k] for k in ('id','uuid','name','total_mb','driver')} for g in selected_gpus(config['runtime'])]
    return dict(gpus=devices,config_hash=json_hash(config),code_hash=json_hash(code_hashes()),asset_hash=json_hash(hashes),model_hashes=models,environment=environment())

def check_identity(run,config):
    manifest=read_json(run/'run_manifest.json');current=identity(config)
    if current!=manifest['identity']:raise ValueError('Resume refused: configuration, code, environment, model, asset hashes or GPU identities changed')
    return current


def worker_identity(run,config,job,gpu_id,expected_hash):
    """Consume the coordinator's verified identity without rereading asset bytes."""
    manifest=read_json(run/'run_manifest.json');ident=manifest['identity']
    if not ident or json_hash(ident)!=expected_hash:raise ValueError('Worker experiment identity mismatch')
    if ident['config_hash']!=json_hash(config):raise ValueError('Worker configuration snapshot mismatch')
    if job.values() not in manifest['planned_tasks']:raise ValueError('Worker task is outside this run')
    from adam_mf.runtime.scheduler import selected_gpus
    actual=next(g for g in selected_gpus(config['runtime']) if g['id']==gpu_id)
    expected=next(g for g in ident['gpus'] if g['id']==gpu_id)
    if actual['uuid']!=expected['uuid']:raise ValueError('Worker physical GPU UUID mismatch')
    from adam_mf.assets import catalog,asset_path
    for row in catalog()['files']:
        path=asset_path(row,config)
        if not path.is_file() or path.is_symlink():raise ValueError(f'Missing worker asset: {path}')
    return ident
