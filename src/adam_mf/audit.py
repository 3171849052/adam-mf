import hashlib,json,os,tempfile,csv
from pathlib import Path
from adam_mf.paths import ROOT

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def json_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text())
def atomic_json(p,value): atomic_text(p,json.dumps(value,indent=2,allow_nan=False)+'\n')
def atomic_text(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.')
    try:
        with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def write_csv(p,rows,fields=None):
    import io
    rows=list(rows);out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_text(p,out.getvalue())
def code_hashes():
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths += list((ROOT/'scripts').glob('*.sh'))
    paths += [ROOT/p for p in ('start.sh','status.sh','stop.sh','pyproject.toml','environment.yml')]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}
def verify_assets():
    hashes={}
    for folder in ('data','model','assets/matrices'):
        manifest=ROOT/folder/'manifest.json'
        data=read_json(manifest)
        hashes[str(manifest.relative_to(ROOT))]=file_hash(manifest)
        for item in data['files']:
            p=ROOT/item['path']
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:
                raise ValueError(f'Asset hash mismatch: {p}')
            hashes[item['path']]=item['sha256']
    return hashes
