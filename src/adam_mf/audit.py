import hashlib,json,os,tempfile,csv,time
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

_validation_cache = ContextVar("asset_validation_cache", default=None)

@contextmanager
def hash_session():
    """Reuse hashes only within one validation pass; changed files invalidate entries."""
    token = _validation_cache.set({})
    try:
        yield
    finally:
        _validation_cache.reset(token)

from adam_mf.paths import ROOT

def file_hash(path):
    cache=_validation_cache.get();path=Path(path)
    before=path.stat()
    key=(str(path.absolute()),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    # Filesystems may give two rapid writes identical timestamps. Never reuse
    # hashes of recent writes; newly downloaded/prepared files get a fresh read.
    stable=time.time_ns()-max(before.st_mtime_ns,before.st_ctime_ns)>1_000_000_000
    if cache is not None and stable and key in cache:return cache[key]
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    after=path.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")):raise ValueError(f"File changed during hashing: {path}")
    digest=h.hexdigest()
    if cache is not None and stable:cache[key]=digest
    return digest
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
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths += list((ROOT/'scripts').glob('*.sh'))
    paths += list(ROOT.glob('*.sh')) + [ROOT/p for p in ('pyproject.toml','environment.yml','requirements.lock','config/assets.lock.json')]
    paths += list((ROOT/'config/tokenizer').glob('*.json'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}
def verify_assets(config=None):
    from adam_mf.assets import verify_assets as verify
    return verify(config)
