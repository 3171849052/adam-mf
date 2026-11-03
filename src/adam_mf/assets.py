"""Pinned, resumable downloads and offline verification. No training-time network."""
import contextlib
import fcntl
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from adam_mf.paths import ROOT,local_path
from adam_mf.audit import read_json,atomic_json,file_hash

CATALOG=ROOT/'config/assets.lock.json'


def catalog():return read_json(CATALOG)


def asset_path(row,config):
    path=row['path'];p=config['paths']
    for prefix,key in [('data/preprocessed','preprocessed'),('assets/matrices','matrices'),('model/vit-tiny','cv_model'),('model/bert-tiny','nlp_model'),('data','data')]:
        if path.startswith(prefix+'/'):return local_path(p[key])/path[len(prefix)+1:]
    raise ValueError(f'Unknown asset path {path}')


def matches(path,row):
    return path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and file_hash(path)==row['sha256']


def download(url,target,size,sha256,retries=3):
    """Keep .part bytes on network errors; promote only verified full bytes."""
    target=Path(target);row=dict(bytes=size,sha256=sha256)
    if matches(target,row):return False
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_name(target.name+'.part');last=None
    for attempt in range(retries):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset>size:part.unlink();offset=0
            if offset<size:
                req=urllib.request.Request(url,headers={'User-Agent':'adam-mf/0.1','Accept-Encoding':'identity',**({'Range':f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(req,timeout=60) as response:
                    status=response.status
                    if status==206:
                        content_range=response.headers.get('Content-Range','')
                        if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{size}'):
                            raise ValueError(f'Invalid Content-Range: {content_range}')
                    elif status==200:offset=0
                    else:raise ValueError(f'Unexpected HTTP status {status}')
                    length=response.headers.get('Content-Length')
                    if length is not None and int(length)!=size-offset:raise ValueError('Download size header mismatch')
                    with part.open('ab' if offset else 'wb') as output:
                        while True:
                            block=response.read(1024*1024)
                            if not block:break
                            output.write(block)
                            if output.tell()>size:raise ValueError('Download exceeds pinned size')
                        output.flush();os.fsync(output.fileno())
            if not matches(part,row):
                if part.exists() and part.stat().st_size>=size:part.unlink()
                raise ValueError(f'Download SHA-256/size mismatch: {target.name}')
            os.replace(part,target);return True
        except (OSError,ValueError,urllib.error.URLError) as error:
            last=error
            if isinstance(error,urllib.error.HTTPError) and error.code==416:part.unlink(missing_ok=True)
            if attempt+1<retries:time.sleep(min(2**attempt,4))
    raise RuntimeError(f'Download failed after {retries} attempts: {url}: {last}') from last


def extract_member(archive,member,target,row):
    if matches(target,row):return
    target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as tar:
        info=tar.getmember(member)
        if not info.isfile() or info.size!=row['bytes']:raise ValueError('Unsafe/mismatched archive member')
        fd,tmp=tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as output,tar.extractfile(info) as source:shutil.copyfileobj(source,output)
            if not matches(Path(tmp),row):raise ValueError(f'Archive member hash mismatch: {member}')
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


@contextlib.contextmanager
def asset_lock(config):
    logs=local_path(config['paths']['logs']);logs.mkdir(parents=True,exist_ok=True)
    with (logs/'assets.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);yield


def manifests(config,prepared=False,groups=None):
    c=catalog()
    for prefix,key in [('data/','data'),('model/','model'),('assets/matrices/','matrices')]:
        if groups is not None and key not in groups:continue
        rows=[]
        for row in c['files']:
            if not row['path'].startswith(prefix):continue
            path=asset_path(row,config)
            if row['kind'] in ('prepared','generated-matrix'):
                if row['kind']=='prepared' and not prepared:continue
                if not path.is_file():raise FileNotFoundError(path)
            elif not matches(path,row):raise ValueError(f'Pinned asset mismatch: {path}')
            rows.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=file_hash(path) if row['kind'] in ('prepared','generated-matrix') else row['sha256']))
        folder=local_path(config['paths']['data'] if key=='data' else config['paths']['matrices'] if key=='matrices' else str(local_path(config['paths']['cv_model']).parent.relative_to(ROOT)))
        atomic_json(folder/'manifest.json',dict(catalog_sha256=file_hash(CATALOG),files=rows,total_bytes=sum(r['bytes'] for r in rows)))


def fetch_assets(config):
    c=catalog()
    with asset_lock(config):
        from adam_mf.privacy.matrix_design import prepare_matrices
        prepare_matrices(local_path(config['paths']['matrices']))
        for row in c['files']:
            target=asset_path(row,config)
            if row['kind'] in ('prepared','generated-matrix') or matches(target,row):continue
            print(f'Prepare {target.relative_to(ROOT)}',flush=True)
            if row['kind']=='download':download(row['url'],target,row['bytes'],row['sha256'])
            elif row['kind']=='bundled':
                source=local_path(row['source'])
                if not matches(source,row):raise ValueError(f'Bundled tokenizer mismatch: {source}')
                # Use the same validated, atomic publication rule as extraction.
                target.parent.mkdir(parents=True,exist_ok=True)
                fd,tmp=tempfile.mkstemp(dir=target.parent)
                try:
                    with os.fdopen(fd,'wb') as out,source.open('rb') as inp:shutil.copyfileobj(inp,out)
                    os.replace(tmp,target)
                finally:
                    if os.path.exists(tmp):os.unlink(tmp)
            elif row['kind']=='cifar-member':
                a=c['cifar_archive'];archive=local_path(config['paths']['data'])/'.downloads/cifar-100-python.tar.gz'
                for url in [a['url']]+a.get('mirrors',[]):
                    try:
                        download(url,archive,a['bytes'],a['sha256']);break
                    except RuntimeError:
                        if url==([a['url']]+a.get('mirrors',[]))[-1]:raise
                extract_member(archive,row['member'],target,row)
        manifests(config,prepared=False)
    return verify_assets(config,include_prepared=False)


def verify_assets(config=None,include_prepared=True):
    if config is None:
        from adam_mf.config import load_config
        config=load_config()
    hashes={};c=catalog()
    if any(r['kind']=='generated-matrix' for r in c['files']):
        from adam_mf.privacy.matrix_design import verify_all
        verify_all(config)
    for row in c['files']:
        path=asset_path(row,config)
        if row['kind'] in ('prepared','generated-matrix'):
            if row['kind']=='prepared' and not include_prepared:continue
            if not path.is_file() or path.is_symlink():raise ValueError(f'Missing prepared asset: {path}')
        elif not matches(path,row):raise ValueError(f'Pinned asset hash/size mismatch: {path}')
        hashes[str(path.relative_to(ROOT))]=file_hash(path) if row['kind'] in ('prepared','generated-matrix') else row['sha256']
    if include_prepared:
        from adam_mf.data.preprocessing import verify_prepared_content
        verify_prepared_content(config)
    # Manifest is an audit record, never the trust root; expected files come from catalog.
    groups=[(local_path(config['paths']['data']),'data/'),(local_path(config['paths']['cv_model']).parent,'model/'),(local_path(config['paths']['matrices']),'assets/matrices/')]
    for folder,prefix in groups:
        m=folder/'manifest.json';data=read_json(m)
        actual={r['path']:r for r in data['files']}
        all_names={str(asset_path(r,config).relative_to(ROOT)) for r in c['files'] if r['path'].startswith(prefix)}
        required={str(asset_path(r,config).relative_to(ROOT)) for r in c['files'] if r['path'].startswith(prefix) and (include_prepared or r['kind']!='prepared')}
        if len(actual)!=len(data['files']) or not required<=set(actual)<=all_names or data.get('catalog_sha256')!=file_hash(CATALOG):raise ValueError(f'Manifest/catalog mismatch: {m}')
        for name in required:
            row=actual[name];path=ROOT/name
            if row['sha256']!=hashes[name] or row['bytes']!=path.stat().st_size:raise ValueError(f'Manifest hash/size mismatch: {path}')
        hashes[str(m.relative_to(ROOT))]=file_hash(m)
    return hashes
