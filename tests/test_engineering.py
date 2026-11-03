"""Deployment invariants; local HTTP fixtures and synthetic workers, no formal training."""
import copy
import hashlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import pytest
from adam_mf.paths import ROOT
from adam_mf.config import load_config,plan
from adam_mf.assets import download
from adam_mf.runtime.scheduler import selected_gpus,simulation


def test_remaining_experiment_parameters_unchanged():
    import yaml
    before=yaml.safe_load((ROOT/'docs/baseline/experiment.before.yaml').read_text());now=load_config()
    for method in ('adam-no-noise','momentum-no-noise'):
        before['methods'].pop(method)
        assert method not in now['methods']
    before.pop('runtime');now.pop('runtime')
    assert before==now


@pytest.fixture
def http_asset():
    body=bytes(range(256))*1000;events=[];failures=[0];ignore_range=[False]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            start=int(self.headers.get('Range','bytes=0-')[6:].split('-')[0]);events.append(start)
            if failures[0]:failures[0]-=1;self.send_error(503);return
            if ignore_range[0]:start=0
            self.send_response(206 if start else 200)
            if start:self.send_header('Content-Range',f'bytes {start}-{len(body)-1}/{len(body)}')
            self.send_header('Content-Length',str(len(body)-start));self.end_headers();self.wfile.write(body[start:])
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/asset',body,events,failures,ignore_range
    server.shutdown();server.server_close();thread.join()


def test_download_resume_skip_retry_and_hash(http_asset,tmp_path,monkeypatch):
    import adam_mf.assets as a
    monkeypatch.setattr(a.time,'sleep',lambda _:None)
    url,body,events,failures,ignore=http_asset;p=tmp_path/'asset';digest=hashlib.sha256(body).hexdigest()
    part=p.with_name('asset.part');part.write_bytes(body[:12345]);failures[0]=1
    assert download(url,p,len(body),digest)
    assert events==[12345,12345] and p.read_bytes()==body and not part.exists()
    assert not download(url,p,len(body),digest) and len(events)==2
    p.unlink();part.write_bytes(body[:100]);ignore[0]=True
    assert download(url,p,len(body),digest) and p.read_bytes()==body
    p.unlink()
    with pytest.raises(RuntimeError,match='SHA-256'):download(url,p,len(body),'0'*64,retries=1)
    assert not p.exists() and not part.exists()
    failures[0]=5
    with pytest.raises(RuntimeError,match='Download failed'):download(url,p,len(body),digest,retries=2)
    assert not p.exists()


def test_missing_gpu_never_falls_back():
    r=load_config()['runtime'];inventory=[dict(id=i,uuid=f'GPU-{i}') for i in range(3)]
    with pytest.raises(RuntimeError,match='missing.*3'):selected_gpus(r,inventory)
    assert [g['id'] for g in selected_gpus(r,inventory+[dict(id=3),dict(id=9)])]==[0,1,2,3]


def test_simulated_120_mixed_tasks_respect_family_capacity():
    c=load_config();jobs=plan(c)
    # Interleave task families to expose cross-family capacity bugs.
    jobs=jobs[::2]+jobs[1::2];mixed=[v for pair in zip(jobs[:60],jobs[60:]) for v in pair]
    result=simulation(c,mixed);active={};seen=set();completed=[]
    for e in result['events']:
        if e['event']=='start':
            occupants=active.setdefault(e['gpu'],[])
            task=e['job'].split('__')[0]
            assert all(j.split('__')[0]==task for j in occupants)
            occupants.append(e['job']);seen.add(e['gpu'])
            assert len(occupants)<=c['runtime'][task+'_workers_per_gpu']
            assert sum(map(len,active.values()))<=c['runtime']['max_parallel_tasks']
        else:
            active[e['gpu']].remove(e['job']);completed.append(e['job'])
            if not active[e['gpu']]:del active[e['gpu']]
    assert seen=={0,1,2,3} and len(set(completed))==120 and not active


def test_selected_interpreter_has_no_implicit_fallback(tmp_path):
    import os
    script=f'adam_mf_root="{ROOT}"; source "{ROOT}/scripts/python.sh"; printf "%s" "$adam_mf_python"'
    env=dict(os.environ,ADAM_MF_PYTHON='/nonexistent/python')
    assert subprocess.run(['bash','-c',script],env=env,capture_output=True).returncode!=0
    env['ADAM_MF_PYTHON']=str(Path(__import__('sys').executable))
    r=subprocess.run(['bash','-c',script],env=env,text=True,capture_output=True)
    assert r.returncode==0 and r.stdout==env['ADAM_MF_PYTHON']
    # A Conda installation without adam-mf must fail even when another Python works.
    conda=tmp_path/'conda';conda.write_text('#!/bin/sh\nexit 1\n');conda.chmod(0o755)
    env.pop('ADAM_MF_PYTHON');env['PATH']=str(tmp_path)+':/usr/bin:/bin'
    r=subprocess.run(['bash','-c',script],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'setup_env.sh' in r.stderr


def test_production_scheduler_refills_and_stops_on_failure(tmp_path,monkeypatch):
    import adam_mf.runtime.scheduler as s
    import adam_mf.training.engine as engine
    c=load_config();c['runtime']['poll_seconds']=.001;jobs=plan(c)[:12]
    inventory=[dict(id=i,uuid=f'GPU-test-{i}',name='synthetic',total_mb=24000,free_mb=24000) for i in range(4)]
    monkeypatch.setattr(s,'gpu_inventory',lambda:inventory);monkeypatch.setattr(s,'process_memory',lambda:{})
    monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/'logs').mkdir();(tmp_path/'checkpoints').mkdir()
    for failure in (False,True):
        tick=[0];active={};finished=set();events=[]
        class Worker:
            counter=0
            def __init__(self,args,env,**kwargs):
                Worker.counter+=1;self.pid=Worker.counter;self.job=args[args.index('--job')+1];self.gpu=int(env['CUDA_VISIBLE_DEVICES']);self.end=tick[0]+1+self.gpu%2;self.ret=None
                assert self.gpu not in active;active[self.gpu]=self;events.append(self.gpu)
                assert len(active)<=4 and env['HF_HUB_OFFLINE']=='1' and env['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
            def poll(self):
                if self.ret is None and tick[0]>=self.end:
                    self.ret=1 if failure and self.pid==1 else 0;active.pop(self.gpu)
                    if self.ret==0:finished.add(self.job)
                return self.ret
            def send_signal(self,_):self.end=tick[0];self.ret=75;active.pop(self.gpu,None)
            def wait(self):self.send_signal(None);return self.ret
        monkeypatch.setattr(s.subprocess,'Popen',Worker)
        monkeypatch.setattr(s.time,'sleep',lambda _:tick.__setitem__(0,tick[0]+1))
        monkeypatch.setattr(engine,'verify_result',lambda run,j,identity:j.id in finished)
        assert s.schedule(tmp_path,c,jobs,{'synthetic':True}) is (not failure)
        assert events[:4]==[0,1,2,3] and not active
        assert len(events)==(4 if failure else 12)


def test_gpu_lock_is_inherited_by_worker(tmp_path):
    import os,sys
    from adam_mf.runtime.scheduler import reserve_gpu
    gpu=dict(uuid='GPU-lock-test');fd=reserve_gpu(gpu,tmp_path)
    worker=subprocess.Popen([sys.executable,'-u','-c','import sys; print("ready"); sys.stdin.read(1)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,pass_fds=(fd,))
    os.close(fd)
    try:
        assert worker.stdout.readline().strip()=='ready'
        assert reserve_gpu(gpu,tmp_path) is None
    finally:worker.communicate('x',timeout=5)
    fd=reserve_gpu(gpu,tmp_path);assert fd is not None;os.close(fd)


def test_stop_orphan_workers_checks_pid_identity(tmp_path,monkeypatch):
    import signal
    import adam_mf.runtime.launcher as launcher
    from adam_mf.audit import atomic_json
    monkeypatch.setattr(launcher,'local_path',lambda _:tmp_path)
    atomic_json(tmp_path/'current.json',dict(pid=10,process_token='old',run_dir='synthetic'))
    atomic_json(tmp_path/'status.json',dict(status='running',tasks={'valid':dict(status='running',pid=20,process_token='valid'),'stale':dict(status='running',pid=30,process_token='stale')}))
    monkeypatch.setattr(launcher,'alive',lambda record:record.get('process_token')=='valid')
    sent=[];monkeypatch.setattr(launcher.os,'kill',lambda pid,sig:sent.append((pid,sig)))
    launcher.stop();assert sent==[(20,signal.SIGTERM)]


def test_manifest_cannot_override_pinned_hash(tmp_path,monkeypatch):
    import adam_mf.assets as a
    from adam_mf.audit import read_json,atomic_json
    c=load_config();body=b'original data';row=dict(path='data/sst2/train.parquet',kind='download',bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'local_path',lambda v:tmp_path/v);monkeypatch.setattr(a,'catalog',lambda:dict(files=[row]))
    p=tmp_path/row['path'];p.parent.mkdir(parents=True);p.write_bytes(body)
    a.manifests(c);a.verify_assets(c,include_prepared=False)
    p.write_bytes(b'changed! data');m=tmp_path/'data/manifest.json';data=read_json(m)
    data['files'][0]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();atomic_json(m,data)
    with pytest.raises(ValueError,match='Pinned asset'):a.verify_assets(c,include_prepared=False)


def test_preflight_failure_never_reaches_training(tmp_path,monkeypatch):
    import os
    import adam_mf.runtime.launcher as launcher
    import adam_mf.runtime.scheduler as scheduler
    from adam_mf.audit import atomic_json,read_json
    monkeypatch.setattr(launcher,'load_config',lambda _:load_config())
    def fail(*_):raise RuntimeError('Download failed: synthetic HTTP error')
    def train(*_):raise AssertionError('Training must not start after failed download')
    monkeypatch.setattr(launcher,'preflight',fail);monkeypatch.setattr(scheduler,'schedule',train)
    atomic_json(tmp_path/'status.json',dict(status='created',planned=120,completed=0))
    fd=os.open(tmp_path/'pipeline.lock',os.O_CREAT|os.O_RDWR,0o600)
    assert launcher.pipeline(tmp_path,fd)==1
    assert read_json(tmp_path/'status.json')['status']=='failed'
    assert not (tmp_path/'task_plan.json').exists()


def test_checkpoint_pins_original_gpu_uuid(tmp_path):
    from adam_mf.runtime.scheduler import resume_gpu
    from adam_mf.audit import atomic_json
    job=plan(load_config())[0];folder=tmp_path/'checkpoints'/job.id
    assert resume_gpu(tmp_path,job,[]) is None
    folder.mkdir(parents=True);(folder/'latest.pt').write_bytes(b'synthetic marker')
    atomic_json(folder/'gpu_mapping.json',dict(gpu_uuid='GPU-original'))
    assert resume_gpu(tmp_path,job,[dict(uuid='GPU-original',id=2)])=='GPU-original'
    with pytest.raises(ValueError,match='UUID unavailable'):resume_gpu(tmp_path,job,[dict(uuid='GPU-other',id=2)])


def test_lock_covers_cuda_extras_and_installed_dependency_closure():
    import tomllib
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from adam_mf.runtime.environment import dependency_closure
    project=tomllib.loads((ROOT/'pyproject.toml').read_text())
    closure=dependency_closure(project['project']['dependencies'])
    locked={canonicalize_name(Requirement(s).name) for s in (ROOT/'requirements.lock').read_text().splitlines() if s and not s.startswith(('#','--'))}
    assert set(closure)<=locked
    assert {'nvidia-cuda-runtime-cu12','nvidia-cufft-cu12','nvidia-cusolver-cu12','nvidia-cusparse-cu12'}<=set(closure)


def test_worker_refuses_unconfigured_physical_gpu(tmp_path,monkeypatch):
    import sys
    import adam_mf.runtime.worker as worker
    c=load_config();job=plan(c)[0]
    monkeypatch.setattr(worker,'load_config',lambda _:c)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','9')
    monkeypatch.setattr(sys,'argv',['worker','--run',str(tmp_path),'--job',job.id,'--gpu','9'])
    def identity(*_):raise AssertionError('Unconfigured GPU must be rejected before asset/model loading')
    monkeypatch.setattr(worker,'worker_identity',identity)
    with pytest.raises(RuntimeError,match='not configured'):worker.main()
