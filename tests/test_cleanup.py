"""Regression checks for coordinator trust boundaries and grouped finite checks."""
import copy
from pathlib import Path
import pytest
import torch
from adam_mf.audit import atomic_json,json_hash,file_hash,hash_session
from adam_mf.config import load_config,plan,validate


def test_hash_session_reuses_reads_and_invalidates_changes(tmp_path,monkeypatch):
    import builtins
    p=tmp_path/'asset';p.write_bytes(b'abc');opened=[];original=builtins.open
    __import__('time').sleep(1.1)
    def tracked(path,*a,**kw):
        if Path(path)==p:opened.append(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(builtins,'open',tracked)
    with hash_session():
        old=file_hash(p);assert file_hash(p)==old and len(opened)==1
        p.write_bytes(b'def');assert file_hash(p)!=old and len(opened)==2
    file_hash(p);assert len(opened)==3


def test_worker_uses_parent_identity_without_hashing(tmp_path,monkeypatch):
    import adam_mf.runtime.state as state
    import adam_mf.runtime.scheduler as scheduler
    import adam_mf.assets as assets
    c=load_config();job=plan(c)[0];gpu=dict(id=0,uuid='GPU-correct')
    ident=dict(config_hash=json_hash(c),gpus=[gpu]);expected=json_hash(ident)
    manifest=dict(identity=ident,planned_tasks=[job.values()])
    atomic_json(tmp_path/'run_manifest.json',manifest)
    monkeypatch.setattr(scheduler,'selected_gpus',lambda _: [gpu])
    monkeypatch.setattr(assets,'catalog',lambda:dict(files=[]))
    def forbidden(*a):raise AssertionError('Worker must not compute full identity')
    monkeypatch.setattr(state,'identity',forbidden)
    assert state.worker_identity(tmp_path,c,job,0,expected)==ident
    with pytest.raises(ValueError,match='identity'):state.worker_identity(tmp_path,c,job,0,'wrong')
    changed=copy.deepcopy(c);changed['methods'][job.method]['cv']['learning_rate']*=2
    with pytest.raises(ValueError,match='snapshot'):state.worker_identity(tmp_path,changed,job,0,expected)
    with pytest.raises(ValueError,match='outside'):state.worker_identity(tmp_path,c,plan(c)[1],0,expected)
    monkeypatch.setattr(scheduler,'selected_gpus',lambda _: [dict(gpu,uuid='wrong')])
    with pytest.raises(ValueError,match='UUID'):state.worker_identity(tmp_path,c,job,0,expected)


def test_resume_still_performs_full_identity_check(tmp_path,monkeypatch):
    import adam_mf.runtime.state as state
    atomic_json(tmp_path/'run_manifest.json',dict(identity={'asset_hash':'original'}))
    calls=[]
    def identity(c):calls.append(c);return dict(asset_hash='changed')
    monkeypatch.setattr(state,'identity',identity)
    with pytest.raises(ValueError,match='Resume refused'):state.check_identity(tmp_path,{'config':'test'})
    assert calls==[{'config':'test'}]


def test_stpex_rejects_last_bad_parameter_before_any_update():
    from adam_mf.optimizers.stpex import Optimizer
    params=[torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    opt=Optimizer(params,'stpex-mf-momentum-bias-b2lt',.1,(.9,.999),.01);opt.freeze_scale()
    before=copy.deepcopy(opt.state_dict())
    for p in params:p.grad=torch.ones_like(p)
    params[-1].grad[-1]=float('nan')
    with pytest.raises(FloatingPointError):opt.step()
    for p in params:assert torch.equal(p,torch.ones_like(p))
    for key,state in before['state'].items():
        for name,value in state.items():
            current=opt.state_dict()['state'][key][name]
            assert torch.equal(value,current) if torch.is_tensor(value) else value==current


def test_configurable_family_capacity():
    c=load_config();c['runtime'].update(nlp_workers_per_gpu=2,max_parallel_tasks=8)
    validate(c)
    from adam_mf.runtime.scheduler import simulation
    result=simulation(c,[j for j in plan(c) if j.task=='nlp']);active={};peak=0
    for event in result['events']:
        gpu=event['gpu'];active[gpu]=active.get(gpu,0)+(1 if event['event']=='start' else -1)
        assert 0<=active[gpu]<=2
        peak=max(peak,sum(active.values()))
    assert peak==8
    for key in ('cv_workers_per_gpu','nlp_workers_per_gpu'):
        bad=copy.deepcopy(c);bad['runtime'][key]=0
        with pytest.raises(ValueError):validate(bad)


def test_training_code_identity_excludes_tests_and_docs():
    from adam_mf.audit import code_hashes
    hashes=code_hashes()
    assert 'src/adam_mf/training/engine.py' in hashes
    assert not any(p.startswith(('tests/','docs/')) for p in hashes)


def test_nlp_scheduler_shares_one_gpu_lock(tmp_path,monkeypatch):
    import adam_mf.runtime.scheduler as s
    import adam_mf.training.engine as engine
    c=load_config();c['runtime'].update(gpu_ids=[0],max_parallel_tasks=2)
    jobs=[j for j in plan(c) if j.task=='nlp'][:4]
    gpu=dict(id=0,uuid='GPU-test',name='synthetic',total_mb=24000,free_mb=24000)
    monkeypatch.setattr(s,'gpu_inventory',lambda:[gpu]);monkeypatch.setattr(s,'process_memory',lambda:{})
    monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/'logs').mkdir()
    tick=[0];active={};finished=set();locks=[];peak=[0]
    class Worker:
        def __init__(self,args,pass_fds,**kwargs):
            self.job=args[args.index('--job')+1];self.pid=100+len(locks);self.end=tick[0]+1;self.ret=None
            locks.append(pass_fds[0]);active[self.job]=self;peak[0]=max(peak[0],len(active))
        def poll(self):
            if self.ret is None and tick[0]>=self.end:
                self.ret=0;active.pop(self.job);finished.add(self.job)
            return self.ret
        def send_signal(self,_):self.ret=75;active.pop(self.job,None)
        def wait(self):return self.ret
    monkeypatch.setattr(s.subprocess,'Popen',Worker)
    monkeypatch.setattr(s.time,'sleep',lambda _:tick.__setitem__(0,tick[0]+1))
    monkeypatch.setattr(engine,'verify_result',lambda run,j,identity:j.id in finished)
    assert s.schedule(tmp_path,c,jobs,{'synthetic':True})
    assert peak[0]==2 and locks[0]==locks[1] and len(finished)==4
    fd=s.reserve_gpu(gpu,tmp_path/'logs/gpu_locks');assert fd is not None
    __import__('os').close(fd)


def test_prepared_content_reused_only_for_identical_bytes(tmp_path,monkeypatch):
    import numpy as np
    import adam_mf.assets as assets
    import adam_mf.data.preprocessing as prep
    from adam_mf.audit import array_hash
    p=tmp_path/'cache.npz';values=np.arange(3)
    np.savez(p,values=values)
    row=dict(kind='prepared',content={'values':dict(shape=[3],dtype=str(values.dtype),sha256=array_hash(values))})
    monkeypatch.setattr(assets,'catalog',lambda:dict(files=[row]));monkeypatch.setattr(assets,'asset_path',lambda *_:p)
    calls=[];original=np.load
    def load(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(np,'load',load)
    with hash_session():
        prep.verify_prepared_content({});prep.verify_prepared_content({});assert len(calls)==1
        np.savez(p,values=values+1)
        with pytest.raises(ValueError,match='content mismatch'):prep.verify_prepared_content({})
    assert len(calls)==2


@pytest.mark.parametrize('error',[KeyboardInterrupt(),SystemExit(9)])
def test_pipeline_control_exit_propagates_and_releases_lock(tmp_path,monkeypatch,error):
    import os
    import adam_mf.runtime.launcher as launcher
    from adam_mf.audit import read_json
    monkeypatch.setattr(launcher,'load_config',lambda _:load_config())
    def stop(*_):raise error
    monkeypatch.setattr(launcher,'preflight',stop)
    atomic_json(tmp_path/'status.json',dict(status='created'))
    fd=os.open(tmp_path/'lock',os.O_CREAT|os.O_RDWR,0o600)
    with pytest.raises(type(error)):launcher.pipeline(tmp_path,fd)
    assert read_json(tmp_path/'status.json')['status']=='paused'
    with pytest.raises(OSError):os.fstat(fd)
