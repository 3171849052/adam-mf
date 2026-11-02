from config_helpers import load_config
import pytest
from adam_mf.runtime.scheduler import can_launch
from adam_mf.reporting.statistics import statistics

def test_memory_and_family_capacity():
    r=dict(cv_workers_per_gpu=1,nlp_workers_per_gpu=1,max_tasks_per_gpu=1,min_free_memory_mb=dict(cv=10000,nlp=5000));gpu=dict(id=0,free_mb=12000,total_mb=12288)
    assert can_launch('cv',gpu,[],r,{})
    assert not can_launch('cv',gpu,[dict(gpu=0,task='cv',pid=123)],r,{})
    assert not can_launch('nlp',gpu,[dict(gpu=0,task='cv',pid=123)],r,{})
    active=[dict(gpu=0,task='nlp',pid=123)]
    assert not can_launch('nlp',gpu,active,r,{})
    assert not can_launch('nlp',dict(gpu,id=0,free_mb=8000),active,r,{})
    assert not can_launch('nlp',dict(gpu,id=0,free_mb=8000),active,r,{123:4000})
    assert not can_launch('nlp',gpu,active+[dict(gpu=0,task='nlp',pid=456)],r,{})

def test_t_interval_and_sample_sd():
    s=statistics(range(10))
    assert s['mean']==4.5 and s['n']==10
    assert s['sd']==pytest.approx(3.0276503540974917)
    assert s['ci95_low']==pytest.approx(2.3341494103)
    assert s['ci95_high']==pytest.approx(6.6658505897)
    assert statistics([.1]*10)['sd']<1e-16
    assert statistics([1])['sd'] is None

def test_duplicate_launch_rejected_by_file_lock(tmp_path,monkeypatch):
    import fcntl,os
    import adam_mf.runtime.launcher as launcher
    monkeypatch.setattr(launcher,'local_path',lambda _:tmp_path)
    fd=os.open(tmp_path/'pipeline.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already running'):launcher.launch(dry_run=True)
    finally:os.close(fd)
