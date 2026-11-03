"""Old snapshots must be rejected clearly, without rewriting or launching."""
import copy
import yaml
import pytest
from adam_mf.audit import atomic_json
from adam_mf.config import load_config


@pytest.mark.parametrize('change',['old_schema','config','code'])
def test_start_refuses_changed_run_without_validating_old_schema(tmp_path,monkeypatch,capsys,change):
    import adam_mf.runtime.launcher as launcher
    c=load_config();snapshot=copy.deepcopy(c)
    if change=='old_schema':snapshot['runtime']['max_tasks_per_gpu']=1
    elif change=='config':snapshot['runtime']['nlp_workers_per_gpu']+=1
    codes={'src/example.py':'current'}
    monkeypatch.setattr(launcher,'load_config',lambda _:c)
    monkeypatch.setattr(launcher,'local_path',lambda _:tmp_path)
    monkeypatch.setattr(launcher,'code_hashes',lambda:codes)
    def no_launch(*args,**kwargs):raise AssertionError('Incompatible run must not launch')
    monkeypatch.setattr(launcher.subprocess,'Popen',no_launch)
    atomic_json(tmp_path/'last_formal.json',dict(run_dir='old-run'))
    atomic_json(tmp_path/'status.json',dict(status='paused'))
    atomic_json(tmp_path/'run_manifest.json',dict(code_sha256={} if change=='code' else codes))
    (tmp_path/'config_snapshot.yaml').write_text(yaml.safe_dump(snapshot))
    checkpoint=tmp_path/'latest.pt';checkpoint.write_bytes(b'preserve checkpoint')
    paths=list(tmp_path.iterdir());before={p:p.read_bytes() for p in paths}
    assert launcher.launch()==2
    message=capsys.readouterr().err
    assert 'Resume refused' in message and './adam-mf run --new-run' in message
    assert all(p.read_bytes()==content for p,content in before.items())


def test_new_run_preserves_old_run_and_uses_current_snapshot(tmp_path,monkeypatch):
    import adam_mf.runtime.launcher as launcher
    c=load_config();logs=tmp_path/c['paths']['logs'];logs.mkdir()
    old=tmp_path/'old-run';old.mkdir();(old/'config_snapshot.yaml').write_text('obsolete: schema\n')
    atomic_json(logs/'last_formal.json',dict(run_dir='old-run'))
    monkeypatch.setattr(launcher,'load_config',lambda _:c)
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'local_path',lambda path:tmp_path/path)
    monkeypatch.setattr(launcher,'code_hashes',lambda:{'src/example.py':'current'})
    calls=[];options=[]
    checkpoint=old/'latest.pt';checkpoint.write_bytes(b'preserve checkpoint')
    class Process:
        pid=123456789
        def __init__(self,args,**kwargs):calls.append(args);options.append(kwargs)
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    assert launcher.launch(new_run=True)==0
    assert len(calls)==1
    run=__import__('pathlib').Path(calls[0][calls[0].index('--run')+1])
    assert yaml.safe_load((run/'config_snapshot.yaml').read_text())==c
    assert (old/'config_snapshot.yaml').read_text()=='obsolete: schema\n'
    assert checkpoint.read_bytes()==b'preserve checkpoint'
    assert calls[0][0]==__import__('sys').executable
    assert options[0]['start_new_session'] is True
    assert options[0]['stdin']==launcher.subprocess.DEVNULL
    assert options[0]['stdout'] is options[0]['stderr'] or options[0]['stderr']==launcher.subprocess.STDOUT
    assert len(options[0]['pass_fds'])==1


def test_compatible_run_resumes_same_directory_and_preserves_checkpoint(tmp_path,monkeypatch):
    import adam_mf.runtime.launcher as launcher
    c=load_config();logs=tmp_path/'logs';old=tmp_path/'outputs/runs/old';old.mkdir(parents=True)
    codes={'src/example.py':'unchanged'}
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'local_path',lambda p:tmp_path/p)
    monkeypatch.setattr(launcher,'load_config',lambda _:c)
    monkeypatch.setattr(launcher,'code_hashes',lambda:codes)
    atomic_json(logs/'last_formal.json',dict(run_dir='outputs/runs/old'))
    atomic_json(old/'status.json',dict(status='paused',completed=7))
    atomic_json(old/'run_manifest.json',dict(code_sha256=codes))
    (old/'config_snapshot.yaml').write_text(yaml.safe_dump(c))
    checkpoint=old/'latest.pt';checkpoint.write_bytes(b'original rng/optimizer/model checkpoint')
    before={p:p.read_bytes() for p in old.iterdir()}
    calls=[]
    class Process:
        pid=123456789
        def __init__(self,args,**kwargs):calls.append(args)
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    assert launcher.launch()==0
    assert calls[0][calls[0].index('--run')+1]==str(old)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert list(old.parent.iterdir())==[old]
