"""Deployment regressions with synthetic Conda, metadata, and coordinators only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from adam_mf import bootstrap
from adam_mf.audit import atomic_json, code_hashes
from adam_mf.config import load_config
from adam_mf.paths import ROOT


@pytest.fixture
def fake_conda(tmp_path, monkeypatch):
    root = tmp_path / 'repository with spaces'
    root.mkdir()
    (root / 'environment.yml').write_text('name: adam-mf\n')
    (root / 'requirements.lock').write_text('pytest==9.1.1\n')
    prefix = tmp_path / 'envs' / 'adam-mf'
    calls = []
    state = dict(exists=False, create_code=0, partial=False)
    monkeypatch.setattr(bootstrap, 'ROOT', root)
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda _: '/fake/conda')
    def install_prefix():
        (prefix / 'bin').mkdir(parents=True, exist_ok=True)
        (prefix / 'bin' / 'python').symlink_to(sys.executable)
        state['exists'] = True
    def run(args, **kwargs):
        calls.append(args)
        if args[1:] == ['env', 'list', '--json']:
            return SimpleNamespace(returncode=0, stdout=json.dumps({'envs': [str(prefix)] if state['exists'] else []}), stderr='')
        if args[1:] == ['info', '--json']:
            return SimpleNamespace(returncode=0, stdout=json.dumps({'envs_dirs': [str(prefix.parent)]}), stderr='')
        assert args == ['/fake/conda', 'env', 'create', '-f', str(root / 'environment.yml'), '--yes']
        if state['partial']:
            prefix.mkdir(parents=True)
            (prefix / 'partial').write_bytes(b'preserve partial installation')
        if not state['create_code']:
            install_prefix()
        return SimpleNamespace(returncode=state['create_code'])
    monkeypatch.setattr(bootstrap.subprocess, 'run', run)
    return root, prefix, calls, state, install_prefix


def test_conda_missing_is_actionable_without_install(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda _: None)
    monkeypatch.delenv('CONDA_EXE', raising=False)
    with pytest.raises(RuntimeError, match='未找到 Conda'):
        bootstrap.ensure_environment()


def test_create_only_missing_environment_and_reuse_exact_python(fake_conda, monkeypatch):
    root, prefix, calls, state, _ = fake_conda
    checks = []
    monkeypatch.setattr(bootstrap, 'validate_environment', lambda p: checks.append(p))
    assert bootstrap.ensure_environment() == prefix / 'bin' / 'python'
    assert bootstrap.ensure_environment() == prefix / 'bin' / 'python'
    assert len(checks) == 2
    assert sum('create' in call for call in calls) == 1
    assert all(call[1:3] in (['env', 'list'], ['env', 'create'], ['info', '--json']) for call in calls)


def test_existing_valid_environment_never_installs(fake_conda, monkeypatch):
    _, prefix, calls, _, install = fake_conda
    install()
    monkeypatch.setattr(bootstrap, 'validate_environment', lambda _: None)
    assert bootstrap.ensure_environment() == prefix / 'bin' / 'python'
    assert len(calls) == 1 and calls[0][1:] == ['env', 'list', '--json']


def test_invalid_existing_environment_never_repairs(fake_conda, monkeypatch):
    _, _, calls, _, install = fake_conda
    install()
    def invalid(_):
        raise RuntimeError('缺少依赖：torch')
    monkeypatch.setattr(bootstrap, 'validate_environment', invalid)
    with pytest.raises(RuntimeError, match='缺少依赖'):
        bootstrap.ensure_environment()
    assert all(call[1:] == ['env', 'list', '--json'] for call in calls)


def test_creation_failure_leaves_other_and_partial_environments(fake_conda):
    root, prefix, calls, state, _ = fake_conda
    other = root / 'other-environment'
    other.write_bytes(b'preserve')
    partial = prefix / 'partial'
    state.update(create_code=9, partial=True)
    with pytest.raises(RuntimeError, match='创建失败'):
        bootstrap.ensure_environment()
    assert other.read_bytes() == b'preserve'
    assert partial.read_bytes() == b'preserve partial installation'
    assert len(calls) == 3 and 'create' in calls[-1]


@pytest.mark.parametrize('problem,expected', [
    ('python', 'Python 版本不匹配'),
    ('missing', '缺少依赖：synthetic'),
    ('version', '版本不匹配：synthetic'),
    ('pip', 'pip check 失败'),
    ('all', 'Python 版本不匹配'),
    ('ok', None),
])
def test_strict_version_and_pip_diagnostics_execute_probe(tmp_path, monkeypatch, problem, expected):
    (tmp_path / 'requirements.lock').write_text('synthetic==1.0\n')
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    original_run = subprocess.run
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        if '-c' in args:
            # Execute the real stdlib probe, replacing only installed metadata.
            prefix = 'import importlib.metadata as m, sys\n'
            if problem in ('python', 'all'):
                prefix += "sys.version_info=(3,10,0); sys.version='3.10.0 synthetic'\n"
            if problem in ('missing', 'all'):
                prefix += 'def version(name): raise m.PackageNotFoundError(name)\nm.version=version\n'
            else:
                prefix += f'm.version=lambda name: {"2.0" if problem == "version" else "1.0"!r}\n'
            return original_run([sys.executable, '-B', '-c', prefix + args[args.index('-c') + 1], args[-1]], **kwargs)
        assert args[1:] == ['-B', '-m', 'pip', 'check']
        return SimpleNamespace(returncode=int(problem in ('pip', 'all')), stdout='synthetic conflict', stderr='')
    monkeypatch.setattr(bootstrap.subprocess, 'run', run)
    if expected:
        with pytest.raises(RuntimeError, match=expected) as exc:
            bootstrap.validate_environment(Path(sys.executable))
        if problem == 'all':
            assert '缺少依赖' in str(exc.value) and 'pip check 失败' in str(exc.value)
    else:
        bootstrap.validate_environment(Path(sys.executable))
    assert len(commands) == 2


def test_pip_conflict_stops_hardware_preflight(monkeypatch):
    from adam_mf.runtime import environment, scheduler
    monkeypatch.setattr(environment, 'check_versions', lambda: {})
    monkeypatch.setattr(environment.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout='conflict', stderr=''))
    monkeypatch.setattr(scheduler, 'selected_gpus', lambda _: pytest.fail('GPU checks reached after pip failure'))
    with pytest.raises(RuntimeError, match='pip check failed'):
        environment.environment_report(load_config())


def test_management_works_without_site_packages_conda_or_config_dependencies(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PATH=str(tmp_path))
    env.pop('CONDA_EXE', None)
    for command in ('status', 'stop'):
        code = ('from pathlib import Path; from adam_mf.runtime import management; '
                f'management.ROOT=Path({str(tmp_path)!r}); '
                f'from adam_mf.bootstrap import main; raise SystemExit(main([{command!r}]))')
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_entrypoint_help_from_arbitrary_directory_never_checks_environment(tmp_path):
    result = subprocess.run([str(ROOT / 'adam-mf')], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and '{run,check,status,stop}' in result.stdout
    assert os.access(ROOT / 'adam-mf', os.X_OK)


def test_standard_library_stop_validates_pid_and_preserves_state(tmp_path, monkeypatch):
    from adam_mf.runtime import management
    monkeypatch.setattr(management, 'local_path', lambda _: tmp_path)
    rec = dict(pid=10, process_token='old', run_dir='synthetic')
    state = dict(status='running', tasks={
        'valid': dict(status='running', pid=20, process_token='valid'),
        'stale': dict(status='running', pid=30, process_token='stale')})
    atomic_json(tmp_path / 'current.json', rec)
    atomic_json(tmp_path / 'status.json', state)
    before = (tmp_path / 'status.json').read_bytes()
    monkeypatch.setattr(management, 'process_token', lambda pid: 'valid' if pid == 20 else 'reused')
    sent = []
    monkeypatch.setattr(management.os, 'kill', lambda pid, sig: sent.append((pid, sig)))
    management.stop(tmp_path / 'current.json')
    assert sent == [(20, signal.SIGTERM)]
    assert (tmp_path / 'status.json').read_bytes() == before


@pytest.mark.parametrize('passes', [False, True])
def test_check_isolated_from_formal_runs_and_workers(tmp_path, monkeypatch, passes):
    from adam_mf.runtime import launcher, scheduler
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'local_path', lambda p: tmp_path / p)
    monkeypatch.setattr(launcher, 'code_hashes', lambda: {})
    old = tmp_path / 'outputs' / 'old'
    old.mkdir(parents=True)
    (old / 'status.json').write_bytes(b'formal state')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'last_formal.json').write_bytes(b'formal pointer')
    (logs / 'current.json').write_bytes(b'current pointer')
    def preflight(run, config):
        if not passes:
            raise RuntimeError('synthetic preflight failure')
        assert len(launcher.plan(config)) == 120
    monkeypatch.setattr(launcher, 'preflight', preflight)
    monkeypatch.setattr(scheduler, 'schedule', lambda *a: pytest.fail('check scheduled training'))
    assert launcher.check() == (0 if passes else 1)
    assert (old / 'status.json').read_bytes() == b'formal state'
    assert (logs / 'last_formal.json').read_bytes() == b'formal pointer'
    assert (logs / 'current.json').read_bytes() == b'current pointer'
    assert list((tmp_path / 'outputs').iterdir()) == [old]


@pytest.mark.parametrize('failure', ['environment', 'assets', 'tests', 'test_exit', 'simulation', None])
def test_pipeline_only_schedules_after_full_preflight(tmp_path, monkeypatch, failure):
    from adam_mf.runtime import launcher, scheduler, environment, state
    from adam_mf import assets
    from adam_mf.data import preprocessing
    from adam_mf.reporting import report
    config = load_config()
    monkeypatch.setattr(launcher, 'load_config', lambda _: config)
    monkeypatch.setattr(launcher, 'code_hashes', lambda: {})
    calls = []
    def stage(name, value):
        def perform(*a, **kw):
            calls.append(name)
            if name == failure:
                raise RuntimeError(name + ' failed')
            return value
        return perform
    monkeypatch.setattr(environment, 'environment_report', stage('environment', {'gpus': []}))
    monkeypatch.setattr(assets, 'fetch_assets', stage('assets', {}))
    monkeypatch.setattr(preprocessing, 'prepare_data', stage('preprocessing', {}))
    monkeypatch.setattr(launcher, 'identity', stage('identity', {}))
    monkeypatch.setattr(launcher.subprocess, 'run', stage('tests', SimpleNamespace(returncode=int(failure == 'test_exit'))))
    monkeypatch.setattr(scheduler, 'simulation', stage('simulation', {}))
    monkeypatch.setattr(scheduler, 'schedule', stage('schedule', True))
    monkeypatch.setattr(report, 'report', stage('report', {'status': 'passed'}))
    (tmp_path / 'logs').mkdir()
    atomic_json(tmp_path / 'status.json', dict(status='created', completed=0))
    atomic_json(tmp_path / 'run_manifest.json', dict(identity=None, code_sha256={}))
    fd = os.open(tmp_path / 'pipeline.lock', os.O_CREAT | os.O_RDWR, 0o600)
    assert launcher.pipeline(tmp_path, fd) == (1 if failure else 0)
    assert ('schedule' in calls) == (failure is None)
    if failure is None:
        assert calls.index('schedule') > calls.index('simulation') > calls.index('tests') > calls.index('assets')
    else:
        assert json.loads((tmp_path / 'status.json').read_text())['completed'] == 0


def test_completed_run_never_restarts_or_overwrites(tmp_path, monkeypatch):
    from adam_mf.runtime import launcher
    config = load_config()
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'local_path', lambda p: tmp_path / p)
    monkeypatch.setattr(launcher, 'load_config', lambda _: config)
    logs = tmp_path / 'logs'
    old = tmp_path / 'outputs/runs/old'
    old.mkdir(parents=True)
    atomic_json(old / 'status.json', dict(status='completed', completed=120))
    atomic_json(logs / 'last_formal.json', dict(run_dir='outputs/runs/old'))
    before = {p: p.read_bytes() for p in (old / 'status.json', logs / 'last_formal.json')}
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('completed run restarted'))
    assert launcher.launch() == 0
    assert all(p.read_bytes() == body for p, body in before.items())


def test_fixture_preserves_original_sha256():
    import hashlib
    assert hashlib.sha256((ROOT / 'tests/fixtures/experiment_baseline.yaml').read_bytes()).hexdigest() == 'd25df31beebcb022b7c74ae88c2e971dbacd828be6433abac7028a61c62dc9f1'


def test_unregistered_environment_directory_is_never_modified(fake_conda):
    _, prefix, calls, _, _ = fake_conda
    prefix.mkdir(parents=True)
    marker = prefix / 'keep'
    marker.write_bytes(b'existing directory')
    with pytest.raises(RuntimeError, match='目录已存在'):
        bootstrap.ensure_environment()
    assert marker.read_bytes() == b'existing directory'
    assert not any('create' in call for call in calls)


@pytest.mark.parametrize('args', [['run'], ['run', '--new-run'], ['check']])
def test_bootstrap_executes_absolute_python_with_source_path(tmp_path,monkeypatch,args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bootstrap,'ensure_environment',lambda:Path(sys.executable))
    class Executed(Exception):pass
    def execute(path,command,env):
        assert path==sys.executable and Path(path).is_absolute()
        assert command==[sys.executable,'-B','-m','adam_mf.cli',*args]
        assert Path.cwd()==ROOT
        assert env['PYTHONPATH']==str(ROOT/'src')
        assert env['PYTHONDONTWRITEBYTECODE']=='1'
        raise Executed()
    monkeypatch.setattr(bootstrap.os,'execve',execute)
    with pytest.raises(Executed):bootstrap.main(args)


def test_environment_failure_never_executes_application(monkeypatch,capsys):
    def invalid():raise RuntimeError('strict dependency failure')
    monkeypatch.setattr(bootstrap,'ensure_environment',invalid)
    monkeypatch.setattr(bootstrap.os,'execve',lambda *a:pytest.fail('application started after environment failure'))
    assert bootstrap.main(['run'])==1
    assert 'strict dependency failure' in capsys.readouterr().err
