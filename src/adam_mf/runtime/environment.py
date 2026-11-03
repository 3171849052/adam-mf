"""Read-only dependency and physical-device validation."""
import json
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from adam_mf.paths import ROOT
from adam_mf.audit import atomic_json


def dependency_closure(requirements):
    """Resolve installed metadata including extras, without installing or selecting versions."""
    queue=[Requirement(s) for s in requirements];processed={};versions={}
    while queue:
        req=queue.pop();key=canonicalize_name(req.name);requested=set(req.extras)
        version=metadata.version(req.name)
        if not req.specifier.contains(version):raise RuntimeError(f'{req.name}: installed {version}, dependency requires {req.specifier}')
        if key in processed and requested<=processed[key]:continue
        processed.setdefault(key,set()).update(requested);versions[key]=version
        for spec in metadata.requires(req.name) or []:
            dep=Requirement(spec)
            if dep.marker is None or any(dep.marker.evaluate({'extra':extra}) for extra in {''}|processed[key]):queue.append(dep)
    return versions


def check_versions():
    failures=[];versions={}
    project=tomllib.loads((ROOT/'pyproject.toml').read_text())
    if sys.version_info[:2]!=(3,11):failures.append(f'Python {platform.python_version()} installed, Python 3.11 required')
    locked=[line.strip() for line in (ROOT/'requirements.lock').read_text().splitlines() if line.strip() and not line.startswith(('#','--'))]
    for spec in list(project['project']['dependencies'])+locked:
        r=Requirement(spec)
        try:
            versions[r.name]=metadata.version(r.name)
            if not r.specifier.contains(versions[r.name]):failures.append(f'{r.name}: installed {versions[r.name]}, required {r.specifier}')
        except metadata.PackageNotFoundError:failures.append(f'Missing {r.name}')
    if failures:raise RuntimeError('\n'.join(failures))
    required=dependency_closure(project['project']['dependencies'])
    missing=set(required)-{canonicalize_name(Requirement(s).name) for s in locked}
    if missing:raise RuntimeError(f'Dependency lock misses required packages/extras: {sorted(missing)}')
    return versions


def environment_report(config, cuda_tests=True):
    from adam_mf.runtime.scheduler import selected_gpus
    versions=check_versions()
    check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
    if check.returncode:raise RuntimeError('pip check failed: '+check.stdout+check.stderr)
    import torch
    gpus=selected_gpus(config['runtime'])
    result=dict(python=platform.python_version(),python_executable=sys.executable,
                conda_environment=Path(sys.prefix).name if (Path(sys.prefix)/'conda-meta').is_dir() else None,conda_prefix=sys.prefix,
                shell_conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),shell_conda_prefix=os.environ.get('CONDA_PREFIX'),
                interpreter_environment=Path(sys.prefix).name,packages=versions,cuda_runtime=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),gpus=gpus)
    result['pip_check']=dict(exit_code=check.returncode,output=(check.stdout+check.stderr).strip())
    if not torch.cuda.is_available():raise RuntimeError('PyTorch CUDA is unavailable')
    if cuda_tests:
        for gpu in gpus:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['id']),CUDA_DEVICE_ORDER='PCI_BUS_ID')
            test=subprocess.run([sys.executable,'-c',
                'import json,torch; assert torch.cuda.device_count()==1; '
                'x=torch.arange(16.,device="cuda:0"); '
                'torch.testing.assert_close((x*x).sum().cpu(),torch.tensor(1240.)); '
                'print(json.dumps(dict(name=torch.cuda.get_device_name(0),uuid=str(torch.cuda.get_device_properties(0).uuid))))'],env=env,capture_output=True,text=True)
            gpu['cuda_test']=dict(exit_code=test.returncode,output=(test.stdout+test.stderr).strip())
            if not test.returncode and json.loads(test.stdout)['uuid'].removeprefix('GPU-')!=gpu['uuid'].removeprefix('GPU-'):raise RuntimeError(f'Physical GPU {gpu["id"]} UUID mapping mismatch')
            if test.returncode:raise RuntimeError(f'Physical GPU {gpu["id"]} CUDA test failed: {test.stderr}')
    return result
