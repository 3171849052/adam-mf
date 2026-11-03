"""Standard-library bootstrap. Existing Conda environments are strictly read-only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from adam_mf.paths import ROOT


def _capture(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'命令失败：{" ".join(map(str, args))}\n{result.stdout}{result.stderr}')
    return result.stdout


def validate_environment(python):
    # No packaging/imports from the target site-packages until every lock entry
    # has been checked. Even a missing packaging/PyYAML gets a useful diagnosis.
    locked = []
    for line in (ROOT / 'requirements.lock').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '--')):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if not match:
            raise RuntimeError(f'无法检查未精确锁定的依赖：{line}')
        locked.append(match.groups())
    code = '''import importlib.metadata as m, json, sys
locked=json.loads(sys.argv[1]); failures=[]
if sys.version_info[:2] != (3,11):
    failures.append('Python 版本不匹配：当前 '+sys.version.split()[0]+'，要求 3.11')
for name, expected in locked:
    try:
        actual=m.version(name)
        if actual != expected:
            failures.append('版本不匹配：'+name+' 当前 '+actual+'，要求 '+expected)
    except m.PackageNotFoundError:
        failures.append('缺少依赖：'+name+'=='+expected)
print(json.dumps(failures, ensure_ascii=False))
'''
    failures = json.loads(_capture([str(python), '-B', '-c', code, json.dumps(locked)]))
    check = subprocess.run([str(python), '-B', '-m', 'pip', 'check'], cwd=ROOT,
                           text=True, capture_output=True)
    if check.returncode:
        failures.append('依赖冲突（pip check 失败）：\n' + check.stdout + check.stderr)
    if failures:
        raise RuntimeError('环境检查失败；已有环境未作修改：\n' + '\n'.join(failures))


def ensure_environment():
    if sys.platform != 'linux':
        raise RuntimeError('本项目需要 Linux。')
    conda = shutil.which('conda')
    if not conda:
        candidate = os.environ.get('CONDA_EXE')
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            conda = candidate
    if not conda:
        raise RuntimeError('未找到 Conda。请先安装 Conda，并使 conda 可通过 PATH 或 CONDA_EXE 访问。')
    # Serialize first-time installation and recheck existence inside the lock.
    logs = ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'environment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        def prefixes():
            data = json.loads(_capture([conda, 'env', 'list', '--json']))
            return sorted({Path(p).absolute() for p in data['envs']
                           if Path(p).name == 'adam-mf' and Path(p) != Path(data.get('root_prefix', '/'))})
        envs = prefixes()
        if not envs:
            info = json.loads(_capture([conda, 'info', '--json']))
            remnants = [str(Path(d) / 'adam-mf') for d in info['envs_dirs'] if (Path(d) / 'adam-mf').exists()]
            if remnants:
                raise RuntimeError(f'adam-mf 目录已存在但未注册为有效环境：{remnants}；未修改环境。')
            print('首次创建 adam-mf 环境（environment.yml → requirements.lock）…', flush=True)
            # This is the sole installation command in the application. On any
            # failure leave even a partially created environment intact.
            result = subprocess.run([conda, 'env', 'create', '-f', str(ROOT / 'environment.yml'), '--yes'], cwd=ROOT)
            if result.returncode:
                raise RuntimeError('Conda 环境创建失败；未删除或修复任何环境。请检查安装输出。')
            envs = prefixes()
        if len(envs) != 1:
            raise RuntimeError(f'无法唯一定位 adam-mf 环境：{envs}')
        python = envs[0] / 'bin' / 'python'
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RuntimeError(f'已有环境缺少可执行 Python：{python}；未修改环境。')
        validate_environment(python)
        return python


def main(argv=None):
    parser = argparse.ArgumentParser(prog='./adam-mf', description='adam-mf：自动部署、预检查与正式实验管理')
    sub = parser.add_subparsers(dest='command')
    run = sub.add_parser('run', help='自动准备后后台启动或恢复正式实验')
    run.add_argument('--new-run', action='store_true', help='保留旧实验，创建独立运行')
    sub.add_parser('check', help='完整预检查，不启动正式训练')
    sub.add_parser('status', help='查看实验状态与 GPU 分配')
    sub.add_parser('stop', help='安全停止并保存检查点')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command in ('status', 'stop'):
            from adam_mf.runtime import management
            return getattr(management, args.command)() or 0
        python = ensure_environment()
        command = [str(python), '-B', '-m', 'adam_mf.cli', args.command]
        if args.command == 'run' and args.new_run:
            command.append('--new-run')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   ARROW_DEFAULT_MEMORY_POOL='system', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        os.chdir(ROOT)
        os.execve(str(python), command, env)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'错误：{error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
