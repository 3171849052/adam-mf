"""Process management using only Python's standard library, even with broken deps."""
import json
import os
from pathlib import Path
import signal

from adam_mf.paths import ROOT, local_path


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def alive(record):
    return record.get('process_token') is not None and process_token(record['pid']) == record['process_token']


def current_path():
    # New launches record a pointer for custom log directories; the fallback
    # keeps status/stop available for experiments created by previous versions.
    pointer = ROOT / '.adam-mf-current.json'
    if pointer.exists():
        return local_path(json.loads(pointer.read_text())['current'])
    return ROOT / 'logs' / 'current.json'


def status(path=None):
    path = current_path() if path is None else Path(path)
    if not path.exists():
        print('No pipeline has been started.')
        return
    rec = json.loads(path.read_text())
    run = local_path(rec['run_dir'])
    state = json.loads((run / 'status.json').read_text())
    print(f'Run: {run.name}\nPID: {rec["pid"]} ({"alive" if alive(rec) else "exited"})\n'
          f'State: {state["status"]}\nCompleted: {state.get("completed", 0)}/{state.get("planned", 0)}')
    for job, task in state.get('tasks', {}).items():
        if task['status'] in ('running', 'failed', 'paused'):
            print(f'  {job}: {task["status"]} GPU={task.get("gpu_id")} '
                  f'step={task.get("progress", {}).get("logical_step")} log={task.get("log")}')
    if state.get('error'):
        print('Error:', state['error'])


def _signal(record):
    if not alive(record):
        return False
    try:
        os.kill(record['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def stop(path=None):
    path = current_path() if path is None else Path(path)
    if not path.exists():
        print('No pipeline to stop.')
        return
    rec = json.loads(path.read_text())
    if _signal(rec):
        print('Graceful stop requested; workers checkpoint at the next logical step. Use ./adam-mf status to check exit.')
        return
    run = local_path(rec['run_dir'])
    state = json.loads((run / 'status.json').read_text())
    sent = sum(_signal(task) for task in state.get('tasks', {}).values() if task.get('status') == 'running')
    print(f'Coordinator exited; graceful stop sent to {sent} surviving workers. Checkpoints are preserved.')
