"""Detached coordinator with an inherited flock, PID identity and resumable runs."""
import fcntl,os,signal,subprocess,sys,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import yaml
from adam_mf.paths import ROOT,local_path
from adam_mf.config import load_config,plan
from adam_mf.audit import atomic_json,atomic_text,read_json,code_hashes,verify_assets,json_hash
from adam_mf.runtime.state import identity,check_identity,process_token,alive

class PipelineStopped(Exception):pass

def launch(config_path=None,dry_run=False,wait=False,new_run=False):
    config=load_config(config_path);logs=local_path(config['paths']['logs']);logs.mkdir(parents=True,exist_ok=True)
    lock=os.open(logs/'pipeline.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A pipeline or its worker is already running; use ./status.sh')
        last=logs/'last_formal.json';run=None
        if not dry_run and not new_run and config['runtime']['resume'] and last.exists():
            candidate=local_path(read_json(last)['run_dir'])
            status=read_json(candidate/'status.json')['status'] if (candidate/'status.json').exists() else 'created'
            if status!='completed':
                if read_json(candidate/'run_manifest.json').get('identity'):check_identity(candidate,config)
                if json_hash(load_config(candidate/'config_snapshot.yaml'))!=json_hash(config) or read_json(candidate/'run_manifest.json')['code_sha256']!=code_hashes():raise ValueError('Resume refused: configuration/code changed')
                run=candidate
        if run is None:
            run_id=('dry-' if dry_run else '')+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')+f'-{os.getpid()}'
            run=local_path(config['paths']['outputs'])/run_id;run.mkdir(parents=True)
            for name in ('checkpoints','per_seed/cv','per_seed/nlp','figures','audit','logs'):(run/name).mkdir(parents=True,exist_ok=True)
            atomic_text(run/'config_snapshot.yaml',yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
            ident=None
            atomic_json(run/'run_manifest.json',dict(run_id=run_id,dry_run=dry_run,identity=ident,code_sha256=code_hashes(),planned_tasks=[j.values() for j in plan(config)]))
            atomic_json(run/'audit/config_hash.json',dict(sha256=json_hash(config)))
            atomic_json(run/'audit/code_hash.json',code_hashes())
            atomic_json(run/'status.json',dict(status='created',planned=len(plan(config)),completed=0))
        args=[sys.executable,'-B','-m','adam_mf.cli','pipeline','--run',str(run),'--lock-fd',str(lock)]
        if dry_run:args.append('--dry-run')
        with (logs/'pipeline.log').open('a') as log:
            proc=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock,),env=dict(os.environ,PYTHONUNBUFFERED='1'))
        record=dict(pid=proc.pid,process_token=process_token(proc.pid),run_dir=str(run.relative_to(ROOT)),dry_run=dry_run)
        atomic_json(logs/'current.json',record)
        if not dry_run:atomic_json(last,record)
        print(f'PID {proc.pid} | {run.relative_to(ROOT)} | '+('dry-run (no training)' if dry_run else 'formal pipeline'),flush=True)
    finally:os.close(lock)
    if wait:return proc.wait()
    return 0

def preflight(run,config):
    from adam_mf.runtime.environment import environment_report
    from adam_mf.runtime.scheduler import simulation
    from adam_mf.assets import fetch_assets
    from adam_mf.data.preprocessing import prepare_data
    print('Environment and four-GPU CUDA checks...',flush=True)
    env=environment_report(config)
    atomic_json(run/'audit/environment_report.json',env)
    atomic_json(run/'audit/gpu_report.json',env['gpus'])
    manifest=read_json(run/'run_manifest.json')
    if manifest.get('identity'):
        # Never silently repair or replace inputs of an existing experiment.
        check_identity(run,config)
    else:
        if manifest['code_sha256']!=code_hashes():raise ValueError('Code changed since launch')
        print('Downloading/verifying pinned assets...',flush=True)
        fetch_assets(config);prepare_data(config)
        ident=identity(config);manifest['identity']=ident
        atomic_json(run/'run_manifest.json',manifest)
        atomic_json(run/'audit/asset_hashes.json',verify_assets(config))
    logpath=run/'logs/unit_tests.log'
    with logpath.open('w') as f:
        test=subprocess.run([sys.executable,'-B','-m','pytest',str(ROOT/'tests'),'-q'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(config['runtime']['gpu_ids'][0]),CUDA_DEVICE_ORDER='PCI_BUS_ID',ADAM_MF_TEST_CUDA='1',ADAM_MF_CONFIG=str(run/'config_snapshot.yaml'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1'))
    if test.returncode:raise RuntimeError(f'Unit tests failed: {logpath}')
    atomic_json(run/'audit/scheduler_dry_run.json',simulation(config,plan(config)))
    atomic_json(run/'audit/preflight.json',dict(status='passed',tests_exit_code=test.returncode,planned=len(plan(config)),dry_run_training_tasks=0))


def pipeline(run,lock_fd,dry_run=False):
    config=load_config(run/'config_snapshot.yaml')
    def stop_preflight(signum,frame):raise PipelineStopped('Stopped during preflight; no formal training started')
    signal.signal(signal.SIGTERM,stop_preflight);signal.signal(signal.SIGINT,stop_preflight)
    try:
        # FD is inherited from the launcher; keep it until all worker processes exit.
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read_json(run/'status.json')
        atomic_json(run/'status.json',dict(previous,status='preflight'))
        preflight(run,config);jobs=plan(config)
        atomic_json(run/'task_plan.json',[dict(id=j.id,**j.values(),**j.parameters,total_steps=j.total_steps,physical_batch_size=j.physical_batch) for j in jobs])
        if dry_run:
            check_identity(run,config)
            atomic_json(run/'status.json',dict(status='dry_run_complete',planned=len(jobs),completed=0,training_started=False))
            print(f'Dry-run passed: {len(jobs)} formal jobs generated, zero training tasks started.',flush=True);return 0
        ident=check_identity(run,config)
        from adam_mf.runtime.scheduler import schedule
        complete=schedule(run,config,jobs,ident,lock_fd)
        if complete:
            previous=read_json(run/'status.json')
            atomic_json(run/'status.json',dict(previous,status='reporting'))
        from adam_mf.reporting.report import report
        audit=report(run,config,jobs,ident)
        if complete and len(jobs)==160 and audit['status']!='passed':
            atomic_json(run/'status.json',dict(status='audit_failed',completed=audit['completed'],planned=len(jobs)))
            raise RuntimeError('Training completed but final integrity audit did not pass')
        if complete and audit['status']=='passed':
            previous=read_json(run/'status.json')
            atomic_json(run/'status.json',dict(previous,status='completed'))
        elif complete:raise RuntimeError('Partial plan cannot be marked completed')
        return 0 if complete else 2
    except PipelineStopped as error:
        previous=read_json(run/'status.json')
        atomic_json(run/'status.json',dict(previous,status='paused',message=str(error)))
        print(str(error),flush=True);return 75
    except BaseException as error:
        atomic_json(run/'audit/pipeline_error.json',dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
        previous=read_json(run/'status.json') if (run/'status.json').exists() else {}
        atomic_json(run/'status.json',dict(previous,status='failed',error=str(error)))
        traceback.print_exc();return 1
    finally:os.close(lock_fd)

def status(config_path=None):
    c=load_config(config_path);path=local_path(c['paths']['logs'])/'current.json'
    if not path.exists():print('No pipeline has been started.');return
    rec=read_json(path);run=local_path(rec['run_dir']);s=read_json(run/'status.json')
    print(f'Run: {run.name}\nPID: {rec["pid"]} ({"alive" if alive(rec) else "exited"})\nState: {s["status"]}\nCompleted: {s.get("completed",0)}/{s.get("planned",0)}')
    for job,v in s.get('tasks',{}).items():
        if v['status'] in ('running','failed','paused'):print(f'  {job}: {v["status"]} GPU={v.get("gpu_id")} step={v.get("progress",{}).get("logical_step")} log={v.get("log")}')
    if s.get('error'):print('Error:',s['error'])

def stop(config_path=None):
    c=load_config(config_path);p=local_path(c['paths']['logs'])/'current.json'
    if not p.exists():print('No pipeline to stop.');return
    rec=read_json(p)
    if not alive(rec):
        run=local_path(rec['run_dir']);state=read_json(run/'status.json');sent=0
        for task in state.get('tasks',{}).values():
            if task.get('status')=='running' and alive(task):
                os.kill(task['pid'],signal.SIGTERM);sent+=1
        print(f'Coordinator exited; graceful stop sent to {sent} surviving workers. Checkpoints are preserved.')
        return
    os.kill(rec['pid'],signal.SIGTERM);print('Graceful stop requested; workers checkpoint at the next logical step. Use ./status.sh to check exit.')
