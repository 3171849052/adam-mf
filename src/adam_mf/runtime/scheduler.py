"""Unified physical-GPU queue: configurable per-family capacity, exclusive ownership per run."""
import os,signal,subprocess,sys,time
from adam_mf.audit import atomic_json,read_json,json_hash
from adam_mf.paths import ROOT
from adam_mf.runtime.state import process_token

def gpu_inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in out.strip().splitlines():
        i,uuid,name,total,free,driver=[v.strip() for v in line.split(',')]
        rows.append(dict(id=int(i),uuid=uuid,name=name,total_mb=int(total),free_mb=int(free),driver=driver))
    return rows

def process_memory():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in out.strip().splitlines():
        try:
            pid,mb=line.split(',');rows[int(pid)]=int(mb.strip())
        except ValueError:continue
    return rows

def selected_gpus(runtime, inventory=None):
    inventory=gpu_inventory() if inventory is None else inventory
    ids=runtime['gpu_ids'];by_id={g['id']:g for g in inventory}
    missing=set(ids)-set(by_id)
    if missing:raise RuntimeError(f'Configured physical GPUs missing: {sorted(missing)}; detected {sorted(by_id)}. No automatic fallback.')
    return [by_id[i] for i in ids]


def can_launch(task,gpu,active,runtime):
    occupants=[a for a in active if a['gpu']==gpu['id']]
    if any(a['task']!=task for a in occupants):return False
    if len(occupants)>=runtime[task+'_workers_per_gpu']:return False
    return gpu['free_mb']>=runtime['min_free_memory_mb'][task]


def simulation(config,jobs):
    """Exercise the same resource predicate with mixed tasks, no CUDA or training."""
    r=config['runtime'];gpus=[dict(id=i,free_mb=999999) for i in r['gpu_ids']]
    queue=list(jobs);active=[];events=[];tick=0
    while queue or active:
        for a in list(active):
            if a['until']<=tick:events.append(dict(tick=tick,event='finish',gpu=a['gpu'],job=a['job']));active.remove(a)
        for job in list(queue):
            if len(active)>=r['max_parallel_tasks']:break
            gpu=next((g for g in gpus if can_launch(job.task,g,active,r)),None)
            if gpu is None:continue
            active.append(dict(gpu=gpu['id'],task=job.task,job=job.id,until=tick+1+job.seed%3))
            events.append(dict(tick=tick,event='start',gpu=gpu['id'],job=job.id));queue.remove(job)
        assert len(active)<=r['max_parallel_tasks']
        tick+=1
    return dict(simulated=True,training_started=False,planned=len(jobs),events=events)


def reserve_gpu(gpu,lock_dir):
    import fcntl
    lock_dir.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock_dir/(gpu['uuid']+'.lock'),os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:os.close(fd);return None
    return fd


def resume_gpu(run,job,inventory):
    folder=run/'checkpoints'/job.id
    if not (folder/'latest.pt').exists():return None
    mapping=read_json(folder/'gpu_mapping.json');uuid=mapping['gpu_uuid']
    if uuid not in {g['uuid'] for g in inventory}:raise ValueError(f'Checkpoint physical GPU UUID unavailable: {uuid}')
    return uuid


def schedule(run,config,jobs,identity,lock_fd=None):
    from adam_mf.training.engine import verify_result,result_path
    r=config['runtime'];inventory=selected_gpus(r);allowed=r['gpu_ids']
    for task in {j.task for j in jobs}:
        if not any(g['id'] in allowed and g['total_mb']>=r['min_free_memory_mb'][task] for g in inventory):raise RuntimeError(f'No GPU has configured memory for {task}')
    state={j.id:dict(**j.values(),status='pending') for j in jobs};queue=[]
    for j in jobs:
        if verify_result(run,j,identity):state[j.id]['status']='completed'
        else:queue.append(j)
    active={};gpu_locks={};stopping=[False];failed=False;last_wait=0
    def stop(signum,frame):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def publish(status):
        atomic_json(run/'status.json',dict(status=status,planned=len(jobs),completed=sum(v['status']=='completed' for v in state.values()),failed=sum(v['status']=='failed' for v in state.values()),tasks=state,updated=time.time()))
    try:
        while queue or active:
            for job_id,a in list(active.items()):
                ret=a['process'].poll()
                if ret is None:continue
                a['log'].close();del active[job_id]
                if not any(other['gpu']==a['gpu'] for other in active.values()):os.close(gpu_locks.pop(a['gpu']))
                row=state[job_id];row['exit_code']=ret
                if ret==0 and verify_result(run,a['job'],identity):row['status']='completed'
                elif ret==75:
                    row['status']='paused';stopping[0]=True
                else:
                    row['status']='failed';failed=True
                    if r['stop_on_error']:stopping[0]=True
            if stopping[0]:
                for a in active.values():
                    if not a.get('stop_sent'):
                        a['process'].send_signal(signal.SIGTERM);a['stop_sent']=True
                publish('stopping' if active else 'failed' if failed else 'paused')
                if not active:break
            else:
                inventory=selected_gpus(r);usage=process_memory()
                with (run/'logs/gpu_telemetry.jsonl').open('a') as f:
                    import json
                    f.write(json.dumps(dict(time=time.time(),gpus=inventory,worker_memory_mb=usage))+'\n')
                for job in list(queue):
                    if len(active)>=r['max_parallel_tasks']:break
                    binding=resume_gpu(run,job,inventory)
                    gpu=None;gpu_lock=None
                    occupied=[dict(gpu=a['gpu'],task=a['job'].task,pid=a['process'].pid) for a in active.values()]
                    for candidate in inventory:
                        if (binding and candidate['uuid']!=binding) or candidate['id'] not in allowed or not can_launch(job.task,candidate,occupied,r):continue
                        gpu_lock=gpu_locks.get(candidate['id'])
                        if gpu_lock is None:gpu_lock=reserve_gpu(candidate,ROOT/'logs/gpu_locks')
                        if gpu_lock is not None:
                            gpu_locks[candidate['id']]=gpu_lock;gpu=candidate;break
                    if gpu is None:continue
                    logpath=run/'logs'/(job.id+'.log');log=logpath.open('a')
                    env=dict(os.environ,ADAM_MF_IDENTITY_HASH=json_hash(identity),CUDA_VISIBLE_DEVICES=str(gpu['id']),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
                    proc=None
                    try:proc=subprocess.Popen([sys.executable,'-B','-m','adam_mf.runtime.worker','--run',str(run),'--job',job.id,'--gpu',str(gpu['id'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_lock,)+( (lock_fd,) if lock_fd is not None else ()))
                    finally:
                        if proc is None:log.close()
                    active[job.id]=dict(process=proc,job=job,gpu=gpu['id'],log=log);queue.remove(job)
                    manifest_path=run/'run_manifest.json'
                    if manifest_path.exists():
                        manifest=read_json(manifest_path)
                        manifest.setdefault('gpu_assignments',[]).append(dict(job=job.id,pid=proc.pid,physical_gpu_id=gpu['id'],gpu_uuid=gpu['uuid'],gpu_name=gpu['name'],cuda_visible_devices=str(gpu['id']),logical_cuda_device='cuda:0',free_memory_mb=gpu['free_mb'],started=time.time()))
                        atomic_json(manifest_path,manifest)
                    state[job.id].update(status='running',pid=proc.pid,process_token=process_token(proc.pid),gpu_id=gpu['id'],gpu_uuid=gpu['uuid'],gpu_name=gpu['name'],cuda_visible_devices=str(gpu['id']),logical_device='cuda:0',log=str(logpath.relative_to(run)))
                    print(f'Start {job.id} GPU {gpu["id"]} PID {proc.pid}',flush=True)
                for a in active.values():
                    path=run/'checkpoints'/a['job'].id/'progress.json'
                    if path.exists():state[a['job'].id]['progress']=read_json(path)
                publish('running')
                if queue and not active and time.monotonic()-last_wait>60:
                    print('Waiting for GPU memory; parameters remain frozen.',flush=True);last_wait=time.monotonic()
            if queue or active:time.sleep(r['poll_seconds'])
    finally:
        # On scheduler exceptions, preserve children and the lock until graceful checkpoint exit.
        for a in active.values():
            if a['process'].poll() is None:a['process'].send_signal(signal.SIGTERM)
        for a in active.values():a['process'].wait();a['log'].close()
        for fd in gpu_locks.values():os.close(fd)
    complete=all(v['status']=='completed' for v in state.values())
    publish('trained' if complete else 'failed' if failed else 'paused')
    return complete
