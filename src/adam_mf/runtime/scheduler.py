"""Independent memory-aware GPU queue, one task family per GPU."""
import os,signal,subprocess,sys,time
from adam_mf.audit import atomic_json,read_json
from adam_mf.paths import ROOT

def gpu_inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in out.strip().splitlines():
        i,name,total,free=[v.strip() for v in line.split(',')]
        rows.append(dict(id=int(i),name=name,total_mb=int(total),free_mb=int(free)))
    return rows

def process_memory():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in out.strip().splitlines():
        try:
            pid,mb=line.split(',');rows[int(pid)]=int(mb.strip())
        except ValueError:continue
    return rows

def can_launch(task,gpu,active,runtime,usage):
    same=[a for a in active if a['gpu']==gpu['id']]
    if same and any(a['task']!=task for a in same):return False
    if len(same)>=runtime[task+'_workers_per_gpu']:return False
    # Reserve startup memory not yet reflected in nvidia-smi; never shrink batches.
    pending=sum(max(0,runtime['min_free_memory_mb'][a['task']]-usage.get(a['pid'],0)) for a in same)
    return gpu['free_mb']-pending>=runtime['min_free_memory_mb'][task]

def schedule(run,config,jobs,identity,lock_fd=None):
    from adam_mf.training.engine import verify_result,result_path
    r=config['runtime'];inventory=gpu_inventory();ids=r['gpu_ids']
    allowed=[g['id'] for g in inventory] if ids=='auto' else ids
    if not allowed or not set(allowed)<=set(g['id'] for g in inventory):raise ValueError('Configured GPU does not exist')
    for task in {j.task for j in jobs}:
        if not any(g['id'] in allowed and g['total_mb']>=r['min_free_memory_mb'][task] for g in inventory):raise RuntimeError(f'No GPU has configured memory for {task}')
    state={j.id:dict(**j.values(),status='pending') for j in jobs};queue=[]
    for j in jobs:
        if verify_result(run,j,identity):state[j.id]['status']='completed'
        else:queue.append(j)
    active={};stopping=[False];failed=False;last_wait=0
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
                inventory=gpu_inventory();usage=process_memory()
                with (run/'logs/gpu_telemetry.jsonl').open('a') as f:
                    import json
                    f.write(json.dumps(dict(time=time.time(),gpus=inventory,worker_memory_mb=usage))+'\n')
                for job in list(queue):
                    if len(active)>=r['max_parallel_tasks']:break
                    gpu=next((g for g in inventory if g['id'] in allowed and can_launch(job.task,g,[dict(gpu=a['gpu'],task=a['job'].task,pid=a['process'].pid) for a in active.values()],r,usage)),None)
                    if gpu is None:continue
                    logpath=run/'logs'/(job.id+'.log');log=logpath.open('a')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['id']),PYTHONUNBUFFERED='1')
                    proc=subprocess.Popen([sys.executable,'-B','-m','adam_mf.runtime.worker','--run',str(run),'--job',job.id,'--gpu',str(gpu['id'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock_fd,) if lock_fd is not None else ())
                    active[job.id]=dict(process=proc,job=job,gpu=gpu['id'],log=log);queue.remove(job)
                    state[job.id].update(status='running',pid=proc.pid,gpu_id=gpu['id'],log=str(logpath.relative_to(run)))
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
    complete=all(v['status']=='completed' for v in state.values())
    publish('trained' if complete else 'failed' if failed else 'paused')
    return complete
