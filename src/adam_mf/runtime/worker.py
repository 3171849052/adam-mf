import argparse,os
from pathlib import Path
from adam_mf.audit import read_json
from adam_mf.config import load_config,plan
from adam_mf.runtime.state import check_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--job',required=True);p.add_argument('--gpu',required=True,type=int);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Worker GPU isolation mismatch')
    config=load_config(args.run/'config_snapshot.yaml')
    if args.gpu not in config['runtime']['gpu_ids']:raise RuntimeError(f'Physical GPU {args.gpu} is not configured for this run')
    job=next(c for c in plan(config) if c.id==args.job)
    ident=check_identity(args.run,config)
    from adam_mf.training.engine import run_training,StopRequested
    try:run_training(job,args.run,ident,args.gpu)
    except StopRequested:return 75
    return 0
if __name__=='__main__':raise SystemExit(main())
