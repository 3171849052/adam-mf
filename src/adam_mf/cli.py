import argparse,json
from pathlib import Path
from adam_mf.config import load_config,plan
from adam_mf.audit import verify_assets

def main():
    parser=argparse.ArgumentParser(description='adam-mf: six frozen methods, ten formal seeds, offline only')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('start','status','stop','validate','plan','verify-assets','verify-preprocessing','download-assets','prepare-data','check','simulate'):
        p=sub.add_parser(cmd);p.add_argument('--config',type=Path)
        if cmd=='start':
            p.add_argument('--dry-run',action='store_true');p.add_argument('--wait',action='store_true');p.add_argument('--new-run',action='store_true')
    p=sub.add_parser('pipeline');p.add_argument('--run',type=Path,required=True);p.add_argument('--lock-fd',type=int,required=True);p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('report');p.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='pipeline':
        from adam_mf.runtime.launcher import pipeline
        return pipeline(args.run,args.lock_fd,args.dry_run)
    if args.command=='report':
        from adam_mf.runtime.state import check_identity
        from adam_mf.reporting.report import report
        c=load_config(args.run/'config_snapshot.yaml');print(report(args.run,c,plan(c),check_identity(args.run,c)));return 0
    c=load_config(args.config)
    if args.command=='validate':print(f'Configuration valid; {len(plan(c))} formal jobs.');return 0
    if args.command=='plan':print(json.dumps([dict(id=j.id,**j.values(),**j.parameters) for j in plan(c)],indent=2));return 0
    if args.command=='verify-assets':print(f'Verified {len(verify_assets(c))} asset/manifest hashes.');return 0
    if args.command=='download-assets':
        from adam_mf.runtime.environment import check_versions
        from adam_mf.assets import fetch_assets
        check_versions();fetch_assets(c);print('All raw data, model and matrix files verified.');return 0
    if args.command=='prepare-data':
        from adam_mf.data.preprocessing import prepare_data
        print(f'Prepared and verified {len(prepare_data(c))} hashes.');return 0
    if args.command=='simulate':
        from adam_mf.runtime.scheduler import simulation
        from adam_mf.audit import atomic_json
        from adam_mf.paths import local_path
        result=simulation(c,plan(c));atomic_json(local_path(c['paths']['logs'])/'scheduler_simulation.json',result)
        for e in result['events'][:4]:print(f'GPU {e["gpu"]}: {e["job"]}')
        print(f'Simulated {result["planned"]} tasks; zero training tasks started.');return 0
    if args.command=='check':
        from adam_mf.runtime.environment import environment_report
        from adam_mf.audit import atomic_json
        from adam_mf.paths import local_path
        r=environment_report(c);atomic_json(local_path(c['paths']['logs'])/'environment_report.json',r)
        print(f'Python: {r["python_executable"]}; CUDA: {r["cuda_runtime"]}; physical GPUs: {[g["id"] for g in r["gpus"]]}')
        from adam_mf.assets import asset_path,catalog,fetch_assets
        from adam_mf.data.preprocessing import prepare_data
        if not all(asset_path(row,c).exists() for row in catalog()['files']):
            print('Missing assets: downloading and preparing pinned inputs...',flush=True)
            fetch_assets(c);prepare_data(c)
        print(f'Assets: {len(verify_assets(c))} verified hashes; formal tasks: {len(plan(c))}')
        if r['pip_check']['exit_code']:print('Existing environment pip check:',r['pip_check']['output'])
        return 0
    if args.command=='verify-preprocessing':
        from adam_mf.data.preprocessing import verify_preprocessing
        print(verify_preprocessing(c));return 0
    from adam_mf.runtime.launcher import launch,status,stop
    if args.command=='start':return launch(args.config,args.dry_run,args.wait,args.new_run)
    if args.command=='status':status(args.config)
    if args.command=='stop':stop(args.config)
    return 0
if __name__=='__main__':raise SystemExit(main())
