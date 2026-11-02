import argparse,json
from pathlib import Path
from adam_mf.config import load_config,plan
from adam_mf.audit import verify_assets

def main():
    parser=argparse.ArgumentParser(description='adam-mf: eight frozen methods, ten formal seeds, offline only')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('start','status','stop','validate','plan','verify-assets','verify-preprocessing'):
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
    if args.command=='verify-assets':print(f'Verified {len(verify_assets())} asset/manifest hashes.');return 0
    if args.command=='verify-preprocessing':
        from adam_mf.data.preprocessing import verify_preprocessing
        print(verify_preprocessing(c));return 0
    from adam_mf.runtime.launcher import launch,status,stop
    if args.command=='start':return launch(args.config,args.dry_run,args.wait,args.new_run)
    if args.command=='status':status(args.config)
    if args.command=='stop':stop(args.config)
    return 0
if __name__=='__main__':raise SystemExit(main())
