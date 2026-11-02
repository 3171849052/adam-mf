#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adam_mf.runtime.environment import check_versions,environment_report
from adam_mf.config import load_config
from adam_mf.audit import atomic_json
p=argparse.ArgumentParser();p.add_argument('--versions-only',action='store_true');p.add_argument('--output',type=Path,default=Path('logs/environment_report.json'));p.add_argument('--config',type=Path)
a=p.parse_args()
if a.versions_only:print(json.dumps(check_versions(),indent=2))
else:
 r=environment_report(load_config(a.config));atomic_json(a.output,r);print(json.dumps(r,indent=2))
