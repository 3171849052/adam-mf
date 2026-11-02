#!/usr/bin/env python3
"""Compare the current experiment with the pre-change snapshot, field for field."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import yaml
from adam_mf.paths import ROOT
from adam_mf.config import load_config
from adam_mf.audit import file_hash,json_hash,atomic_json
baseline=ROOT/'docs/baseline/experiment.before.yaml';before=yaml.safe_load(baseline.read_text());after=load_config()
a={k:v for k,v in before.items() if k not in ('runtime','paths')};b={k:v for k,v in after.items() if k not in ('runtime','paths')}
if a!=b:raise SystemExit('FAILED: original experimental parameters changed')
r=dict(status='passed',baseline_sha256=file_hash(baseline),current_sha256=file_hash(ROOT/'config/experiment.yaml'),experiment_parameters_sha256=json_hash(a),all_original_experiment_parameters_unchanged=True,allowed_sections=['runtime','paths'])
atomic_json(ROOT/'docs/config_preservation.json',r);print(r)
