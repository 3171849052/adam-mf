import sys,importlib.util,time,json
from pathlib import Path
from adam_mf.config import load_config,plan
from adam_mf.audit import atomic_json,json_hash
from adam_mf.runtime.state import identity,worker_identity
import tempfile,subprocess,statistics

def old(name):
 spec=importlib.util.spec_from_file_location('before_'+name,Path('/tmp/adam-mf-before/src/adam_mf')/(name.replace('.','/')+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if len(sys.argv)>1:
 mode,run=sys.argv[1:];c=load_config();job=plan(c)[0]
 if mode=='before':
  audit=old('audit');assets=old('assets');state=old('runtime.state')
  assets.file_hash=audit.file_hash;state.verify_assets=assets.verify_assets;state.code_hashes=audit.code_hashes
  state.identity(c)
 else:
  from adam_mf.audit import read_json
  manifest=read_json(Path(run)/'run_manifest.json');worker_identity(Path(run),c,job,0,json_hash(manifest['identity']))
 from adam_mf.training.engine import run_training
else:
 c=load_config();ident=identity(c);result={}
 with tempfile.TemporaryDirectory() as d:
  atomic_json(Path(d)/'run_manifest.json',dict(identity=ident,planned_tasks=[j.values() for j in plan(c)]))
  for mode in ('before','after'):
   times=[]
   for _ in range(3):
    start=time.perf_counter();subprocess.run([sys.executable,__file__,mode,d],check=True,capture_output=True);times.append(time.perf_counter()-start)
   result[mode]=dict(seconds=times,median_seconds=statistics.median(times))
 print(json.dumps(result,indent=2))
