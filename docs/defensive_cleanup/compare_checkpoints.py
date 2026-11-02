from pathlib import Path
import torch,numpy as np,json,yaml
before=Path('/tmp/adam-mf-numeric-before');after=Path('/tmp/adam-mf-numeric-after')
count=[0]
def equal(a,b,path=''):
 if any(part in path.split('/') for part in ('wall_time','training_time','step_seconds')):return
 if torch.is_tensor(a):assert torch.equal(a,b),path;count[0]+=1
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path;count[0]+=1
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 else:assert a==b,(path,a,b)
methods=[]
for p in sorted(before.glob('test_engine_exact_mid_epoch_re[0-7]/full/checkpoints/*/latest.pt')):
 q=after/p.relative_to(before)
 a=torch.load(p,weights_only=False);b=torch.load(q,weights_only=False)
 equal(a,b);methods.append(p.parent.name)
a=yaml.safe_load(Path('/tmp/adam-mf-before/config/experiment.yaml').read_text());b=yaml.safe_load(Path('config/experiment.yaml').read_text())
changes={k:[a['runtime'].get(k),b['runtime'].get(k)] for k in a['runtime'].keys()|b['runtime'].keys() if a['runtime'].get(k)!=b['runtime'].get(k)}
a.pop('runtime');b.pop('runtime');assert a==b
print(json.dumps(dict(methods=methods,exact_tensor_arrays=count[0],config_runtime_changes=changes),indent=2))
assert len(methods)==8
