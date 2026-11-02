import json,time,statistics,torch,copy
from adam_mf.config import load_config,plan
from adam_mf.optimizers.builder import make_optimizer,logical_batch
from torch.profiler import profile,ProfilerActivity
torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.set_device(0)
c=load_config();results={}
for method in c['methods']:
 cfg=next(j for j in plan(c) if j.method==method and j.task=='cv')
 # Synthetic tensors, no formal model/data, same logical accumulation as CV.
 model=torch.nn.Sequential(*[torch.nn.Linear(64,64) for _ in range(16)]).cuda()
 opt=make_optimizer(model,cfg);batch=logical_batch(model,opt,cfg,None)
 gradients=[torch.randn_like(p)*.01 for p in model.parameters()]
 def step():
  for _ in range(batch.accumulation):
   batch.begin_microbatch()
   for p,g in zip(model.parameters(),gradients):p.grad=g.clone()
   batch.finish_microbatch()
 for _ in range(5):step()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[]
 for _ in range(30):
  torch.cuda.synchronize();start=time.perf_counter();step();torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
 with profile(activities=[ProfilerActivity.CPU]) as prof:step();torch.cuda.synchronize()
 items=sum(e.count for e in prof.key_averages() if e.key=='aten::item')
 results[method]=dict(median_ms=statistics.median(elapsed),item_calls=items,peak_bytes=torch.cuda.max_memory_allocated())
print(json.dumps(results,indent=2))
