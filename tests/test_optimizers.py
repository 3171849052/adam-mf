from config_helpers import load_config
import copy
import pytest
import torch
from adam_mf.config import plan
from adam_mf.optimizers.builder import make_optimizer,logical_batch
from adam_mf.optimizers.adam import unclipped_microbatch
from adam_mf.registry import METHODS

@pytest.mark.parametrize('name',list(METHODS))
def test_literal_formulas_and_one_noise(name):
    cfg=next(j for j in plan(load_config()) if j.task=='cv' and j.method==name)
    model=torch.nn.Linear(2,1,bias=False);p=model.weight;opt=make_optimizer(model,cfg)
    class Noise:
        count=0
        def next(self):self.count+=1;return [torch.tensor([[.7,-.4]])]
    noise=Noise() if cfg.private else None
    logical=logical_batch(model,opt,cfg,noise)
    x=p.detach().clone();m=torch.zeros_like(x);v=m.clone();b1,b2=cfg.betas
    for t in range(1,7):
        d=(v/(1-b2**(t-1))).sqrt()+cfg.eps_scale if cfg.scaled and t>1 else torch.full_like(v,cfg.eps_scale or 1)
        for micro in range(4):
            logical.begin_microbatch();p.grad=torch.tensor([[1.,-2.]])*t
            if cfg.scaled:torch.testing.assert_close(p._logical_scale,d.reciprocal())
            assert logical.finish_microbatch()==(micro==3)
            if noise:assert noise.count==t-1+int(micro==3)
        total=torch.tensor([[4.,-8.]])*t
        z=torch.tensor([[.7,-.4]]) if noise else torch.zeros_like(total)
        if name.startswith('stpex'):g=(total/d+z)/1000
        elif cfg.scaled:g=(total/d+z)*d/1000
        else:g=(total+z)/1000
        m=b1*m+(1-b1)*g
        if cfg.spec.optimizer=='adam':
            v=b2*v+(1-b2)*g.square()
            x=x-cfg.lr*(m/(1-b1**t))/((v/(1-b2**t)).sqrt()+cfg.adam_eps)
        elif cfg.spec.optimizer=='stpex':
            v=b2*v+(1-b2)*(d*g).square();x=x-cfg.lr*m/(1-b1**t)
        else:x=x-cfg.lr*m
        torch.testing.assert_close(p,x,rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(opt.state[p]['exp_avg'],m)
        if cfg.spec.optimizer!='momentum':torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v,rtol=2e-5,atol=1e-10)

@pytest.mark.parametrize('name',['adam-no-noise','momentum-no-noise'])
def test_baselines_are_ordinary_unclipped_sums(name):
    cfg=next(j for j in plan(load_config()) if j.method==name)
    model=torch.nn.Linear(2,2);reference=copy.deepcopy(model)
    x=torch.tensor([[1000.,-2000.],[3000.,5000.]]);y=torch.tensor([0,1])
    expected=torch.nn.functional.cross_entropy(reference(x),y,reduction='sum');expected.backward()
    loss,clipped=unclipped_microbatch(model,x,y)
    assert loss==float(expected.detach()) and clipped==0
    for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    logical=logical_batch(model,make_optimizer(model,cfg),cfg,None)
    assert logical.noise is None and not cfg.private and cfg.clip is None
