"""Standard Adam and Exp9's frozen STP scaling, with optional clipping/noise."""
import torch
from adam_mf.optimizers.finite import all_finite
from adam_mf.privacy.clipping import freeze_adam_scale

def adam_optimizer(model,cfg):return torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=cfg.betas,eps=cfg.adam_eps,weight_decay=cfg.weight_decay)
class AdamLogicalBatch:
    def __init__(self,model,optimizer,accumulation,logical_size=1000,noise=None,scaled=False,eps_scale=None):
        self.parameters=list(model.parameters());self.optimizer=optimizer;self.accumulation=accumulation
        self.logical_size=logical_size;self.noise=noise;self.scaled=scaled;self.eps_scale=eps_scale
        self.sums=[torch.zeros_like(p) for p in self.parameters];self.micro_steps=self.optimizer_steps=0
    def begin_microbatch(self):
        if self.micro_steps==0:
            if self.scaled:freeze_adam_scale(self.optimizer,self.eps_scale)
            for s in self.sums:s.zero_()
        self.optimizer.zero_grad(set_to_none=True)
    @torch.no_grad()
    def finish_microbatch(self):
        for s,p in zip(self.sums,self.parameters):s.add_(p.grad)
        self.micro_steps+=1
        if self.micro_steps<self.accumulation:return False
        z=self.noise.next() if self.noise else None
        for i,(s,p) in enumerate(zip(self.sums,self.parameters)):
            if self.scaled:s.mul_(p._logical_scale)
            if z is not None:s.add_(z[i])
            if self.scaled:s.div_(p._logical_scale)
            p.grad=s/self.logical_size
        self.last_gradient_norm=float(torch.stack([p.grad.double().square().sum() for p in self.parameters]).sum().sqrt())
        if not all_finite(p.grad for p in self.parameters):raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        if not all_finite(t for p in self.parameters for t in (p,self.optimizer.state[p]['exp_avg'],self.optimizer.state[p]['exp_avg_sq'])):raise FloatingPointError('Nonfinite Adam state')
        self.micro_steps=0;self.optimizer_steps+=1
        return True
def unclipped_microbatch(model,inputs,targets):
    loss=torch.nn.functional.cross_entropy(model(inputs),targets,reduction='sum')
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();return float(loss.detach()),0
