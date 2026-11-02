"""Exp15b's exact uncorrected Momentum update, plus ordinary PyTorch Adam."""
import torch


class Momentum(torch.optim.Optimizer):
    def __init__(self,params,lr,beta1):
        super().__init__(params,dict(lr=lr,beta1=beta1))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state=self.state[p]
                if not state: state['exp_avg']=torch.zeros_like(p)
                b=group['beta1']
                u=b*state['exp_avg']+(1-b)*p.grad
                update=-group['lr']*u
                state['exp_avg'].copy_(u)
                p.add_(update)


class LogicalBatch:
    """Accumulate sums, add MF noise only for DP, divide once by 1000."""
    def __init__(self,model,optimizer,accumulation,logical_size=1000,noise=None):
        self.parameters=list(model.parameters());self.optimizer=optimizer
        self.accumulation=accumulation;self.logical_size=logical_size;self.noise=noise
        self.sums=[torch.zeros_like(p) for p in self.parameters]
        self.micro_steps=self.optimizer_steps=0
        self.last_gradient_norm=0.

    def begin_microbatch(self):
        if self.micro_steps==0:
            for total in self.sums: total.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def finish_microbatch(self):
        for total,p in zip(self.sums,self.parameters): total.add_(p.grad)
        self.micro_steps+=1
        if self.micro_steps<self.accumulation: return False
        assert self.micro_steps==self.accumulation
        if self.noise is not None:
            for total,z in zip(self.sums,self.noise.next()): total.add_(z)
        for total,p in zip(self.sums,self.parameters): p.grad=total/self.logical_size
        self.last_gradient_norm=float(torch.stack([p.grad.double().square().sum() for p in self.parameters]).sum().sqrt())
        if not all(torch.isfinite(p.grad).all() for p in self.parameters): raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.parameters): raise FloatingPointError('Nonfinite parameter')
        self.optimizer_steps+=1;self.micro_steps=0
        return True


def unclipped_microbatch(model,inputs,targets):
    loss=torch.nn.functional.cross_entropy(model(inputs),targets,reduction='sum')
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    loss.backward()
    return float(loss.detach()),0
