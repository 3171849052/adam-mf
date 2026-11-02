"""FP32 online noise: IID or exact B²LT exponential Q-buffer."""
import numpy as np
import torch
class OnlineNoise:
    def __init__(self,parameters,matrix,mechanism,sigma,total_steps,seed):
        if mechanism not in ('iid','b2lt'): raise ValueError(mechanism)
        self.parameters=list(parameters);self.device=self.parameters[0].device
        self.d=np.ones(1) if mechanism=='iid' else matrix['d'];self.meta=matrix['metadata'];self.mechanism=mechanism
        self.innovation_std=float(sigma);self.total_steps=total_steps;self.step_count=0
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.history=[];self.states=[]
        if mechanism=='b2lt':
            self.theta=self.meta['theta'];self.omega=self.meta['omega'];self.P=self.meta['P']
            self.buffers=[[torch.zeros_like(p) for p in self.parameters] for _ in self.theta]
            self.replay=torch.Generator(device=self.device)
    def innovations(self,generator):
        return [torch.randn(p.shape,device=p.device,dtype=torch.float32,generator=generator)*self.innovation_std for p in self.parameters]
    @torch.no_grad()
    def next(self):
        t=self.step_count
        if t>=self.total_steps:raise ValueError('Noise horizon exhausted')
        if self.mechanism=='b2lt':self.states.append(self.generator.get_state())
        innovations=self.innovations(self.generator)
        if self.mechanism=='b2lt':
            out=[z.clone() for z in innovations]
            for buf,om in zip(self.buffers,self.omega):
                for y,s in zip(out,buf):y.add_(s,alpha=om)
            old=None
            if t-self.P+1>=0:
                self.replay.set_state(self.states[t-self.P+1]);old=self.innovations(self.replay)
            for buf,th in zip(self.buffers,self.theta):
                for i,(s,z) in enumerate(zip(buf,innovations)):
                    s.mul_(th).add_(z)
                    if old is not None:s.add_(old[i],alpha=-th**(self.P-1))
        else:
            out=[z*float(self.d[0]) for z in innovations]
            for lag,old in enumerate(self.history,1):
                for y,z in zip(out,old):y.add_(z,alpha=float(self.d[lag]))
            self.history.insert(0,innovations);del self.history[len(self.d)-1:]
        self.step_count+=1;return out
    def marginal_std(self,step):return self.innovation_std*float(np.linalg.norm(self.d[:step+1]))

    def state_dict(self):
        return dict(step_count=self.step_count,generator=self.generator.get_state(),
                    states=self.states,history=self.history,
                    buffers=getattr(self,'buffers',None))
    def load_state_dict(self,state):
        self.step_count=state['step_count'];self.generator.set_state(state['generator'].cpu())
        self.states=[x.cpu() for x in state['states']]
        self.history=[[t.to(self.device) for t in row] for row in state['history']]
        if self.mechanism=='b2lt':self.buffers=[[t.to(self.device) for t in row] for row in state['buffers']]
