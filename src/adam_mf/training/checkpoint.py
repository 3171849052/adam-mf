"""Commit only complete logical steps; replay DataLoader worker RNG on resume."""
import os,random,tempfile
import numpy as np
import torch

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
def save_checkpoint(path,model,optimizer,logical,progress,identity,loader_state,epoch_start_loader_state):
    if logical.micro_steps:raise ValueError('Cannot checkpoint a partial logical batch')
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),logical_step=logical.optimizer_steps,
               noise=logical.noise.state_dict() if logical.noise else None,progress=progress,identity=identity,
               sampler=dict(loader_generator=loader_state,epoch_start_generator=epoch_start_loader_state,
                            epoch=progress['epoch'],next_microbatch=progress['next_microbatch'],order_hash=progress['order_hash']),rng=rng_state())
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.');os.close(fd)
    try:
        torch.save(state,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def load_checkpoint(path,model,optimizer,logical,identity):
    # Only load our own trusted checkpoints. Python/NumPy RNG tuples need weights_only=False.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('Checkpoint config/code/model/asset hash mismatch')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    logical.optimizer_steps=state['logical_step'];logical.micro_steps=0
    if logical.noise:logical.noise.load_state_dict(state['noise'])
    if bool(logical.noise)!=(state['noise'] is not None):raise ValueError('Noise state mismatch')
    return state
