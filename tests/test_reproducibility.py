from config_helpers import load_config
"""Synthetic inputs only: compare interrupted engine with uninterrupted execution."""
import copy
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from scipy.linalg import toeplitz
from adam_mf.config import Trial
from adam_mf.audit import read_json
from adam_mf.training.checkpoint import rng_state,restore_rng

class AugmentedSynthetic(Dataset):
    def __len__(self):return 20
    def __getitem__(self,i):
        return torch.tensor([i/20,random.random(),float(np.random.rand())])+torch.rand(3),torch.tensor(i%2)

@pytest.mark.parametrize('method',['adam-iid','momentum-iid','adam-mf-sgd-b2lt','adam-mf-sgd-stp-b2lt','stpex-mf-momentum-bias-b2lt','momentum-mf-momentum-b2lt'])
def test_engine_exact_mid_epoch_resume(tmp_path,monkeypatch,method):
    # Exercise the actual engine on tiny CPU data; never a formal dataset/model run.
    import adam_mf.training.engine as e
    c=load_config();c['tasks']['cv'].update(epochs=2,total_steps=10,physical_batch_size=2,logical_batch_size=4)
    cfg=Trial('cv',method,c['seeds'][0],c);ident=dict(config_hash='synthetic',model_hashes={'cv':'synthetic'})
    original_device=torch.device
    # Patch the engine's device selection only, not global torch.device (Optimizer uses its type).
    original_getattr=e.torch
    class TorchProxy:
        def __getattr__(self,name):return original_getattr.__getattribute__(name)
        def device(self,*args):return original_device('cpu')
    monkeypatch.setattr(e,'torch',TorchProxy())
    for name in ('set_device','reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'synthetic-cpu')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    # Model needs head provenance like ViT, without registering the layer twice.
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(3,2)
        def forward(self,x):return self.head(x)
    monkeypatch.setattr(e,'pretrained_vit',lambda p:(Tiny(),{}))
    monkeypatch.setattr(e,'cv_datasets',lambda cfg,meta:(AugmentedSynthetic(),torch.utils.data.TensorDataset(torch.ones(4,3),torch.tensor([0,1,0,1])),np.arange(20)))
    matrix_path=tmp_path/'matrix.npz';np.savez(matrix_path,identity=np.eye(10))
    def matrix(*args):
        d=np.zeros(10);d[0]=1;d[1]=-.2;d[2]=-.08
        D=toeplitz(d,np.r_[1.,np.zeros(9)]) if cfg.mechanism=='b2lt' else np.eye(10)
        return dict(C=np.linalg.inv(D),D=D,A=np.eye(10),d=d,path=matrix_path,metadata=dict(P=3,theta=[.4],omega=[-.2]))
    monkeypatch.setattr(e,'load_matrix',matrix)
    full=tmp_path/'full';split=tmp_path/'split'
    for p in (full,split):p.mkdir()
    a=e.run_training(cfg,full,ident,0)
    original_save=e.save_checkpoint
    def interrupt(path,model,opt,logical,*args,**kwargs):
        original_save(path,model,opt,logical,*args,**kwargs)
        if logical.optimizer_steps==3:raise e.StopRequested('synthetic interruption')
    monkeypatch.setattr(e,'save_checkpoint',interrupt)
    with pytest.raises(e.StopRequested):e.run_training(cfg,split,ident,0)
    monkeypatch.setattr(e,'save_checkpoint',original_save)
    b=e.run_training(cfg,split,ident,0)
    def checkpoint(p):return torch.load(p/'checkpoints'/cfg.id/'latest.pt',weights_only=False)
    ca,cb=checkpoint(full),checkpoint(split)
    for k in ca['model']:torch.testing.assert_close(ca['model'][k],cb['model'][k],rtol=0,atol=0)
    for k in ('accuracy','loss','epsilon','augmentation_hashes','noise_draws','optimizer_steps'):assert a[k]==b[k]
    for ra,rb in zip(ca['progress']['rows'],cb['progress']['rows']):
        assert {k:v for k,v in ra.items() if k!='step_seconds'}=={k:v for k,v in rb.items() if k!='step_seconds'}
    # Tampering must fail, including after a task has already completed.
    bad=dict(ident,config_hash='changed')
    with pytest.raises(ValueError,match='identity'):e.verify_result(split,cfg,bad)
    from adam_mf.optimizers.builder import make_optimizer,logical_batch
    model=Tiny();opt=make_optimizer(model,cfg);logical=logical_batch(model,opt,cfg,None)
    with pytest.raises(ValueError,match='hash mismatch'):
        e.load_checkpoint(split/'checkpoints'/cfg.id/'latest.pt',model,opt,logical,bad)

def test_python_numpy_torch_rng_state_roundtrip(monkeypatch):
    # Normal preflight stays on CPU; GPU noise recovery has its own isolated tests.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    state=rng_state();expected=(random.random(),np.random.rand(),torch.rand(3))
    restore_rng(state);actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[:2]==actual[:2];torch.testing.assert_close(expected[2],actual[2],rtol=0,atol=0)
