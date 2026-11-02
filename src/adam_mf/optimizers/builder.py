from adam_mf.optimizers.adam import adam_optimizer,AdamLogicalBatch
from adam_mf.optimizers.momentum import Momentum,LogicalBatch as MomentumLogicalBatch
from adam_mf.optimizers.stpex import make_optimizer as stpex_optimizer,LogicalBatch as StpexLogicalBatch

def make_optimizer(model,cfg):
    if cfg.spec.optimizer=='momentum':return Momentum(model.parameters(),cfg.lr,cfg.betas[0])
    if cfg.spec.optimizer=='stpex':return stpex_optimizer(model,cfg)
    return adam_optimizer(model,cfg)
def logical_batch(model,opt,cfg,noise):
    size=cfg.task_config['logical_batch_size'];args=(model,opt,size//cfg.physical_batch,size,noise)
    if cfg.spec.optimizer=='momentum':return MomentumLogicalBatch(*args)
    cls=StpexLogicalBatch if cfg.spec.optimizer=='stpex' else AdamLogicalBatch
    return cls(*args,scaled=cfg.scaled,eps_scale=cfg.eps_scale)
