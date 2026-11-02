"""Formal-only offline data protocol, migrated from exp15c/exp16e.data."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,Subset,DataLoader
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import check_integrity
from adam_mf.paths import local_path
class OfflineCIFAR100(CIFAR100):
    def _check_integrity(self):
        files=self.train_list if self.train else self.test_list
        return all(check_integrity(str(Path(self.root)/self.base_folder/name),md5) for name,md5 in files)
def cv_datasets(cfg,metadata):
    from torchvision import transforms
    bicubic=transforms.InterpolationMode.BICUBIC
    norm=transforms.Normalize(metadata['mean'],metadata['std'])
    train_transform=transforms.Compose([transforms.RandomResizedCrop(224,interpolation=bicubic),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
    eval_transform=transforms.Compose([transforms.Resize(int(224/metadata['crop_pct']),interpolation=bicubic),transforms.CenterCrop(224),transforms.ToTensor(),norm])
    root=local_path(cfg.config['paths']['data'])/'cifar100'
    train=OfflineCIFAR100(root,train=True,transform=train_transform,download=False)
    valid=OfflineCIFAR100(root,train=False,transform=eval_transform,download=False)
    if len(train)!=50000 or len(valid)!=10000:raise ValueError('CV counts changed')
    indices=np.arange(50000)
    order=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    return Subset(train,order.tolist()),valid,order

def nlp_datasets(cfg):
    root=local_path(cfg.config['paths']['preprocessed'])
    inputs=torch.load(root/'nlp_tokens_128.pt',weights_only=True)
    official=torch.load(root/'nlp_official_tokens_128.pt',weights_only=True)
    keys=('input_ids','attention_mask','token_type_ids','labels')
    with np.load(root/'nlp_split.npz') as split:train=split['train'].copy()
    if len(train)!=62000 or len(official['labels'])!=872:raise ValueError('NLP counts changed')
    order=train[torch.randperm(len(train),generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    return TensorDataset(*(inputs[k][torch.from_numpy(order)] for k in keys)),TensorDataset(*(official[k] for k in keys)),order

def make_loaders(cfg,training,validation):
    kwargs=dict(batch_size=cfg.physical_batch,shuffle=False,pin_memory=True,num_workers=cfg.task_config['loader_workers'],generator=torch.Generator().manual_seed(cfg.seed))
    if kwargs['num_workers']:kwargs['multiprocessing_context']='spawn'
    # exp15c evaluates in batches of 100; exp16e CV evaluates in batches of 250.
    eval_size=cfg.physical_batch if cfg.task=='cv' and cfg.private else 100
    return DataLoader(training,**kwargs),DataLoader(validation,batch_size=eval_size,shuffle=False)
