"""Offline reconstruction of the copied inputs; compare every tensor and split."""
import pickle
import numpy as np
import torch
from adam_mf.paths import local_path

def verify_preprocessing(config):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    p=config['paths'];root=local_path(p['data']);cache=local_path(p['preprocessed']);seed=config['split_seed']
    tokenizer=BertTokenizer.from_pretrained(str(local_path(p['nlp_model'])),local_files_only=True)
    for filename,expected,count in [('train.parquet','nlp_tokens_128.pt',67349),('validation.parquet','nlp_official_tokens_128.pt',872)]:
        raw=pd.read_parquet(root/'sst2'/filename)
        if len(raw)!=count:raise ValueError('Raw SST-2 count')
        tokens=dict(tokenizer(raw['sentence'].tolist(),padding='max_length',max_length=128,truncation=True,return_tensors='pt'),labels=torch.tensor(raw['label'].tolist()))
        old=torch.load(cache/expected,weights_only=True)
        if set(tokens)!=set(old) or any(not torch.equal(tokens[k],old[k]) for k in old):raise ValueError('Tokenizer output differs from original')
        if filename=='train.parquet':
            train,valid=train_test_split(np.arange(count),train_size=62000,random_state=seed,stratify=raw['label'])
            arrays=dict(train=np.random.default_rng(seed+1).permutation(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))
            with np.load(cache/'nlp_split.npz') as old:
                for k in arrays:np.testing.assert_array_equal(arrays[k],old[k])
    with (root/'cifar100/cifar-100-python/train').open('rb') as f:labels=np.array(pickle.load(f,encoding='bytes')[b'fine_labels'])
    train,valid=train_test_split(np.arange(50000),train_size=45000,random_state=seed,stratify=labels)
    with np.load(cache/'cv_split.npz') as old:
        np.testing.assert_array_equal(np.sort(train),old['train']);np.testing.assert_array_equal(np.sort(valid),old['search_validation'])
    return dict(status='passed',token_tensors_equal=True,split_indices_equal=True)


def verify_prepared_content(config):
    """Pin tensor values, dtypes and shapes independently of torch ZIP metadata."""
    from adam_mf.assets import catalog,asset_path
    from adam_mf.audit import array_hash
    for row in catalog()['files']:
        if row['kind']!='prepared':continue
        path=asset_path(row,config)
        if path.suffix=='.pt':arrays={k:v.numpy() for k,v in torch.load(path,weights_only=True).items()}
        else:
            with np.load(path) as f:arrays={k:f[k] for k in f.files}
        actual={k:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=array_hash(a)) for k,a in arrays.items()}
        if actual!=row['content']:raise ValueError(f'Prepared tensor/split content mismatch: {path}')
    return True


def prepare_data(config):
    import os,tempfile
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    from adam_mf.assets import asset_lock,verify_assets,manifests
    with asset_lock(config):
        verify_assets(config,include_prepared=False)
        try:
            verify_prepared_content(config)
        except (OSError,ValueError,RuntimeError,EOFError):
            p=config['paths'];root=local_path(p['data']);cache=local_path(p['preprocessed']);cache.mkdir(parents=True,exist_ok=True)
            seed=config['split_seed'];tokenizer=BertTokenizer.from_pretrained(str(local_path(p['nlp_model'])),local_files_only=True)
            generated={}
            for source,target,count in [('train','nlp_tokens_128.pt',67349),('validation','nlp_official_tokens_128.pt',872)]:
                raw=pd.read_parquet(root/'sst2'/f'{source}.parquet')
                if len(raw)!=count:raise ValueError('SST-2 count mismatch')
                generated[target]=dict(tokenizer(raw['sentence'].tolist(),padding='max_length',max_length=128,truncation=True,return_tensors='pt'),labels=torch.tensor(raw['label'].tolist()))
                if source=='train':
                    train,valid=train_test_split(np.arange(count),train_size=62000,random_state=seed,stratify=raw['label'])
                    generated['nlp_split.npz']=dict(train=np.random.default_rng(seed+1).permutation(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))
            with (root/'cifar100/cifar-100-python/train').open('rb') as f:labels=np.array(pickle.load(f,encoding='bytes')[b'fine_labels'])
            train,valid=train_test_split(np.arange(50000),train_size=45000,random_state=seed,stratify=labels)
            generated['cv_split.npz']=dict(train=np.sort(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))
            # Verify every result against the original content before publishing any cache.
            from adam_mf.assets import catalog
            from adam_mf.audit import array_hash
            for row in catalog()['files']:
                if row['kind']!='prepared':continue
                name=row['path'].split('/')[-1];arrays=generated[name]
                actual={k:dict(shape=list(v.shape),dtype=str(v.numpy().dtype if torch.is_tensor(v) else v.dtype),sha256=array_hash(v.numpy() if torch.is_tensor(v) else v)) for k,v in arrays.items()}
                if actual!=row['content']:raise ValueError(f'Reconstructed original protocol mismatch: {name}')
            for name,value in generated.items():
                fd,tmp=tempfile.mkstemp(dir=cache)
                try:
                    with os.fdopen(fd,'wb') as f:
                        if name.endswith('.pt'):torch.save(value,f)
                        else:np.savez(f,**value)
                        f.flush();os.fsync(f.fileno())
                    os.replace(tmp,cache/name)
                finally:
                    if os.path.exists(tmp):os.unlink(tmp)
        verify_prepared_content(config);manifests(config,prepared=True)
    return verify_assets(config)
