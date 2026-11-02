from config_helpers import load_config
import numpy as np
import torch
from adam_mf.paths import ROOT
from adam_mf.audit import verify_assets,read_json
from adam_mf.config import plan
from adam_mf.data.loaders import cv_datasets,nlp_datasets

def test_copied_assets_are_real_and_sha256_valid():
    assert len(verify_assets())==34
    for folder in ('data','model','assets/matrices'):
        m=read_json(ROOT/folder/'manifest.json')
        assert m['total_bytes']>0
        for row in m['files']:assert not (ROOT/row['path']).is_symlink()

def test_all_methods_share_formal_order():
    cfgs=plan(load_config());meta=dict(mean=(.5,)*3,std=(.5,)*3,crop_pct=.9)
    c=next(c for c in cfgs if c.task=='cv');training,valid,order=cv_datasets(c,meta)
    assert len(training)==50000 and len(valid)==10000
    g=torch.Generator().manual_seed(c.seed)
    np.testing.assert_array_equal(order,torch.randperm(50000,generator=g).numpy())
    c=next(c for c in cfgs if c.task=='nlp');training,valid,order=nlp_datasets(c)
    assert len(training)==62000 and len(valid)==872
    assert training.tensors[0].shape==(62000,128)
    with np.load(ROOT/'data/preprocessed/nlp_split.npz') as s:
        assert len(set(s['train'])&set(s['search_validation']))==0
        expected=s['train'][torch.randperm(62000,generator=torch.Generator().manual_seed(c.seed)).numpy()]
    np.testing.assert_array_equal(order,expected)

def test_local_tokenizer_matches_copied_tokens():
    import pandas as pd
    from transformers import BertTokenizer
    raw=pd.read_parquet(ROOT/'data/sst2/train.parquet').iloc[[0,123,4567]]
    tok=BertTokenizer.from_pretrained(str(ROOT/'model/bert-tiny'),local_files_only=True)
    actual=tok(raw['sentence'].tolist(),padding='max_length',max_length=128,truncation=True,return_tensors='pt')
    old=torch.load(ROOT/'data/preprocessed/nlp_tokens_128.pt',weights_only=True)
    for k,v in actual.items():torch.testing.assert_close(v,old[k][[0,123,4567]],rtol=0,atol=0)
