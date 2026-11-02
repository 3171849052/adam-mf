from config_helpers import load_config
import os
from adam_mf.paths import ROOT,local_path
"""Explicit per-example autograd is the independent clipping reference."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel
from adam_mf.models.bert_tiny import Classifier, seed_all, make_model
from adam_mf.privacy.clipping import (StandardGhostModule, ScaledGhostModule, embedding_norm,
                            freeze_adam_scale, clipped_microbatch)
from adam_mf.optimizers.adam import AdamLogicalBatch as LogicalBatch


def tiny():
    cfg = BertConfig(vocab_size=19,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
                     intermediate_size=24,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    cfg._attn_implementation = 'eager'
    return Classifier(BertModel(cfg))


@pytest.mark.parametrize('scaled',[False,True])
@pytest.mark.parametrize('padding_idx',[None,0])
@pytest.mark.parametrize('device',(['cpu','cuda:0'] if os.environ.get('ADAM_MF_TEST_CUDA')=='1' and torch.cuda.is_available() else ['cpu']))
def test_embedding_explicit_repeated_tokens(scaled,padding_idx,device):
    seed_all(42)
    layer = nn.Embedding(19,7,padding_idx=padding_idx).to(device)
    layer.weight._logical_scale = torch.rand_like(layer.weight)*3+.01
    ids = torch.tensor([[0,2,2,2,5,0],[3,3,1,3,0,1],[0,0,0,0,0,0]],device=device)
    backprops = torch.randn(3,6,7,device=device)
    reference=[]
    for x,b in zip(ids,backprops):
        layer.zero_grad()
        (layer(x)*b).sum().backward()
        g = layer.weight.grad * layer.weight._logical_scale if scaled else layer.weight.grad
        reference.append(g.norm())
    actual=embedding_norm(layer,[ids],backprops,scaled)[layer.weight]
    torch.testing.assert_close(actual,torch.stack(reference),rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('architecture',['small','pretrained'])
@pytest.mark.parametrize('geometry',['standard','scale'])
@pytest.mark.parametrize('device',(['cpu','cuda:0'] if os.environ.get('ADAM_MF_TEST_CUDA')=='1' and torch.cuda.is_available() else ['cpu']))
def test_bert_norm_coefficient_and_accumulation(geometry,device,architecture):
    seed_all(43)
    base = (tiny() if architecture=='small' else make_model(local_path(load_config()['paths']['nlp_model']),dropout=0.)).to(device)
    C=1.3
    wrapper = ScaledGhostModule if geometry=='scale' else StandardGhostModule
    model = wrapper(copy.deepcopy(base),C)
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    # Nonuniform completed second moment, including embedding coordinates.
    for p in model.parameters():
        opt.state[p].update(step=torch.tensor(3.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.001)
    if geometry=='scale': freeze_adam_scale(opt,.1)
    ids=torch.tensor([[1,2,2,0,0,0],[3,3,4,3,0,0],[1,1,1,1,1,0],[5,6,7,7,0,0]],device=device)
    inputs=(ids,(ids!=0).long(),torch.zeros_like(ids))
    labels=torch.tensor([0,1,1,0],device=device)
    totals=[torch.zeros_like(p) for p in base.parameters()]
    expected_norms=[]
    for i in range(4):
        base.zero_grad()
        logits=base(tuple(t[i:i+1] for t in inputs))
        nn.functional.cross_entropy(logits,labels[i:i+1]).backward()
        grads=[p.grad*q._logical_scale if geometry=='scale' else p.grad for p,q in zip(base.parameters(),model.parameters())]
        norm=sum(g.square().sum() for g in grads).sqrt()
        expected_norms.append(norm)
        for total,p in zip(totals,base.parameters()): total.add_(p.grad*min(1.,C/norm.item()))
    actual_norms=[]; actual_coefs=[]
    sums=[torch.zeros_like(p) for p in model.parameters()]
    for start in (0,2):
        model.zero_grad(set_to_none=True)
        clipped_microbatch(model,tuple(t[start:start+2] for t in inputs),labels[start:start+2],C)
        actual_norms.extend(model.last_norms)
        actual_coefs.extend(model.last_coefficients)
        for total,p in zip(sums,model.parameters()): total.add_(p.grad)
    expected_norms=torch.stack(expected_norms)
    actual_norms=torch.stack(actual_norms)
    expected_coefs=(C/expected_norms).clamp(max=1.)
    torch.testing.assert_close(actual_norms,expected_norms,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(torch.stack(actual_coefs),expected_coefs,rtol=3e-5,atol=3e-6)
    for actual,expected in zip(sums,totals): torch.testing.assert_close(actual,expected,rtol=5e-5,atol=3e-6)
    # The full batch must produce the same clipped aggregate as two microbatches.
    model.zero_grad(set_to_none=True)
    clipped_microbatch(model,inputs,labels,C)
    for p,expected in zip(model.parameters(),totals): torch.testing.assert_close(p.grad,expected,rtol=5e-5,atol=3e-6)



@pytest.mark.parametrize('scaled',[False,True])
def test_one_noise_per_logical_and_previous_vhat(scaled):
    seed_all(44)
    model=nn.Linear(3,2)
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    class Noise:
        count=0
        def next(self):
            self.count+=1
            return [torch.full_like(p,2.) for p in model.parameters()]
    noise=Noise()
    logical=LogicalBatch(model,opt,4,1000,noise,scaled=scaled,eps_scale=.1)
    for step in range(3):
        previous=[torch.zeros_like(p) if step==0 else opt.state[p]['exp_avg_sq']/(1-.999**step) for p in model.parameters()]
        for micro in range(4):
            logical.begin_microbatch()
            if scaled:
                for p,v in zip(model.parameters(),previous): torch.testing.assert_close(p._logical_scale,1/(v.sqrt()+.1))
            scales=[p._logical_scale.clone() if scaled else torch.ones_like(p) for p in model.parameters()]
            for p in model.parameters(): p.grad=torch.ones_like(p)
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==logical.optimizer_steps==step+int(micro==3)
        for p,s in zip(model.parameters(),scales): torch.testing.assert_close(p.grad,(4+2/s)/1000)
