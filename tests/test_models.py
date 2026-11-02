from config_helpers import load_config
import os
from adam_mf.paths import ROOT,local_path
import copy
import pytest
import torch
from adam_mf.privacy.clipping import StandardGhostModule,ScaledGhostModule,freeze_adam_scale,clipped_microbatch
from adam_mf.models.vit_tiny import ViTTiny,pretrained_vit,checkpoint_path
from adam_mf.models.bert_tiny import seed_all
@pytest.mark.parametrize('scaled',[False,True])
@pytest.mark.parametrize('device',(['cpu','cuda:0'] if os.environ.get('ADAM_MF_TEST_CUDA')=='1' and torch.cuda.is_available() else ['cpu']))
def test_cv_exact_clipping(scaled,device):
    seed_all(42)
    base=ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3).to(device)
    C=1.3;model=(ScaledGhostModule if scaled else StandardGhostModule)(copy.deepcopy(base),C)
    opt=torch.optim.Adam(model.parameters())
    for p in model.parameters(): opt.state[p].update(step=torch.tensor(3.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.001)
    if scaled: freeze_adam_scale(opt,.1)
    x=torch.randn(3,3,16,16,device=device);y=torch.tensor([0,1,2],device=device)
    totals=[torch.zeros_like(p) for p in base.parameters()];norms=[]
    for xi,yi in zip(x,y):
        base.zero_grad();torch.nn.functional.cross_entropy(base(xi[None]),yi[None]).backward()
        norm=sum((p.grad*q._logical_scale if scaled else p.grad).square().sum() for p,q in zip(base.parameters(),model.parameters())).sqrt()
        norms.append(norm)
        for total,p in zip(totals,base.parameters()): total.add_(p.grad*min(1.,C/norm.item()))
    clipped_microbatch(model,x,y,C)
    torch.testing.assert_close(model.last_norms,torch.stack(norms),rtol=5e-5,atol=3e-6)
    for p,total in zip(model.parameters(),totals): torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=3e-6)
    model.remove_hooks()

def test_pretrained_vit_matches_timm_logits():
    import timm
    from safetensors.torch import load_file
    seed_all(45);model,_=pretrained_vit(local_path(load_config()['paths']['cv_model']));model.eval()
    reference=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=100).eval()
    weights=load_file(str(checkpoint_path(local_path(load_config()['paths']['cv_model']))))
    weights={k:v for k,v in weights.items() if not k.startswith('head.')}
    weights.update({'head.weight':model.head.weight,'head.bias':model.head.bias})
    reference.load_state_dict(weights,strict=True)
    images=torch.randn(2,3,224,224)
    with torch.no_grad():
        # Patch convolution vs unfolded Linear have different FP32 accumulation order.
        torch.testing.assert_close(model(images),reference(images),rtol=3e-5,atol=1e-5)
        torch.testing.assert_close(model.double()(images.double()),reference.double()(images.double()),rtol=1e-9,atol=1e-10)
