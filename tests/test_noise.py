import numpy as np
import pytest
import torch
from adam_mf.paths import ROOT
from adam_mf.privacy.calibration import load_matrix,workload
from adam_mf.privacy.noise_b2lt import OnlineNoise

@pytest.mark.parametrize('n',[250,310])
@pytest.mark.parametrize('work,mech',[('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')])
def test_full_noise_stream_matches_dense_matrix(n,work,mech):
    a=load_matrix(ROOT/'assets/matrices',work,mech,n,.9)
    np.testing.assert_allclose(a['A'],workload(work,n,.9),rtol=1e-13)
    p=torch.zeros(7);noise=OnlineNoise([p],a,mech,1.7,n,73);g=torch.Generator().manual_seed(73);xs=[];ys=[]
    for i in range(n):
        xs.append(torch.randn(p.shape,generator=g)*1.7);ys.append(noise.next()[0].clone())
        torch.manual_seed(i);torch.randn(5)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],dtype=torch.float32)@torch.stack(xs),rtol=3e-5,atol=3e-5)
    with pytest.raises(ValueError):noise.next()

def test_iid_distribution_and_independent_rng():
    matrix=load_matrix(ROOT/'assets/matrices','identity','iid',250,.9)
    p=torch.zeros(12000);noise=OnlineNoise([p],matrix,'iid',2.,250,17)
    x=noise.next()[0];torch.randn(100);y=noise.next()[0]
    assert abs(float(x.mean()))<.06 and abs(float(x.std())-2)<.05
    assert abs(float(torch.corrcoef(torch.stack([x,y]))[0,1]))<.04

@pytest.mark.parametrize('mechanism',['iid','b2lt'])
@pytest.mark.parametrize('device',(['cpu','cuda:0'] if __import__('os').environ.get('ADAM_MF_TEST_CUDA')=='1' and torch.cuda.is_available() else ['cpu']))
def test_noise_checkpoint_including_old_innovation_replay(mechanism,device):
    import io
    work='identity' if mechanism=='iid' else 'sgd'
    a=load_matrix(ROOT/'assets/matrices',work,mechanism,250,.9)
    parameters=[torch.zeros(7,device=device),torch.zeros(2,3,device=device)]
    noise=OnlineNoise(parameters,a,mechanism,1.3,250,71)
    for _ in range(73):noise.next()
    stream=io.BytesIO();torch.save(noise.state_dict(),stream);stream.seek(0)
    state=torch.load(stream,map_location='cpu',weights_only=True)
    resumed=OnlineNoise(parameters,a,mechanism,1.3,250,71);resumed.load_state_dict(state)
    for _ in range(177):
        for expected,actual in zip(noise.next(),resumed.next()):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert resumed.step_count==250
