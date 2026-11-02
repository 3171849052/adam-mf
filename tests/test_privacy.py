from config_helpers import load_config
import numpy as np
import pytest
from adam_mf.config import plan
from adam_mf.paths import ROOT,local_path
from adam_mf.privacy.calibration import load_matrix
from adam_mf.privacy.accountant import calibrate,epsilon_from_mu,fixed_epoch_sensitivity,target_mu,gdp_delta

@pytest.mark.parametrize('task',['cv','nlp'])
@pytest.mark.parametrize('method',['adam-iid','momentum-iid','adam-mf-sgd-b2lt','adam-mf-sgd-stp-b2lt','stpex-mf-momentum-bias-b2lt','momentum-mf-momentum-b2lt'])
def test_whole_training_calibration_and_prefixes(task,method):
    cfg=next(c for c in plan(load_config()) if c.task==task and c.method==method)
    a=load_matrix(local_path(load_config()['paths']['matrices']),cfg.workload,cfg.mechanism,cfg.total_steps,cfg.betas[0])
    cal=calibrate(a['C'],cfg.protocol());p=cfg.config['privacy'];eps=[]
    for step in (1,cfg.spacing,cfg.spacing+1,cfg.total_steps):
        sens=fixed_epoch_sensitivity(a['C'],5,cfg.spacing,steps=step)
        eps.append(epsilon_from_mu(cfg.clip*sens/cal['innovation_std_sum'],p['delta']))
    assert eps==sorted(eps) and eps[-1]==pytest.approx(p['epsilon'],abs=1e-8)
    assert cal['sampling_amplification'] is False
    assert gdp_delta(target_mu(p['epsilon'],p['delta']),p['epsilon'])==pytest.approx(p['delta'],abs=1e-12)

def test_absolute_gram_bound_independent_enumeration():
    # Different directions at different fixed participations, signed strategy.
    S=np.array([[1.,0.,0.,0.],[-.2,1,0,0],[.1,-.3,1,0],[-.4,.5,-.6,1]])
    actual=fixed_epoch_sensitivity(S,2,2)
    expected=max(np.sqrt(sum(abs(S[:,i]@S[:,j]) for i in ids for j in ids)) for ids in ([0,2],[1,3]))
    assert actual==pytest.approx(expected)
