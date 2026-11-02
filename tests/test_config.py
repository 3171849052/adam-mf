from config_helpers import load_config
import copy
import pytest
from adam_mf.config import validate,plan
from adam_mf.registry import METHODS

def test_exact_formal_plan_and_frozen_values():
    c=load_config();jobs=plan(c)
    assert len(METHODS)==8 and set(c['tasks'])=={'cv','nlp'} and len(c['seeds'])==10
    assert len(jobs)==160 and len({j.id for j in jobs})==160
    expected={
      'adam-no-noise':[(.0003,None,None),(.001,None,None)],
      'momentum-no-noise':[(.05,None,None),(.1,None,None)],
      'adam-iid':[(.0007,30,None),(.002,.3,None)],
      'momentum-iid':[(2,.3,None),(2,.3,None)],
      'adam-mf-sgd-b2lt':[(.003,30,None),(.01,40,None)],
      'adam-mf-sgd-stp-b2lt':[(.003,200,.05),(.01,500,.001)],
      'stpex-mf-momentum-bias-b2lt':[(.1,10,3),(.1,20,.001)],
      'momentum-mf-momentum-b2lt':[(2,1,None),(.3,20,None)]}
    for method,params in expected.items():
        for task,values in zip(('cv','nlp'),params):
            j=next(j for j in jobs if j.task==task and j.method==method)
            assert (j.lr,j.clip,j.eps_scale)==values
            assert j.physical_batch==(250 if task=='cv' else 1000)
            assert j.task_config['logical_batch_size']==1000
            assert j.total_steps==(250 if task=='cv' else 310)
    assert c['training']==dict(beta1=.9,beta2=.999,adam_eps=1e-8,weight_decay=0)
    assert c['privacy']==dict(epsilon=8.,delta=1e-5,adjacency='add_remove_zero_out',sampling_amplification=False)

@pytest.mark.parametrize('mutation',[
 lambda c:c['methods'].pop('adam-iid'),
 lambda c:c['methods'].update({'unknown':c['methods']['adam-iid']}),
 lambda c:c.update(seeds=c['seeds'][:9]),
 lambda c:c['methods']['adam-no-noise']['cv'].update(clip=1),
 lambda c:c['methods']['adam-iid']['cv'].update(learning_rate=float('nan')),
 lambda c:c['tasks']['cv'].update(physical_batch_size=333),
 lambda c:c['tasks']['nlp'].update(total_steps=250),
 lambda c:c['privacy'].update(sampling_amplification=True),
 lambda c:c['runtime'].update(gpu_ids=[0,0]),
 lambda c:c['paths'].update(data='../dp-adam-correlated/data'),
 lambda c:c['methods']['adam-iid']['cv'].pop('clip'),
 lambda c:c.update(search={}),
])
def test_invalid_config_stops(mutation):
    c=copy.deepcopy(load_config());mutation(c)
    with pytest.raises((ValueError,TypeError)):validate(c)

def test_enable_flags_only_reduce_formal_plan():
    c=load_config();c['tasks']['cv']['enabled']=False;c['methods']['adam-iid']['enabled']=False
    assert len(plan(validate(c)))==70

def test_duplicate_yaml_keys_are_rejected(tmp_path):
    p=tmp_path/'duplicate.yaml';p.write_text('schema_version: 1\nschema_version: 1\n')
    with pytest.raises(ValueError,match='Duplicate YAML'):load_config(p)
