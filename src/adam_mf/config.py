"""Validated single configuration, formal identities only."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml
from adam_mf.paths import ROOT,local_path
from adam_mf.registry import METHODS
from adam_mf.audit import json_hash

def exact(value,keys,name):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError(f'{name}: expected keys {keys}')
def positive(value,name,integer=False):
    if isinstance(value,bool) or not isinstance(value,int if integer else (int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(f'{name}: positive {"integer" if integer else "number"} required')
def boolean(v,name):
    if type(v) is not bool:raise ValueError(f'{name}: boolean required')
def validate(c):
    exact(c,('schema_version','split_seed','seeds','training','privacy','tasks','methods','runtime','paths'),'root')
    if c['schema_version']!=1 or c['split_seed']!=20261008 or c['seeds']!=list(range(20261111,20261121)):raise ValueError('Invalid protocol version/split/ten formal seeds')
    exact(c['training'],('beta1','beta2','adam_eps','weight_decay'),'training')
    for key,v in c['training'].items():
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError(key)
    if not (0<c['training']['beta1']<1 and 0<c['training']['beta2']<1 and c['training']['adam_eps']>0 and c['training']['weight_decay']==0):raise ValueError('Invalid optimizer constants')
    exact(c['privacy'],('epsilon','delta','adjacency','sampling_amplification'),'privacy')
    p=c['privacy'];positive(p['epsilon'],'epsilon')
    if not 0<p['delta']<1 or p['adjacency']!='add_remove_zero_out' or p['sampling_amplification'] is not False:raise ValueError('Unsupported privacy protocol')
    exact(c['tasks'],('cv','nlp'),'tasks')
    for task,t in c['tasks'].items():
        exact(t,('enabled','dataset','model','physical_batch_size','logical_batch_size','epochs','total_steps','precision','full_finetuning','loader_workers'),task)
        boolean(t['enabled'],task)
        for key in ('physical_batch_size','logical_batch_size','epochs','total_steps'):positive(t[key],key,True)
        n=50000 if task=='cv' else 62000
        if t['dataset']!=('cifar100' if task=='cv' else 'sst2') or t['model']!=('vit_tiny' if task=='cv' else 'bert_tiny'):raise ValueError('Dataset/model mismatch')
        if t['total_steps'] not in (250,310) or t['epochs']!=5 or n%t['logical_batch_size'] or t['total_steps']!=t['epochs']*n//t['logical_batch_size']:raise ValueError('Horizon incompatible with frozen matrices and participation')
        if t['logical_batch_size']%t['physical_batch_size'] or t['precision']!='fp32' or t['full_finetuning'] is not True:raise ValueError('Batch/FP32/full finetuning protocol')
        if type(t['loader_workers']) is not int or t['loader_workers']!=(2 if task=='cv' else 0):raise ValueError('Loader worker count changes RNG protocol')
    exact(c['methods'],METHODS,'methods')
    for name,m in c['methods'].items():
        exact(m,('enabled','cv','nlp'),name);boolean(m['enabled'],name)
        for task in ('cv','nlp'):
            v=m[task];exact(v,('learning_rate','clip','eps_scale'),name+'/'+task)
            positive(v['learning_rate'],'learning_rate')
            if METHODS[name].private:positive(v['clip'],'clip')
            elif v['clip'] is not None:raise ValueError('No-noise methods cannot clip')
            if METHODS[name].scaled:positive(v['eps_scale'],'eps_scale')
            elif v['eps_scale'] is not None:raise ValueError('Scale only for STP methods')
    r=c['runtime'];exact(r,('gpu_ids','cv_workers_per_gpu','nlp_workers_per_gpu','max_parallel_tasks','min_free_memory_mb','poll_seconds','resume','stop_on_error','checkpoint_every_steps'),'runtime')
    if r['gpu_ids']!='auto' and (not isinstance(r['gpu_ids'],list) or not r['gpu_ids'] or len(set(r['gpu_ids']))!=len(r['gpu_ids']) or any(type(i) is not int or i<0 for i in r['gpu_ids'])):raise ValueError('Invalid GPU IDs')
    for key in ('cv_workers_per_gpu','nlp_workers_per_gpu','max_parallel_tasks','checkpoint_every_steps'):positive(r[key],key,True)
    positive(r['poll_seconds'],'poll_seconds');boolean(r['resume'],'resume');boolean(r['stop_on_error'],'stop_on_error')
    exact(r['min_free_memory_mb'],('cv','nlp'),'memory')
    for v in r['min_free_memory_mb'].values():positive(v,'memory',True)
    exact(c['paths'],('data','preprocessed','cv_model','nlp_model','matrices','outputs','logs'),'paths')
    for v in c['paths'].values():
        if not isinstance(v,str) or not v:raise ValueError('Invalid path')
        local_path(v)
    if not any(t['enabled'] for t in c['tasks'].values()) or not any(m['enabled'] for m in c['methods'].values()):raise ValueError('Empty experiment')
    return c

class StrictLoader(yaml.SafeLoader):pass
def _mapping(loader,node,deep=False):
    result={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if key in result:raise ValueError(f'Duplicate YAML key: {key}')
        result[key]=loader.construct_object(value_node,deep=deep)
    return result
StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)
def load_config(path=None):return validate(yaml.load((Path(path) if path else ROOT/'config/experiment.yaml').read_text(),Loader=StrictLoader))
@dataclass(frozen=True)
class Trial:
    task:str
    method:str
    seed:int
    config:dict
    @property
    def spec(self):return METHODS[self.method]
    @property
    def task_config(self):return self.config['tasks'][self.task]
    @property
    def parameters(self):return self.config['methods'][self.method][self.task]
    @property
    def lr(self):return self.parameters['learning_rate']
    @property
    def clip(self):return self.parameters['clip']
    @property
    def eps_scale(self):return self.parameters['eps_scale']
    @property
    def private(self):return self.spec.private
    @property
    def scaled(self):return self.spec.scaled
    @property
    def mechanism(self):return self.spec.mechanism
    @property
    def workload(self):return self.spec.workload
    @property
    def betas(self):return (self.config['training']['beta1'],self.config['training']['beta2'])
    @property
    def adam_eps(self):return self.config['training']['adam_eps']
    @property
    def weight_decay(self):return self.config['training']['weight_decay']
    @property
    def total_steps(self):return self.task_config['total_steps']
    @property
    def physical_batch(self):return self.task_config['physical_batch_size']
    @property
    def spacing(self):return self.total_steps//self.task_config['epochs']
    @property
    def id(self):return f'{self.task}__{self.method}__{self.seed}'
    def values(self):return dict(task=self.task,method=self.method,seed=self.seed)
    def protocol(self):
        p=self.config['privacy']
        return dict(privacy=dict(p,k=self.task_config['epochs'],b_participation=self.spacing,max_grad_norm=self.clip))
def plan(c):
    return [Trial(task,method,seed,c) for task,t in c['tasks'].items() if t['enabled'] for method,m in c['methods'].items() if m['enabled'] for seed in c['seeds']]
