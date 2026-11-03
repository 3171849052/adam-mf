from config_helpers import load_config
"""Independent synthetic records under pytest tempdirs; never formal outputs."""
import csv
from adam_mf.audit import atomic_json,json_hash,read_json
from adam_mf.config import plan
from adam_mf.reporting.report import report
from adam_mf.training.engine import result_path

def test_complete_120_record_statistics_plots_and_audit(tmp_path):
    c=load_config();jobs=plan(c);identity=dict(config_hash='synthetic',model_hashes={'cv':'synthetic-cv','nlp':'synthetic-nlp'})
    for j in jobs:
        eps=c['privacy']['epsilon'] if j.private else None
        r=dict(**j.values(),learning_rate=j.lr,clip=j.clip,eps_scale=j.eps_scale,status='completed',identity=identity,
               config_hash=identity['config_hash'],model_hash=identity['model_hashes'][j.task],
               optimizer_steps=j.total_steps,noise_draws=j.total_steps if j.private else 0,
               accuracy=.6+(j.seed-c['seeds'][0])*.001,loss=.7,
               epsilon=eps,delta=c['privacy']['delta'] if j.private else None,clip_fraction=.2 if j.private else None,
               training_time=1.,peak_memory=1024,gpu_id=0,gpu_name='synthetic',
               epochs=[dict(epoch=i,accuracy=.5+i*.01) for i in range(1,6)],artifact_sha256={},
               initialization_hash='same',head_hash='same',train_order_hash='same',augmentation_hashes=['same'])
        r['record_hash']=json_hash(r);atomic_json(result_path(tmp_path,j),r)
    audit=report(tmp_path,c,jobs,identity)
    assert audit['status']=='passed' and audit['completed']==120
    with (tmp_path/'formal_per_seed.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==120 and len({(r['task'],r['method'],r['seed']) for r in rows})==120
    assert all(float(r['epsilon'])==c['privacy']['epsilon'] and float(r['delta'])==c['privacy']['delta'] for r in rows)
    with (tmp_path/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    assert len(summary)==12 and all(r['completed_seeds']=='10' for r in summary)
    assert len(list((tmp_path/'figures').glob('*.png')))==4
    assert (tmp_path/'final_report.md').is_file()

def test_missing_runs_are_never_filled_with_historical_results(tmp_path):
    c=load_config();a=report(tmp_path,c,plan(c),{'config_hash':'synthetic'})
    assert a['status']=='incomplete' and a['completed']==0
    with (tmp_path/'formal_per_seed.csv').open() as f:assert list(csv.DictReader(f))==[]
