import itertools
import numpy as np
from adam_mf.audit import read_json,write_csv,atomic_json,atomic_text,json_hash
from adam_mf.registry import METHODS
from adam_mf.reporting.statistics import statistics
from adam_mf.training.engine import verify_result,result_path
FIELDS=('task','method','seed','learning_rate','clip','eps_scale','accuracy','loss','epsilon','delta','clip_fraction','training_time','peak_memory','status','config_hash','model_hash','gpu_id','gpu_name','gpu_uuid','cuda_visible_devices','logical_cuda_device')

def report(run,config,jobs,identity):
    results=[];failures=[]
    for job in jobs:
        if verify_result(run,job,identity):results.append(read_json(result_path(run,job)))
        elif result_path(run,job).exists():failures.append(read_json(result_path(run,job)))
    rows=[{k:r.get(k) for k in FIELDS} for r in results+failures]
    write_csv(run/'formal_per_seed.csv',rows,FIELDS)
    summaries=[];pairs=[];pair_audit=[]
    for task in ('cv','nlp'):
        groups={m:[r for r in results if r['task']==task and r['method']==m] for m in METHODS}
        for method,rs in groups.items():
            if not rs:continue
            accuracy=statistics(r['accuracy'] for r in rs);loss=statistics(r['loss'] for r in rs)
            summaries.append(dict(task=task,method=method,completed_seeds=len(rs),accuracy_mean=accuracy['mean'],accuracy_sd=accuracy['sd'],ci95_low=accuracy['ci95_low'],ci95_high=accuracy['ci95_high'],loss_mean=loss['mean'],loss_sd=loss['sd'],training_seconds=sum(r['training_time'] for r in rs),peak_memory_bytes=max(r['peak_memory'] for r in rs)))
        for left,right in itertools.combinations(METHODS,2):
            ls={r['seed']:r for r in groups[left]};rs={r['seed']:r for r in groups[right]};seeds=sorted(set(ls)&set(rs))
            if not seeds:continue
            checks={s:all(ls[s][k]==rs[s][k] for k in ('initialization_hash','head_hash','train_order_hash','augmentation_hashes')) for s in seeds}
            pair_audit.append(dict(task=task,left=left,right=right,matched=all(checks.values()),seeds=seeds))
            if all(checks.values()):
                pairs.append(dict(task=task,left=left,right=right,direction='right_minus_left',**statistics(rs[s]['accuracy']-ls[s]['accuracy'] for s in seeds)))
    summary_fields=('task','method','completed_seeds','accuracy_mean','accuracy_sd','ci95_low','ci95_high','loss_mean','loss_sd','training_seconds','peak_memory_bytes')
    write_csv(run/'summary.csv',summaries,summary_fields)
    write_csv(run/'paired_differences.csv',pairs,('task','left','right','direction','n','mean','sd','ci95_low','ci95_high'))
    full_plan=len(jobs)==160 and len({(j.task,j.method) for j in jobs})==16
    complete=len(results)==len(jobs) and not failures and all({r['seed'] for r in results if r['task']==t and r['method']==m}==set(config['seeds']) for t in ('cv','nlp') for m in METHODS)
    audit=dict(status='passed' if complete and full_plan and all(x['matched'] for x in pair_audit) else 'incomplete',planned=len(jobs),completed=len(results),failed=len(failures),full_160_plan=full_plan,
               independent_new_training=True,all_artifact_hashes_verified=True,identity=identity,paired_rng_checks=pair_audit,
               seeds_per_group={f'{t}/{m}':sum(r['task']==t and r['method']==m for r in results) for t in ('cv','nlp') for m in METHODS})
    atomic_json(run/'audit/final_audit.json',audit)
    if results:plot(run,results,summaries)
    text=['# adam-mf 正式实验报告','',f'完整性状态：{audit["status"]}；完成 {len(results)}/{len(jobs)} 个任务。仅当完整计划的 160 项全部成功且审计通过时，才算完成复现。','',
          '## 实验配置','',f'完整配置见 config_snapshot.yaml；配置 SHA-256：`{identity["config_hash"]}`。',
          'CIFAR-100 / ViT-Tiny；SST-2 / BERT-Tiny。固定划分、相同十个种子、五轮、FP32、全参数微调。每步参数和裁剪阈值固定，无学习率调度。','',
          '## 八种方法','']
    for name,m in METHODS.items():text.append(f'- {m.label}（`{name}`）：{m.optimizer}；'+(f'{m.mechanism} 噪声，{m.workload} workload，'+('缩放空间裁剪。' if m.scaled else '原空间裁剪。') if m.private else '普通优化器，无裁剪、无噪声、无 DP 包装器。'))
    for task in ('cv','nlp'):
        text += ['',f'## {task.upper()} 十种子结果','', '|方法|种子数|准确率均值 ± 样本SD (%)|95% Student-t CI (%)|交叉熵均值 ± SD|','|---|---:|---:|---|---|']
        for s in summaries:
            if s['task']!=task:continue
            sd='N/A' if s['accuracy_sd'] is None else f'{100*s["accuracy_sd"]:.3f}'
            ci='N/A' if s['ci95_low'] is None else f'[{100*s["ci95_low"]:.3f}, {100*s["ci95_high"]:.3f}]'
            lsd='N/A' if s['loss_sd'] is None else f'{s["loss_sd"]:.6f}'
            text.append(f'|{METHODS[s["method"]].label}|{s["completed_seeds"]}|{100*s["accuracy_mean"]:.3f} ± {sd}|{ci}|{s["loss_mean"]:.6f} ± {lsd}|')
    text += ['','## 方法比较','', '全部可配对方法的右减左差值与 95% Student-t 区间见 paired_differences.csv；配对前验证初始化、分类头、顺序和增强轨迹。NLP dropout 按 seed+100000+logical_step 重置。区间没有多重比较校正。各方法使用独立冻结参数，差异不能解释为单一机制的因果效应。官方评价集历史上已使用，本次不是新盲测。','',
             '## GPU 与耗时','',f'累计训练步骤耗时 {sum(r["training_time"] for r in results):.1f} 秒。每种方法的耗时、峰值显存见 summary.csv；逐任务 GPU 名称和编号见 formal_per_seed.csv。',
             '', '## 隐私预算','',f'六种 DP 方法每次完整训练目标 ε={config["privacy"]["epsilon"]}、δ={config["privacy"]["delta"]}，GDP 校准，固定五轮参与，add/remove zero-out 邻接，不使用采样放大。十次训练或多种方法的组合不具有相同单次 ε 保证。无噪声组 ε、δ、裁剪比例留空，表示不适用。eps_scale 是内部缩放参数。梯度和裁剪诊断属于内部审计信息，不具有额外隐私发布保证。',
             '', '## 完整性审计','', 'audit/final_audit.json 记录任务数、逐组完成数、资产/配置/代码/环境身份与配对随机轨迹检查。所有数值仅来自本仓库新执行的任务，未导入历史准确率。']
    atomic_text(run/'final_report.md','\n'.join(text)+'\n')
    return audit

def plot(run,results,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=run/'figures';folder.mkdir(exist_ok=True)
    for task in ('cv','nlp'):
        fig,ax=plt.subplots(figsize=(11,6))
        for method in METHODS:
            rs=[r for r in results if r['task']==task and r['method']==method]
            if not rs:continue
            epochs=sorted({e['epoch'] for r in rs for e in r['epochs']})
            values=[np.array([e['accuracy'] for r in rs for e in r['epochs'] if e['epoch']==ep])*100 for ep in epochs]
            mean=np.array([v.mean() for v in values]);sd=np.array([v.std(ddof=1) if len(v)>1 else 0 for v in values])
            ax.plot(epochs,mean,label=method);ax.fill_between(epochs,mean-sd,mean+sd,alpha=.1)
        ax.set(xlabel='Epoch',ylabel='Accuracy (%)',title=f'{task.upper()} mean ± sample SD');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(folder/f'{task}_accuracy.png',dpi=160);plt.close(fig)
        ss=[s for s in summaries if s['task']==task]
        fig,ax=plt.subplots(figsize=(11,7))
        if ss:
            means=[s['accuracy_mean']*100 for s in ss]
            errors=[(s['ci95_high']-s['accuracy_mean'])*100 if s['ci95_high'] is not None else 0 for s in ss]
            ax.barh([s['method'] for s in ss],means,xerr=errors)
        ax.set(xlabel='Accuracy (%)',title=f'{task.upper()} mean and 95% Student-t CI');fig.tight_layout();fig.savefig(folder/f'{task}_comparison.png',dpi=160);plt.close(fig)
