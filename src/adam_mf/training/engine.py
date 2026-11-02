"""Formal training, migrated update/data/RNG order; atomic logical-step recovery."""
import signal,time,traceback
import numpy as np
import torch
from adam_mf.paths import local_path
from adam_mf.audit import atomic_json,file_hash,array_hash,json_hash,read_json,write_csv
from adam_mf.models.bert_tiny import seed_all,make_model,digest
from adam_mf.models.vit_tiny import pretrained_vit,initialization_digest
from adam_mf.data.loaders import cv_datasets,nlp_datasets,make_loaders
from adam_mf.optimizers.builder import make_optimizer,logical_batch
from adam_mf.optimizers.adam import unclipped_microbatch
from adam_mf.privacy.clipping import StandardGhostModule,ScaledGhostModule,clipped_microbatch
from adam_mf.privacy.calibration import load_matrix
from adam_mf.privacy.accountant import calibrate,fixed_epoch_sensitivity,epsilon_from_mu
from adam_mf.privacy.noise_b2lt import OnlineNoise
from adam_mf.training.evaluate import evaluate
from adam_mf.training.checkpoint import save_checkpoint,load_checkpoint,restore_rng

class StopRequested(Exception):pass

def result_path(run,cfg):return run/'per_seed'/cfg.task/(cfg.id+'.json')
def verify_result(run,cfg,identity):
    path=result_path(run,cfg)
    if not path.exists():return False
    result=read_json(path)
    if result['status']!='completed':return False
    if result['identity']!=identity or result['optimizer_steps']!=cfg.total_steps or any(result[k]!=v for k,v in cfg.values().items()):raise ValueError(f'Completed result identity mismatch: {cfg.id}')
    if (result['learning_rate'],result['clip'],result['eps_scale'])!=(cfg.lr,cfg.clip,cfg.eps_scale):raise ValueError('Completed frozen parameters mismatch')
    if not np.isfinite([result['accuracy'],result['loss']]).all() or not 0<=result['accuracy']<=1 or len(result['epochs'])!=cfg.task_config['epochs']:raise ValueError('Invalid completed metrics')
    if identity.get('gpus'):
        physical=next((g for g in identity['gpus'] if g['id']==result['gpu_id']),None)
        if physical is None or result.get('gpu_uuid')!=physical['uuid'] or result.get('cuda_visible_devices')!=str(physical['id']) or result.get('logical_cuda_device')!='cuda:0':raise ValueError('Completed physical GPU mapping mismatch')
    if result['noise_draws']!=(cfg.total_steps if cfg.private else 0):raise ValueError('Completed noise count mismatch')
    if cfg.private:
        if abs(result['epsilon']-cfg.config['privacy']['epsilon'])>1e-7 or result['delta']!=cfg.config['privacy']['delta']:raise ValueError('Completed privacy mismatch')
    elif any(result[k] is not None for k in ('epsilon','delta','clip_fraction')):raise ValueError('No-noise privacy fields must be inapplicable')
    for name,h in result['artifact_sha256'].items():
        if file_hash(run/name)!=h:raise ValueError(f'Completed artifact hash mismatch: {name}')
    copy=dict(result);copy.pop('record_hash')
    if result['record_hash']!=json_hash(copy):raise ValueError('Result record hash mismatch')
    return True

def run_training(cfg,run,identity,gpu_id):
    requested=[False]
    def stop(signum,frame):requested[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    output=run/'checkpoints'/cfg.id;output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.pt';started=time.monotonic();model=None
    seed_all(cfg.seed);device=torch.device('cuda:0');torch.cuda.set_device(device)
    try:
        gpu_record=next((g for g in identity.get('gpus',[]) if g['id']==gpu_id),{})
        gpu_mapping=dict(physical_gpu_id=gpu_id,gpu_uuid=gpu_record.get('uuid'),gpu_name=gpu_record.get('name'),cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),logical_cuda_device='cuda:0',memory_total_mb=gpu_record.get('total_mb'))
        if gpu_record:
            # Check CUDA's actual UUID, not just a possibly reordered numeric index.
            actual_uuid=str(torch.cuda.get_device_properties(device).uuid)
            if actual_uuid.removeprefix('GPU-')!=gpu_record['uuid'].removeprefix('GPU-'):raise RuntimeError(f'CUDA physical mapping mismatch: {actual_uuid} != {gpu_record["uuid"]}')
        if gpu_record:
            free,total=torch.cuda.mem_get_info(device);gpu_mapping.update(memory_free_mb=free//1048576,memory_used_mb=(total-free)//1048576)
        print(gpu_mapping,flush=True)
        if cfg.task=='cv':
            base,metadata=pretrained_vit(local_path(cfg.config['paths']['cv_model']))
            init_hash,head_hash=initialization_digest(base),initialization_digest(base.head)
            training,validation,order=cv_datasets(cfg,metadata)
        else:
            base=make_model(local_path(cfg.config['paths']['nlp_model']))
            init_hash,head_hash=digest(base),digest(base.classifier)
            training,validation,order=nlp_datasets(cfg)
        base=base.to(device)
        model=(ScaledGhostModule if cfg.scaled else StandardGhostModule)(base,cfg.clip) if cfg.private else base
        if not all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters()):raise ValueError('Full FP32 training required')
        optimizer=make_optimizer(model,cfg);noise=None;privacy=dict(private=False);S=None
        if cfg.private:
            matrix=load_matrix(local_path(cfg.config['paths']['matrices']),cfg.workload,cfg.mechanism,cfg.total_steps,cfg.betas[0])
            S=matrix['C'];privacy=calibrate(S,cfg.protocol());p=cfg.config['privacy']
            actual=epsilon_from_mu(cfg.clip*privacy['sensitivity']/privacy['innovation_std_sum'],p['delta'])
            if abs(actual-p['epsilon'])>1e-7:raise ValueError('Privacy calibration mismatch')
            privacy.update(private=True,epsilon=actual,delta=p['delta'],clip_space='scaled' if cfg.scaled else 'original',matrix_sha256=file_hash(matrix['path']))
            noise=OnlineNoise(model.parameters(),matrix,cfg.mechanism,privacy['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
        logical=logical_batch(model,optimizer,cfg,noise)
        loader,eval_loader=make_loaders(cfg,training,validation)
        np.save(output/'train_order.npy',order)
        progress=dict(epoch=0,next_microbatch=0,loss_sum=0.,count=0,clip_count=0,rows=[],epochs=[],
                      augmentation_hashes=[],order_hash=array_hash(order),training_time=0.,wall_time=0.,peak_memory=0,
                      initialization_hash=init_hash,head_hash=head_hash,gpu_mapping=gpu_mapping)
        resume_state=None;epoch_start=None
        if latest.exists():
            resume_state=load_checkpoint(latest,model,optimizer,logical,identity);progress=resume_state['progress']
            if progress['order_hash']!=array_hash(order) or progress['initialization_hash']!=init_hash or progress['head_hash']!=head_hash:raise ValueError('Initialization/order changed')
            if gpu_record and progress.get('gpu_mapping',{}).get('gpu_uuid')!=gpu_record['uuid']:raise ValueError('Checkpoint physical GPU UUID changed')
            loader.generator.set_state(resume_state['sampler']['loader_generator'].cpu())
            restore_rng(resume_state['rng'])
        atomic_json(output/'gpu_mapping.json',gpu_mapping)
        torch.cuda.reset_peak_memory_stats(device)
        def checkpoint():
            progress['wall_time']=wall_before+time.monotonic()-started
            progress['peak_memory']=max(progress['peak_memory'],torch.cuda.max_memory_allocated(device))
            save_checkpoint(latest,model,optimizer,logical,progress,identity,loader.generator.get_state(),epoch_start)
            atomic_json(output/'progress.json',dict(epoch=progress['epoch'],logical_step=logical.optimizer_steps,total_steps=cfg.total_steps))
        wall_before=progress['wall_time']
        while progress['epoch']<cfg.task_config['epochs']:
            model.train();offset=progress['next_microbatch']
            if offset and resume_state:
                epoch_start=resume_state['sampler']['epoch_start_generator'].cpu()
                loader.generator.set_state(epoch_start)
            else:epoch_start=loader.generator.get_state().clone()
            iterator=iter(loader)
            for _ in range(offset):next(iterator)
            if resume_state:
                restore_rng(resume_state['rng'])
                loader.generator.set_state(resume_state['sampler']['loader_generator'].cpu())
                resume_state=None
            loss_part=clips_part=norm_part=0.;norm_max=0.
            for batch in iterator:
                if logical.micro_steps==0:
                    torch.cuda.synchronize();step_start=time.monotonic()
                if cfg.task=='nlp':
                    dropout_seed=cfg.seed+100000+logical.optimizer_steps
                    torch.manual_seed(dropout_seed);torch.cuda.manual_seed(dropout_seed)
                    batch=tuple(t.to(device) for t in batch);inputs,targets=batch[:3],batch[3]
                else:
                    inputs,targets=batch
                    if logical.micro_steps==0:progress['augmentation_hashes'].append(array_hash(inputs[:4].contiguous().numpy()))
                    inputs,targets=inputs.to(device),targets.to(device)
                if len(targets)!=cfg.physical_batch:raise ValueError('Incomplete physical batch')
                logical.begin_microbatch()
                loss,clips=clipped_microbatch(model,inputs,targets,cfg.clip) if cfg.private else unclipped_microbatch(model,inputs,targets)
                loss_part+=loss;clips_part+=clips
                if cfg.private:
                    norm_part+=float(model.last_norms.sum());norm_max=max(norm_max,float(model.last_norms.max()))
                progress['count']+=len(targets);progress['loss_sum']+=loss;progress['clip_count']+=clips;progress['next_microbatch']+=1
                if logical.finish_microbatch():
                    torch.cuda.synchronize();elapsed=time.monotonic()-step_start;progress['training_time']+=elapsed
                    step=logical.optimizer_steps;epsilon=None;size=cfg.task_config['logical_batch_size']
                    if cfg.private:
                        p=cfg.config['privacy'];mu=cfg.clip*fixed_epoch_sensitivity(S,cfg.task_config['epochs'],cfg.spacing,steps=step)/privacy['innovation_std_sum']
                        epsilon=epsilon_from_mu(mu,p['delta'])
                        if epsilon>p['epsilon']+1e-7 or noise.step_count!=step:raise ValueError('Privacy/noise step mismatch')
                    row=dict(step=step,epoch=progress['epoch']+1,train_loss=loss_part/size,clip_fraction=clips_part/size if cfg.private else None,
                             gradient_norm_mean=norm_part/size if cfg.private else None,gradient_norm_max=norm_max if cfg.private else None,
                             gradient_norm=logical.last_gradient_norm,epsilon=epsilon,step_seconds=elapsed,
                             noise_marginal_std_sum=noise.marginal_std(step-1) if noise else None)
                    progress['rows'].append(row)
                    loss_part=clips_part=norm_part=0.;norm_max=0.
                    if step%cfg.config['runtime']['checkpoint_every_steps']==0 or requested[0] or step%cfg.spacing==0:checkpoint()
                    if requested[0]:raise StopRequested('Stopped at a complete logical step; checkpoint preserved')
            del iterator
            expected=(progress['epoch']+1)*cfg.spacing
            if logical.micro_steps or logical.optimizer_steps!=expected:raise ValueError('Incorrect logical step count')
            metrics=evaluate(model,eval_loader,device,cfg.task)
            epoch=dict(epoch=progress['epoch']+1,step=expected,train_loss=progress['loss_sum']/progress['count'],
                       clip_fraction=progress['clip_count']/progress['count'] if cfg.private else None,
                       epsilon=progress['rows'][-1]['epsilon'],**metrics)
            progress['epochs'].append(epoch);print(epoch,flush=True)
            progress.update(epoch=progress['epoch']+1,next_microbatch=0,loss_sum=0.,count=0,clip_count=0)
            checkpoint()
            if requested[0]:raise StopRequested('Stopped after evaluation; checkpoint preserved')
        if cfg.private and abs(progress['epochs'][-1]['epsilon']-cfg.config['privacy']['epsilon'])>1e-7:raise ValueError('Final epsilon mismatch')
        write_csv(output/'metrics.csv',progress['epochs']);write_csv(output/'diagnostics.csv',progress['rows'])
        final=progress['epochs'][-1]
        artifacts={str(p.relative_to(run)):file_hash(p) for p in (latest,output/'train_order.npy',output/'metrics.csv',output/'diagnostics.csv')}
        result=dict(**cfg.values(),status='completed',learning_rate=cfg.lr,clip=cfg.clip,eps_scale=cfg.eps_scale,
                    accuracy=final['accuracy'],loss=final['loss'],epsilon=final['epsilon'],delta=privacy.get('delta'),
                    clip_fraction=final['clip_fraction'],training_time=progress['training_time'],wall_time=progress['wall_time'],
                    peak_memory=progress['peak_memory'],gpu_mapping=gpu_mapping,gpu_uuid=gpu_record.get('uuid'),cuda_visible_devices=gpu_mapping['cuda_visible_devices'],logical_cuda_device='cuda:0',gpu_id=gpu_id,gpu_name=torch.cuda.get_device_name(device),
                    optimizer_steps=logical.optimizer_steps,noise_draws=noise.step_count if noise else 0,epochs=progress['epochs'],
                    initialization_hash=init_hash,head_hash=head_hash,train_order_hash=array_hash(order),
                    augmentation_hashes=progress['augmentation_hashes'],privacy=privacy,identity=identity,
                    config_hash=identity['config_hash'],model_hash=identity['model_hashes'][cfg.task],artifact_sha256=artifacts)
        result['record_hash']=json_hash(result);atomic_json(result_path(run,cfg),result)
        return result
    except StopRequested:
        raise
    except Exception as error:
        failure=dict(**cfg.values(),status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'non_finite' if isinstance(error,FloatingPointError) else 'failed',
                     error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),identity=identity)
        atomic_json(output/f'failure_{time.time_ns()}.json',failure)
        atomic_json(result_path(run,cfg),failure)
        raise
    finally:
        if model is not None and cfg.private:model.remove_hooks()
