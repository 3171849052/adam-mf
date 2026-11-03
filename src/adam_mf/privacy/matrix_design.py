"""Data-independent B²LT design migrated from exp16a/matrices.py.

Only the IID/B²LT designs needed by this experiment are exposed. Training
samples fresh Gaussian innovations; these matrices contain no sampled noise.
"""
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from adam_mf.audit import atomic_json, file_hash, array_hash, read_json
from adam_mf.privacy.calibration import workload, load_matrix
from adam_mf.privacy.accountant import fixed_epoch_sensitivity, target_mu, epsilon_from_mu

DESIGN = "exp16a-b2lt-lbfgs-v1"
PAIRS = (("identity", "iid"), ("sgd", "b2lt"), ("momentum", "b2lt"), ("momentum-bias", "b2lt"))

def recipe():
    return dict(design=DESIGN, source_sha256=file_hash(Path(__file__)),
                calibration_sha256=file_hash(Path(__file__).with_name("calibration.py")),
                accountant_sha256=file_hash(Path(__file__).with_name("accountant.py")),
                beta1=0.9, epochs=5, Q=5, dtype="float64", threads=2,
                torch=torch.__version__, numpy=np.__version__)

def tensor_toeplitz(c,n):
    idx=torch.arange(n)[:,None]-torch.arange(n)[None,:]
    return torch.where(idx>=0,c[idx.clamp_min(0)],0.)

def optimize(A,mechanism,k,work):
    if mechanism != "b2lt":raise ValueError("Only B2LT optimization is supported")
    n=len(A);torch.set_num_threads(2);W=torch.tensor(A,dtype=torch.float64);eye=torch.eye(n,dtype=torch.float64)
    Q=5;P=k;trace=[];candidates=[]
    theta=np.arange(1,Q+1)/(Q+1);x0=np.r_[theta,-(1-theta)/(2*Q)]
    def build(x):
        th,om=x[:Q],x[Q:];sep=torch.pdist(th[:,None])
        margins=torch.cat((-om,th*(1-th),(1+om.sum()).view(1),(1+(om/(1-th)).sum()).view(1),sep.min().view(1)))
        if torch.any(margins<=0):return None
        phi=-margins.log().sum()
        d=torch.cat((x.new_ones(1),(om[:,None]*th[:,None]**torch.arange(P-1)).sum(0),x.new_zeros(n-P)))
        D=tensor_toeplitz(d,n);C=torch.linalg.solve_triangular(D,eye,upper=False)
        if not torch.isfinite(C).all() or float(C.detach().abs().max())>1e10:return None
        indices=torch.arange(k)[:,None]+k*torch.arange(5)[None,:]
        cols=C[:,indices];grams=torch.einsum('njk,njl->jkl',cols,cols)
        sens2=grams.abs().sum((1,2)).max();err=(W@D).square().sum()/n
        return sens2*err,phi,C,D
    x=torch.tensor(x0,dtype=torch.float64);initial=build(x)
    e0=float(initial[0]);lam=1e-4*e0/len(x0)
    # Select among accepted iterates, plus the prescribed initial and DTC IID.
    def retain(v,iteration):
        with torch.no_grad():
            z=build(torch.tensor(v,dtype=torch.float64))
            if z is not None:
                e,phi,_,_=z;record=dict(iteration=iteration,error=float(e),barrier=float(phi),objective=float(e+lam*phi))
                trace.append(record);candidates.append((float(e),np.asarray(v).copy()))
    retain(x0,0)
    def fun(v):
        z=torch.tensor(v,dtype=torch.float64,requires_grad=True);out=build(z)
        if out is None:return float('inf'),np.zeros_like(v)
        e,phi,_,_=out;f=e+lam*phi
        if not torch.isfinite(f):return float('inf'),np.zeros_like(v)
        f.backward();return float(f.detach()),z.grad.detach().numpy().copy()
    # Feasible Armijo L-BFGS: reject barrier violations before accepting any step.
    # Explicit backtracking avoids scipy treating an infinite proposal as convergence.
    current=x0.copy();f,g=fun(current);history=[];message='iteration limit';converged=False
    for iteration in range(1,1001):
        if np.max(np.abs(g))/e0<1e-7:message='gradient tolerance';converged=True;break
        q=g.copy();pairs=[]
        for dx,dg,rho in reversed(history):
            alpha=rho*np.dot(dx,q);pairs.append(alpha);q-=alpha*dg
        scale=np.dot(history[-1][0],history[-1][1])/np.dot(history[-1][1],history[-1][1]) if history else 1/e0
        direction=q*scale
        for (dx,dg,rho),alpha in zip(history,reversed(pairs)):
            direction+=dx*(alpha-rho*np.dot(dg,direction))
        direction=-direction;slope=float(g@direction)
        if slope>=0:direction=-g/e0;slope=float(g@direction);history=[]
        step=1.;accepted=False
        for _ in range(60):
            proposal=current+step*direction;newf,newg=fun(proposal)
            if np.isfinite(newf) and newf<=f+1e-4*step*slope:
                accepted=True;break
            step*=.5
        if not accepted:message='feasible line search stalled';break
        dx=proposal-current;dg=newg-g;curvature=float(dx@dg)
        if curvature>1e-14*np.linalg.norm(dx)*np.linalg.norm(dg):
            history.append((dx,dg,1/curvature));history=history[-20:]
        improvement=f-newf;current=proposal;f,g=newf,newg;retain(current,iteration)
        if improvement<=1e-12*max(1.,abs(f)) and np.max(np.abs(g))/e0<1e-5:
            message='relative objective and gradient tolerance';converged=True;break
    best_error,best=min(candidates,key=lambda z:z[0]);selected='iterate'
    _,_,C,D=build(torch.tensor(best,dtype=torch.float64));C=C.detach().numpy();D=D.detach().numpy()
    iid_error=float(5*np.sum(A*A)/n)
    d=D[:,0].copy()
    meta=dict(optimizer='L-BFGS; memory=20; feasible Armijo backtracking',maxiter=1000,
        iterations=iteration,success=converged,message=message,lambda_barrier=lam,
        initial_parameters=x0.tolist(),selected_parameters=best.tolist(),selected=selected,
        initial_error=e0,selected_error=best_error,iid_error=iid_error,trace=trace,
        bandwidth=P,
        P=P,Q=Q,
        theta=best[:Q].tolist(),omega=best[Q:].tolist())
    if not np.isfinite(best_error) or best_error>e0*(1+1e-9):raise ValueError('Invalid optimization selection')
    return C,D,d,meta

def audit_arrays(A,C,D):
    n=len(A);identity=np.eye(n);k=n//5
    inverse_error=max(np.max(np.abs(C@D-identity)),np.max(np.abs(D@C-identity)))
    if inverse_error>1e-7:raise ValueError('Inverse mismatch')
    B=A@D
    sens=fixed_epoch_sensitivity(C,5,k)
    # Independent columns/direction upper bound, no Gram reuse.
    column_bounds=[float(np.sum(np.abs(C[:,j+k*np.arange(5)]).sum(1)**2)) for j in range(k)]
    absolute_bound=np.sqrt(max(column_bounds))
    if sens>absolute_bound*(1+1e-10):raise ValueError('Sensitivity upper bound failed')
    # Toeplitz and nonnegative cases attain the bound with aligned participation vectors.
    nonnegative=bool(np.min(C)>=-1e-12)
    aligned=max(np.linalg.norm(C[:,j+k*np.arange(5)].sum(1)) for j in range(k))
    if nonnegative and not np.isclose(sens,aligned,rtol=1e-9):raise ValueError('Aligned sensitivity mismatch')
    mu=target_mu(8.,1e-5);sigma=sens/mu;actual=epsilon_from_mu(sens/sigma,1e-5)
    covariance=sigma*sigma*(D@D.T)
    whiten=C@covariance@C.T
    if not np.allclose(whiten,sigma*sigma*identity,atol=1e-6,rtol=1e-6):raise ValueError('Covariance mismatch')
    return dict(inverse_max_abs=inverse_error,factorization_max_abs=float(np.max(np.abs(B@C-A))),
        sensitivity=sens,absolute_column_upper_bound=float(absolute_bound),aligned_lower_bound=float(aligned),nonnegative_strategy=nonnegative,
        mean_workload_error=float(np.sum(B*B)/n),objective=float(sens*sens*np.sum(B*B)/n),
        sigma_sum_clip1=sigma,mu=mu,epsilon=actual,delta=1e-5,
        covariance_sha256=array_hash(covariance),mean_gradient_noise_MSE_clip1=float(np.trace(covariance)/n/1e6),
        workload_noise_MSE_clip1=float(sigma*sigma*np.sum(B*B)/n/1e6))

def verify_generated(folder, work, mechanism, n):
    path=Path(folder)/f'{work}_{mechanism}_T{n}.npz'
    meta=read_json(path.with_suffix('.json'))
    if path.is_symlink() or path.with_suffix('.json').is_symlink():raise ValueError('Matrix symlink refused')
    if meta.get('recipe')!=recipe():raise ValueError('Matrix generation recipe changed; explicitly rebuild matrices for a new run')
    if meta.get('sha256')!=file_hash(path):raise ValueError('Generated matrix hash mismatch')
    if (meta.get('workload'),meta.get('mechanism'),meta.get('total_steps'))!=(work,mechanism,n):raise ValueError('Matrix metadata identity mismatch')
    arrays=load_matrix(Path(folder),work,mechanism,n,.9)
    audit=audit_arrays(arrays['A'],arrays['C'],arrays['D'])
    expected=audit['sigma_sum_clip1']**2*(arrays['D']@arrays['D'].T)
    if not np.allclose(arrays['covariance_clip1'],expected,rtol=1e-12,atol=1e-12):raise ValueError('Saved covariance mismatch')
    if mechanism=='b2lt':
        th=np.asarray(meta['theta']);om=np.asarray(meta['omega'])
        if meta['P']!=n//5 or meta['Q']!=5 or th.shape!=(5,) or om.shape!=(5,):raise ValueError('B2LT dimensions mismatch')
        if not (np.all(om<0) and np.all((th>0)&(th<1)) and len(np.unique(th))==5 and 1+om.sum()>0 and 1+(om/(1-th)).sum()>0):raise ValueError('Infeasible B2LT parameters')
        if audit['objective']>meta['initial_error']*(1+1e-9):raise ValueError('B2LT objective worsened')
    return audit


def prepare_matrices(folder, rebuild=False, horizons=(250,310)):
    """Compute absent matrices; reuse only a validated result of this recipe.

    Caller owns the asset lock. Each matrix is solved in a temporary directory
    and audited before publication. Rebuild is explicit and invalidates resume.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rows=[]
    for n in horizons:
        if n<10 or n%5:raise ValueError('Horizon must be a multiple of five, at least ten')
        for work,mechanism in PAIRS:
            p=folder/f'{work}_{mechanism}_T{n}.npz'
            if (p.exists() or p.with_suffix('.json').exists()) and not rebuild:
                audit=verify_generated(folder,work,mechanism,n)
            else:
                print(f'Solving {work}/{mechanism}/T{n} ...',flush=True)
                started=time.monotonic();A=workload(work,n,.9)
                if mechanism=='iid':C=D=np.eye(n);d=np.ones(1);meta=dict(optimizer='none')
                else:C,D,d,meta=optimize(A,mechanism,n//5,work);d=d[:n//5]
                audit=audit_arrays(A,C,D)
                meta.update(recipe=recipe(),audit=audit,seconds=time.monotonic()-started,
                            workload=work,mechanism=mechanism,total_steps=n)
                with tempfile.TemporaryDirectory(dir=folder,prefix='.design-') as tmp:
                    target=Path(tmp)/p.name
                    np.savez(target,A=A,C=C,D=D,d=d,covariance_clip1=audit['sigma_sum_clip1']**2*(D@D.T))
                    meta['sha256']=file_hash(target);atomic_json(target.with_suffix('.json'),meta)
                    verify_generated(Path(tmp),work,mechanism,n)
                    os.replace(target,p);os.replace(target.with_suffix('.json'),p.with_suffix('.json'))
            rows.append(dict(workload=work,mechanism=mechanism,total_steps=n,**audit))
            print(f'Verified {work}/{mechanism}/T{n}: objective={audit["objective"]:.10g}',flush=True)
    atomic_json(folder/'generation_audit.json',dict(recipe=recipe(),matrices=rows))
    return rows


def verify_all(config):
    from adam_mf.paths import local_path
    folder=local_path(config['paths']['matrices'])
    for n in (250,310):
        for work,mechanism in PAIRS:verify_generated(folder,work,mechanism,n)
