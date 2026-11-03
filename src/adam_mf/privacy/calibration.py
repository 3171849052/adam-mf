"""Load and validate computed matrices; solving happens during asset preparation."""
import numpy as np
from scipy.linalg import toeplitz
from adam_mf.audit import file_hash,read_json
from adam_mf.privacy.accountant import fixed_epoch_sensitivity

def workload(name,n,beta1):
    t=np.arange(1,n+1);lag=t[:,None]-t[None,:]
    if name=='identity':return np.eye(n)
    if name=='sgd':return np.tril(np.ones((n,n)))
    if name=='momentum':return np.where(lag>=0,1-beta1**(np.maximum(lag,0)+1),0.)
    if name=='momentum-bias':
        moments=np.where(lag>=0,beta1**np.maximum(lag,0),0.)/(1-beta1**t)[:,None]
        return np.tril(np.cumsum(moments,axis=0))
    raise ValueError(name)
def load_matrix(folder,work,mechanism,n,beta1):
    path=folder/f'{work}_{mechanism}_T{n}.npz'
    meta=read_json(path.with_suffix('.json'))
    if meta['sha256']!=file_hash(path):raise ValueError('Matrix file hash mismatch')
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    A,C,D,d=(arrays[k] for k in ('A','C','D','d'))
    if any(x.shape!=(n,n) for x in (A,C,D)) or not all(np.isfinite(x).all() for x in (A,C,D,d)):raise ValueError('Invalid matrix arrays')
    if not np.allclose(A,workload(work,n,beta1),rtol=1e-13,atol=1e-13):raise ValueError('Frozen workload mismatch; beta1 cannot change with these matrices')
    if not np.allclose(C@D,np.eye(n),atol=1e-7,rtol=0):raise ValueError('Matrix inverse mismatch')
    expected=toeplitz(np.pad(d,(0,n-len(d))),np.r_[d[0],np.zeros(n-1)])
    if not np.allclose(D,expected,atol=1e-12,rtol=1e-12):raise ValueError('Toeplitz mismatch')
    sens=fixed_epoch_sensitivity(C,5,n//5)
    objective=sens*sens*np.sum((A@D)**2)/n
    if not np.isclose(objective,meta['audit']['objective'],rtol=1e-9):raise ValueError('Matrix objective audit mismatch')
    if mechanism=='b2lt':
        P=meta['P'];theta=np.array(meta['theta']);omega=np.array(meta['omega'])
        coeff=np.zeros(n);coeff[0]=1
        for lag in range(1,min(P,n)):coeff[lag]=np.sum(omega*theta**(lag-1))
        if not np.allclose(d,np.pad(coeff[:len(d)],(0,max(0,len(d)-len(coeff))))[:len(d)],atol=1e-12):raise ValueError('B²LT recurrence mismatch')
    return dict(arrays,path=path,metadata=meta)
