import math
import numpy as np
from scipy.stats import t

def statistics(values):
    a=np.asarray(list(values),dtype=float);n=len(a)
    if not n or not np.isfinite(a).all():raise ValueError('Statistics require finite observations')
    mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None
    radius=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-radius if radius is not None else None,ci95_high=mean+radius if radius is not None else None)
