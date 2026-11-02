from adam_mf.config import load_config as _load_config

def load_config(path=None):
    c=_load_config(path)
    # Unit tests cover the complete registry even if a user disables some jobs.
    for t in c['tasks'].values():t['enabled']=True
    for m in c['methods'].values():m['enabled']=True
    return c
