"""Eight frozen optimizers, offline formal experiments only."""
import os
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL','system')
__version__='0.1.0'
