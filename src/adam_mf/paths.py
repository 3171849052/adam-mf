from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def local_path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT): raise ValueError(f'Path must stay in repository: {value}')
    return p
