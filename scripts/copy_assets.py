#!/usr/bin/env python3
"""One-time import; copy bytes, dereference source links, never overwrite conflicts."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT.parent/'dp-adam-correlated');args=parser.parse_args();src=args.source.resolve()
    groups={'data':[], 'model':[], 'assets/matrices':[]}
    for name in ('train','test','meta'): groups['data'].append((src/'data/cifar-100-python'/name,ROOT/'data/cifar-100-python'/name))
    for name in ('train.parquet','validation.parquet'):groups['data'].append((src/'data/sst2'/name,ROOT/'data/sst2'/name))
    for name in ('cv_split.npz','nlp_split.npz','nlp_tokens_128.pt','nlp_official_tokens_128.pt'):
        groups['data'].append((src/'exp16e/runtime/assets'/name,ROOT/'data/preprocessed'/name))
    for p in sorted((src/'cache/bert-tiny').iterdir()):
        if p.is_file():groups['model'].append((p,ROOT/'model/bert-tiny'/p.name))
    hub=src/'cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k'
    revision=(hub/'refs/main').read_text().strip()
    groups['model'].append((hub/'snapshots'/revision/'model.safetensors',ROOT/'model/vit-tiny/model.safetensors'))
    acceptance=json.loads((src/'exp16a/results/phase1_acceptance.json').read_text())
    if acceptance['status']!='passed':raise ValueError('Source matrix acceptance failed')
    for n in (250,310):
        for work,mech in [('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')]:
            for suffix in ('.npz','.json'):
                p=src/'exp16a/results/matrices'/f'{work}_{mech}_T{n}{suffix}'
                if sha(p)!=acceptance['matrix_sha256'][str(p.relative_to(src))]:raise ValueError(f'Source matrix hash: {p}')
                groups['assets/matrices'].append((p,ROOT/'assets/matrices'/p.name))
    # Validate the entire plan before copying anything.
    for files in groups.values():
        for source,target in files:
            if not source.is_file():raise FileNotFoundError(source)
            if target.is_symlink() or (target.exists() and sha(source)!=sha(target)):raise FileExistsError(f'Different destination: {target}')
    for folder,files in groups.items():
        rows=[]
        for source,target in files:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target,follow_symlinks=True)
            digest=sha(target)
            if digest!=sha(source):raise ValueError(f'Copy hash mismatch: {target}')
            rows.append(dict(path=str(target.relative_to(ROOT)),source=str(source.relative_to(src)),bytes=target.stat().st_size,sha256=digest))
        output=ROOT/folder/'manifest.json'
        content=json.dumps(dict(files=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2)+'\n'
        if output.exists() and output.read_text()!=content:raise FileExistsError(output)
        output.write_text(content)
        print(folder,len(rows),'files',sum(r['bytes'] for r in rows),'bytes')
if __name__=='__main__':main()
