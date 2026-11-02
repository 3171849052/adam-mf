#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adam_mf.audit import verify_assets
print(f'Verified {len(verify_assets())} asset/manifest hashes.')
