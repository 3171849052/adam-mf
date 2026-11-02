# Frozen matrix bundle v1

`matrices-v1.tar.gz` contains exactly the 16 accepted NPZ/JSON files loaded by exp16e from exp16a: identity/IID and SGD, Momentum, Momentum-Bias B²LT at T=250 and T=310. These are copied bytes, not newly optimized matrices. The archive is small enough for ordinary Git distribution; do not apply Git LFS filters.

The authoritative archive hash, per-file hashes, and sizes are in `config/assets.lock.json`. `download_assets.sh` verifies and extracts only the listed regular members. See `docs/assets.md` for provenance and protocol.
