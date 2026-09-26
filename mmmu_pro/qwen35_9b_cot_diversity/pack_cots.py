#!/usr/bin/env python
"""Pack a raw outputs/*.jsonl trace file into cots/<stem>.shardNN.jsonl.gz pieces of ~40 MB
compressed (GitHub's per-file limit is 100 MB), optionally keeping only some top_p values.
  python pack_cots.py outputs/t12_sweep.jsonl t12_sweep --top-ps 0.5,0.6,0.7,0.75,0.8,0.85"""
import argparse, gzip, json, os
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("stem")
ap.add_argument("--top-ps", default="", help="keep only these top_p (default: all)")
ap.add_argument("--max-mb", type=float, default=40)
a = ap.parse_args()
keep = {float(x) for x in a.top_ps.split(",")} if a.top_ps else None
os.makedirs("cots", exist_ok=True)
n = i = 0; gz = None
def new():
    global gz, i
    if gz: gz.close()
    gz = gzip.open(f"cots/{a.stem}.shard{i:02d}.jsonl.gz", "wb", compresslevel=6); i += 1
new()
for line in open(a.src):
    try: r = json.loads(line)
    except Exception: continue            # a torn line from a kill mid-write
    if keep is not None and r["top_p"] not in keep: continue
    gz.write((json.dumps(r) + "\n").encode()); n += 1
    if n % 500 == 0 and gz.fileobj.tell() > a.max_mb * 1e6:
        new()
gz.close()
print(f"[pack] {a.src} -> cots/{a.stem}.shard00..{i-1:02d}.jsonl.gz  ({n} rows)")
