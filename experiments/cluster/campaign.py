"""Freeze a portable campaign and its absolute deadline; no numerical imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify(campaign):
    expected=(campaign/'campaign.sha256').read_text().strip()
    if file_hash(campaign/'campaign.json') != expected:
        raise RuntimeError('Campaign manifest changed; deadline/settings are frozen')
    manifest = json.loads((campaign / 'campaign.json').read_text())
    if hashes(campaign / 'bundle') != manifest['bundle_sha256']:
        raise RuntimeError('Frozen bundle changed; use a new campaign directory')
    return manifest


def create(out, *, mode, run_hours=24, concurrency=24, include_decisionboost=False,
           model_dir=None, threads=8):
    if not 0 < run_hours <= 239.75:
        raise ValueError('run-hours must be > 0 and <= 239.75 (save/termination reserve)')
    if not 1 <= threads <= 96 or not 1 <= concurrency <= 125:
        raise ValueError('Invalid CPU count or concurrency')
    if mode == 'pilot' and threads != 8:
        raise ValueError('The pilot screens 1, 2, 4, 8 threads; allocate eight CPUs')
    if include_decisionboost and not model_dir:
        raise ValueError('DecisionBoost requires --model-dir from prepare-model.sh')
    model_files=None
    if model_dir:
        model_dir = str(Path(model_dir).resolve())
        for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
            if not (Path(model_dir) / name).is_file():
                raise ValueError(f'Missing encoder file: {name}')
        pin = json.loads((Path(model_dir) / 'cluster_pin.json').read_text())
        if pin['revision'] != '45bb4654a4d5aaff24dd11d4781fa46d39bf8c13':
            raise ValueError('Wrong encoder revision')
        for name, digest in pin['files'].items():
            if file_hash(Path(model_dir) / name) != digest:
                raise ValueError(f'Encoder file changed: {name}')
        model_files=pin['files']
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(HERE, out / 'bundle', ignore=shutil.ignore_patterns('__pycache__', '.testdeps'))
    (out / 'logs').mkdir()
    (out / 'tasks').mkdir()
    protocol = json.loads((out / 'bundle/protocol.json').read_text())
    protocol['threads'] = threads
    save(out / 'bundle/protocol.json', protocol)
    if mode == 'pilot':
        tasks = [dict(arm='pilot', seed=0, seq_id='representatives')]
        deadline_seconds, run_seconds = 864000, 3000
    else:
        # Interleave methods/seeds/lengths so a partial campaign has broad coverage.
        tasks = [dict(arm=arm, seed=seed, seq_id=seq_id)
                 for seed in protocol['seeds']
                 for seq_id in protocol['configurations']
                 for arm in ('cp_sat', 'rl', 'rl_cp_sat')]
        if include_decisionboost:
            for seed in reversed(protocol['seeds']):
                tasks.insert(seed * (len(tasks) // 5), dict(arm='decisionboost', seed=seed, seq_id='fresh_corpus'))
        deadline_seconds, run_seconds = 864000, run_hours * 3600
    now = time.time()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = dict(schema_version=1, mode=mode, created_epoch=now,
                    deadline_epoch=now + deadline_seconds,
                    deadline_utc=datetime.fromtimestamp(now + deadline_seconds, timezone.utc).isoformat(),
                    deadline_scope='Entire campaign from creation, includes queue time; no extension',
                    run_seconds=run_seconds, concurrency=concurrency, threads=threads,
                    tasks=tasks, model_dir=model_dir, model_files=model_files, git_revision=revision,
                    python_executable=sys.executable,
                    bundle_sha256=hashes(out / 'bundle'),
                    maximum_concurrent_allocated_cpus=threads * min(concurrency, len(tasks)),
                    reserved_cpu_hours_ceiling=threads * min(concurrency, len(tasks)) * deadline_seconds / 3600,
                    comparison='Primary common CPU domain and per-run elapsed budget; DecisionBoost is a separate call-budget study')
    save(out / 'campaign.json', manifest)
    (out/'campaign.sha256').write_text(file_hash(out/'campaign.json')+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['pilot', 'campaign'])
    p.add_argument('out', type=Path)
    p.add_argument('--run-hours', type=float, default=24)
    p.add_argument('--concurrency', type=int, default=24)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--include-decisionboost', action='store_true')
    p.add_argument('--model-dir', type=Path)
    args = p.parse_args()
    m = create(**vars(args))
    print(json.dumps(dict(directory=str(args.out.resolve()), tasks=len(m['tasks']),
                          deadline_utc=m['deadline_utc'], run_hours=m['run_seconds']/3600,
                          reserved_cpu_hours_ceiling=m['reserved_cpu_hours_ceiling']), indent=2))
