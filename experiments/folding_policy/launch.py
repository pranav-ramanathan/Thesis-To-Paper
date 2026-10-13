"""Freeze a bounded, validation-only policy pilot and launch one array cell."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
CONDITIONS = ('rl_scratch', 'patternboost', 'cp_bc', 'patternboost_rl', 'cp_rl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def software():
    versions = {}
    for package in ('torch', 'numpy', 'ortools'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'packages': versions}


def validate_data(data):
    """Check split separation before any model is allowed to see this data."""
    keys = ('train', 'validation', 'test', 'benchmark')
    seen = {}
    for key in keys:
        if not data.get(key):
            raise ValueError(f'Missing or empty {key} split')
        for row in data[key]:
            sequence = row['sequence']
            if len(sequence) < 3 or set(sequence) - {'H', 'P'}:
                raise ValueError(f'Invalid sequence in {key}')
            identity = min(sequence, sequence[::-1])
            if identity in seen:
                raise ValueError(f'Sequence or reverse shared by {seen[identity]} and {key}')
            seen[identity] = key
    if len(data['validation']) < 16:
        raise ValueError('Pilot requires the frozen 16-sequence validation partition')
    chosen = [data['validation'][i]['id'] for i in (0, 5, 10, 15)]
    if len({len(data['validation'][i]['sequence']) for i in (0, 5, 10, 15)}) != 4:
        raise ValueError('Pilot requires four different validation lengths')
    return chosen


def prepare(args):
    run = Path(args.directory).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    required = ('dataset.py', 'protocol.json', 'train.py', 'model.py', 'geometry.py',
                'evaluate.py', 'report.py', 'launch.py', 'pilot.sbatch')
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete implementation: {missing}')
    config = json.loads((HERE / 'protocol.json').read_text())
    fixed = {'stage': 'validation-only-engineering-pilot', 'seeds': [0, 1],
             'train_seconds_per_condition_seed': 1200, 'inference_seconds_per_sequence_seed': 60,
             'threads': 4, 'memory_gib': 16, 'array_concurrency': 10, 'slurm_minutes': 45,
             'final_test_enabled': False, 'automatic_extension': False}
    if any(config.get(key) != value for key, value in fixed.items()) or config.get('conditions') != list(CONDITIONS):
        raise ValueError('Protocol differs from the bounded validation pilot settings')
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python executable missing: {python}')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise ValueError('sbatch is unavailable; submit on Apocrita or use --dry-run')
    if not args.dry_run and any(software()['packages'][name] is None for name in ('torch', 'ortools')):
        raise ValueError('The cluster virtual environment must contain Torch and OR-Tools before submission')
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    (run / 'tasks').mkdir()
    bundle = run / 'bundle'
    policy = bundle / 'folding_policy'
    policy.mkdir(parents=True)
    for source in HERE.iterdir():
        if source.is_file() and source.suffix in {'.py', '.json', '.sh', '.sbatch', '.md', '.txt'}:
            shutil.copy2(source, policy / source.name)
    cluster = bundle / 'cluster'
    cluster.mkdir()
    for name in ('hp_solver.py', 'geometry.py', 'protocol.json'):
        shutil.copy2(HERE.parent / 'cluster' / name, cluster / name)
    subprocess.run([str(python), str(policy / 'dataset.py'), '--output', str(run / 'data.json')], check=True)
    data = json.loads((run / 'data.json').read_text())
    validation_ids = validate_data(data)
    tasks = []
    for condition in CONDITIONS:
        for seed in (0, 1):
            tasks.append({'task': len(tasks), 'condition': condition, 'seed': seed,
                          'controls': ['greedy', 'cp_sat'] if condition == CONDITIONS[0] else []})
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(HERE), 'status', '--porcelain'], capture_output=True, text=True)
    files = sorted(p for p in bundle.rglob('*') if p.is_file()) + [run / 'data.json']
    manifest = {
        'schema': 1, 'kind': 'validation_engineering_pilot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': git.stdout.strip() if git.returncode == 0 else None,
        'source_had_uncommitted_files': bool(dirty.stdout.strip()),
        'python_executable': str(python), 'software_at_freeze': software(),
        'hashes': {str(p.relative_to(run)): sha(p) for p in files},
        'threads': 4, 'memory_gib': 16, 'concurrency': 10, 'slurm_seconds': 2700,
        'training_seconds': 1200, 'evaluation_seconds_per_sequence_per_seed': 60,
        'evaluation_seeds': 2, 'total_evaluation_seconds_per_sequence_per_method': 120,
        'validation_ids': validation_ids, 'tasks': tasks,
        'final_test_access': False, 'no_automatic_followup': True,
        'reserved_cpu_hours_ceiling': len(tasks) * 4 * 2700 / 3600,
    }
    write_json(run / 'pilot.json', manifest)
    (run / 'pilot.sha256').write_text(sha(run / 'pilot.json') + '\n')
    command = ['sbatch', '--parsable', f'--chdir={run}',
               f'--output={run}/logs/%A_%a.out', f'--error={run}/logs/%A_%a.err']
    if os.environ.get('HP_MAIL_USER'):
        command.append('--mail-user=' + os.environ['HP_MAIL_USER'])
    command += [str(policy / 'pilot.sbatch'), str(run)]
    write_json(run / 'submission.json', {'command': command, 'submitted': False})
    print(json.dumps({'directory': str(run), 'tasks': len(tasks), 'validation_ids': validation_ids,
                      'training_minutes_per_task': 20, 'max_task_minutes_including_controls': 32,
                      'reserved_cpu_hours_ceiling': manifest['reserved_cpu_hours_ceiling']}))
    if args.dry_run:
        print('Prepared only; submission command: ' + shlex.join(command))
    else:
        job = subprocess.check_output(command, text=True).strip()
        write_json(run / 'submission.json', {'command': command, 'submitted': True, 'job_id': job})
        (run / 'job_id.txt').write_text(job + '\n')
        print(f'Submitted {job}. Validation pilot only; no campaign follows automatically.')


def verify(run):
    checksum = run / 'pilot.sha256'
    if checksum.exists() and checksum.read_text().strip() != sha(run / 'pilot.json'):
        raise ValueError('Frozen pilot manifest changed')
    manifest = json.loads((run / 'pilot.json').read_text())
    if manifest.get('final_test_access') is not False:
        raise ValueError('Pilot may only access validation sequences')
    for relative, expected in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or sha(path) != expected:
            raise ValueError(f'Frozen artifact changed: {relative}')
    data = json.loads((run / 'data.json').read_text())
    from dataset import validate_dataset
    validate_dataset(data)
    if validate_data(data) != manifest['validation_ids']:
        raise ValueError('Validation selection changed')
    return manifest


def execute(args):
    run = Path(args.directory).resolve()
    manifest = verify(run)
    if not 0 <= args.task < len(manifest['tasks']):
        raise ValueError('Task index is outside the frozen pilot array')
    task = manifest['tasks'][args.task]
    folder = run / 'tasks' / f"{task['task']:02d}_{task['condition']}_seed{task['seed']}"
    folder.mkdir(exist_ok=False)
    start = time.monotonic()
    cpu_model = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    record = {'task': task, 'hostname': socket.gethostname(), 'cpu_model': cpu_model,
              'platform': platform.platform(), 'software': software(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'phases': []}
    write_json(folder / 'launch.json', record)
    policy = run / 'bundle' / 'folding_policy'
    base = [sys.executable]
    try:
        if record['software'] != manifest['software_at_freeze']:
            raise ValueError('Python or package versions changed since the pilot was frozen')
        if int(os.environ.get('SLURM_CPUS_PER_TASK', manifest['threads'])) != manifest['threads']:
            raise ValueError('CPU allocation does not match frozen pilot')
        phases = [('training', base + [str(policy / 'train.py'), '--data', str(run / 'data.json'),
                   '--condition', task['condition'], '--seed', str(task['seed']),
                   '--seconds', str(manifest['training_seconds']), '--threads', str(manifest['threads']),
                   '--output', str(folder / 'training'), '--config', str(policy / 'protocol.json')],
                   manifest['training_seconds'] + 90)]
        common = ['--data', str(run / 'data.json'), '--split', 'validation',
                  '--ids', *manifest['validation_ids'], '--seed', str(task['seed']),
                  '--seconds', str(manifest['evaluation_seconds_per_sequence_per_seed']),
                  '--threads', str(manifest['threads'])]
        seconds = len(manifest['validation_ids']) * manifest['evaluation_seconds_per_sequence_per_seed'] + 60
        phases.append(('model_evaluation', base + [str(policy / 'evaluate.py'), '--method', 'model',
                      '--checkpoint', str(folder / 'training' / 'checkpoint.pt'),
                      '--output', str(folder / 'model_evaluation.json'), *common], seconds))
        for control in task['controls']:
            phases.append((control + '_evaluation', base + [str(policy / 'evaluate.py'), '--method', control,
                           '--output', str(folder / (control + '_evaluation.json')), *common], seconds))
        for name, command, timeout in phases:
            phase_start = time.monotonic()
            remaining = manifest['slurm_seconds'] - (phase_start - start) - 15
            if remaining <= 0:
                raise TimeoutError('Pilot task deadline')
            try:
                with (folder / (name + '.log')).open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                record['phases'].append({'name': name, 'seconds': time.monotonic() - phase_start,
                                         'exit_code': None, 'timed_out': True})
                write_json(folder / 'launch.json', record)
                raise
            record['phases'].append({'name': name, 'seconds': time.monotonic() - phase_start,
                                     'exit_code': process.returncode})
            write_json(folder / 'launch.json', record)
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}; inspect its log')
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - start
        write_json(folder / 'launch.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('directory')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--python', default=sys.executable)
    p = sub.add_parser('run')
    p.add_argument('directory')
    p.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else execute(args)


if __name__ == '__main__':
    main()
