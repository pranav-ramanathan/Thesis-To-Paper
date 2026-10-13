"""Bounded standalone search from frozen policy endpoints; never train or repair."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from allocation_evaluate import validate_allocation_checkpoint
from dataset import validate_dataset
from evaluate import evaluate_sequence
from inference_launch import task_directory, verify
from launch import sha, software, write_json


def find_task(manifest, task_id):
    matches = [row for row in manifest['tasks'] if row['task'] == task_id]
    if len(matches) != 1:
        raise ValueError('Task is outside the frozen inference comparison')
    return matches[0]


def checkpoint_path(run, task):
    if task['method'] != 'model':
        if task.get('checkpoint') is not None:
            raise ValueError('A standalone baseline must not load a policy checkpoint')
        return None
    path = (run / task['checkpoint']).resolve()
    if not path.is_relative_to(run) or not path.is_file() or sha(path) != task['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint file changed or escaped the run directory')
    return path


def validate_runtime(manifest):
    if software() != manifest['software_at_freeze']:
        raise ValueError('Python or package versions changed since inference was frozen')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', manifest['threads'])) != manifest['threads']:
        raise ValueError('CPU allocation does not match frozen inference comparison')


def validate_endpoint(saved, data_hash, task, threads):
    """A fixed two-hour endpoint, including the declared branch and seed."""
    validate_allocation_checkpoint(saved, data_hash, task['seed'])
    expected = {'patternboost': ('patternboost', 'patternboost', 'patternboost'),
                'rl_scratch': ('rl_scratch', 'rl_scratch', 'rl'),
                'balanced': ('rl_branch', 'patternboost_rl', 'rl'),
                'more_pretrain': ('rl_branch', 'patternboost_rl', 'rl')}
    if task['arm'] not in expected:
        raise ValueError('Unknown trained policy arm')
    if tuple(saved.get(key) for key in ('mode', 'condition', 'stage')) != expected[task['arm']]:
        raise ValueError('Checkpoint is not the declared arm and training stage')
    if saved.get('training_seconds') != task['training_seconds'] or task['training_seconds'] != 7200:
        raise ValueError('Expected the fixed two-hour training endpoint')
    if saved.get('threads') != threads:
        raise ValueError('Checkpoint training resources differ from the frozen comparison')
    if saved.get('state_sha256') != task['state_sha256']:
        raise ValueError('Checkpoint weights differ from the frozen endpoint')
    return {key: saved.get(key) for key in
            ('mode', 'stage', 'condition', 'seed', 'training_seconds', 'dataset_sha256',
             'state_sha256', 'kind', 'schema', 'threads')}


def evaluate_one(run, task, manifest, sequence_id, output):
    """Run one sequence in its own process, with imports and model load timed."""
    run, output = Path(run).resolve(), Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing sequence result')
    validate_runtime(manifest)
    if sequence_id not in manifest['validation_ids']:
        raise ValueError('Requested sequence is outside the frozen validation partition')
    data = json.loads((run / 'data.json').read_text())
    validate_dataset(data)
    rows = {row['id']: row for row in data['validation']}
    if sequence_id not in rows:
        raise ValueError('Requested sequence is not a validation sequence')
    if sha(run / 'data.json') != manifest['hashes']['data.json']:
        raise ValueError('Frozen evaluation dataset changed')
    checkpoint = checkpoint_path(run, task)
    if task['method'] not in ('model', 'cp_sat', 'greedy'):
        raise ValueError('Unknown standalone search method')
    if not math.isfinite(task['seconds']) or task['seconds'] <= 0:
        raise ValueError('Positive finite search time is required')
    metadata = None

    def check(saved, data_hash):
        nonlocal metadata
        metadata = validate_endpoint(saved, data_hash, task, manifest['threads'])

    result = evaluate_sequence(rows[sequence_id], task['method'], task['seed'],
                               task['seconds'], manifest['threads'], checkpoint,
                               data_hash=manifest['hashes']['data.json'],
                               checkpoint_validator=check)
    if checkpoint is not None:
        if metadata is None:
            raise ValueError('Model checkpoint was not validated before inference')
        if sha(checkpoint) != task['checkpoint_sha256']:
            raise ValueError('Checkpoint file changed during evaluation')
    payload = {'schema': 1, 'arm': task['arm'], 'checkpoint_metadata': metadata,
               'checkpoint_sha256': task.get('checkpoint_sha256'), 'result': result}
    write_json(output, payload)
    return payload


def execute(args):
    run = Path(args.directory).resolve()
    manifest = verify(run)
    task = find_task(manifest, args.task)
    folder = task_directory(run, task)
    folder.mkdir(exist_ok=False)
    (folder / 'sequences').mkdir()
    start = time.monotonic()
    cpu_model = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_model = next((line.split(':', 1)[1].strip()
                          for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    record = {'task': task, 'hostname': socket.gethostname(), 'cpu_model': cpu_model,
              'platform': platform.platform(), 'software': software(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'phases': []}
    payload = {'schema': 1, 'kind': 'inference_comparison_evaluation', 'status': 'running',
               'split': 'validation', 'final_test_access': False,
               'arm': task['arm'], 'method': task['method'], 'seed': task['seed'],
               'threads': manifest['threads'], 'ids': manifest['validation_ids'],
               'seconds_per_sequence': task['seconds'],
               'checkpoint_sha256': task.get('checkpoint_sha256'),
               'checkpoint_metadata': None, 'dataset_file_sha256': manifest['hashes']['data.json'],
               'model_setup_charged_per_sequence': True,
               'generated_count_note': 'CP-SAT incumbents are not a count of generated folds.',
               'results': []}
    write_json(folder / 'launch.json', record)
    write_json(folder / 'evaluation.json', payload)
    try:
        validate_runtime(manifest)
        checkpoint = checkpoint_path(run, task)
        for sequence_id in manifest['validation_ids']:
            phase_start = time.monotonic()
            remaining = task['slurm_seconds'] - (phase_start - start) - 15
            if remaining <= 0:
                raise TimeoutError('Inference task deadline')
            output = folder / 'sequences' / (sequence_id + '.json')
            command = [sys.executable, str(run / 'bundle' / 'folding_policy' / 'inference_worker.py'),
                       'sequence', str(run), '--task', str(task['task']),
                       '--id', sequence_id, '--output', str(output)]
            phase = {'name': sequence_id, 'exit_code': None}
            record['phases'].append(phase)
            try:
                with (folder / 'sequences' / (sequence_id + '.log')).open('w') as log:
                    child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=min(task['seconds'] + 60, remaining))
                phase['exit_code'] = child.returncode
            except subprocess.TimeoutExpired:
                phase['timed_out'] = True
                raise
            finally:
                phase['seconds'] = time.monotonic() - phase_start
                write_json(folder / 'launch.json', record)
            if child.returncode:
                raise RuntimeError(f'{sequence_id} exited {child.returncode}; inspect its sequence log')
            item = json.loads(output.read_text())
            row = item['result']
            if (item.get('arm') != task['arm'] or item.get('checkpoint_sha256') != task.get('checkpoint_sha256')
                    or row.get('id') != sequence_id or row.get('method') != task['method']
                    or row.get('seed') != task['seed'] or row.get('budget_s') != task['seconds']):
                raise ValueError('Sequence result differs from its frozen task')
            metadata = item.get('checkpoint_metadata')
            if payload['checkpoint_metadata'] is not None and payload['checkpoint_metadata'] != metadata:
                raise ValueError('Checkpoint metadata changed between sequence evaluations')
            payload['checkpoint_metadata'] = metadata
            payload['results'].append(row)
            write_json(folder / 'evaluation.json', payload)
        if checkpoint is not None and sha(checkpoint) != task['checkpoint_sha256']:
            raise ValueError('Checkpoint file changed during task execution')
        record['status'] = payload['status'] = 'complete'
    except Exception as exc:
        record['status'] = payload['status'] = 'failed'
        record['error'] = payload['error'] = str(exc)
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - start
        write_json(folder / 'evaluation.json', payload)
        write_json(folder / 'launch.json', record)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'sequence'):
        child = sub.add_parser(name)
        child.add_argument('directory', type=Path)
        child.add_argument('--task', type=int, required=True)
        if name == 'sequence':
            child.add_argument('--id', required=True)
            child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        execute(args)
    else:
        run = args.directory.resolve()
        manifest = verify(run)
        task = find_task(manifest, args.task)
        expected_output = task_directory(run, task) / 'sequences' / (args.id + '.json')
        if args.output.resolve() != expected_output.resolve():
            parser.error('Sequence output must belong to its frozen task')
        evaluate_one(run, task, manifest, args.id, args.output)


if __name__ == '__main__':
    main()
