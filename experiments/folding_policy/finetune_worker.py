"""Train bounded branches from frozen parents, then evaluate validation only."""
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
from finetune_launch import task_directory, verify
from launch import sha, software, write_json


def find_task(manifest, task_id):
    matches = [row for row in manifest['tasks'] if row['task'] == task_id]
    if len(matches) != 1:
        raise ValueError('Task is outside the frozen fine-tuning pilot')
    return matches[0]


def validate_runtime(manifest):
    if software() != manifest['software_at_freeze']:
        raise ValueError('Python or package versions changed since the pilot was frozen')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', manifest['threads'])) != manifest['threads']:
        raise ValueError('CPU allocation differs from the frozen pilot')


def parent_path(run, task):
    path = (run / task['parent']).resolve()
    if not path.is_relative_to(run) or not path.is_file() or sha(path) != task['parent_sha256']:
        raise ValueError('Frozen parent changed or escaped the run directory')
    return path


def training_receipt(run, task, manifest):
    """The final checkpoint must be the complete, declared training endpoint."""
    folder = task_directory(run, task) / 'training'
    trained = json.loads((folder / 'result.json').read_text())
    expected = {'status': 'complete', 'arm': task['arm'], 'seed': task['seed'],
                'threads': manifest['threads'], 'budget_s': task['training_seconds'],
                'base_training_seconds': manifest['parent_training_seconds'],
                'additional_training_seconds': task['training_seconds'],
                'training_seconds': task['lineage_training_seconds'],
                'dataset_sha256': manifest['hashes']['data.json'],
                'parent_file_sha256': task['parent_sha256'],
                'parent_state_sha256': task['parent_state_sha256'],
                'initial_state_sha256': task['parent_state_sha256']}
    if any(trained.get(key) != value for key, value in expected.items()):
        raise ValueError('Training receipt does not match the frozen branch or full budget')
    if (trained.get('final_test_enabled') is not False or trained.get('test_evaluated') is not False
            or trained.get('selection', {}).get('selected') is not False
            or trained.get('training_settings', {}).get('learning_rate') != task['learning_rate']
            or trained.get('reference_state_sha256') != task['parent_state_sha256']):
        raise ValueError('Training settings, test restrictions or frozen reference changed')
    handoff = trained.get('handoff', {})
    if (handoff.get('pretraining_sha256') != task['parent_state_sha256']
            or handoff.get('finetune_initial_sha256') != task['parent_state_sha256']
            or handoff.get('pretraining_optimizer_sha256') != task['parent_optimizer_sha256']
            or handoff.get('learning_rate_override') != task['learning_rate']
            or any(handoff.get(key) is not True for key in (
                'optimizer_moments_retained', 'rng_state_retained', 'elite_pool_retained'))):
        raise ValueError('Training did not retain its declared parent state at handoff')
    update_key = {'patternboost': 'supervised_updates', 'grpo': 'rl_updates', 'dr_grpo': 'rl_updates',
                  'reinforce': 'rl_updates', 'dpo': 'preference_updates'}[task['arm']]
    updates = trained.get('segment_counters', {}).get(update_key)
    if type(updates) is not int or updates < 1:
        raise ValueError('Complete training requires an update for its declared method')
    elapsed = trained.get('compute_elapsed_s')
    if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < task['training_seconds']:
        raise ValueError('Training receipt does not establish elapsed-budget coverage')
    records = [row for row in trained.get('checkpoint_records', []) if row.get('filename') == 'final.pt']
    if len(records) != 1:
        raise ValueError('Training receipt must identify one fixed final checkpoint')
    record = records[0]
    if (record.get('training_seconds') != task['lineage_training_seconds']
            or record.get('selected') is not False
            or record.get('state_sha256') != trained.get('final_state_sha256')
            or not record.get('state_sha256') or record.get('file_sha256') != sha(folder / 'final.pt')):
        raise ValueError('Final checkpoint differs from its complete training receipt')
    return trained, record


def endpoint(run, task, manifest):
    parent = parent_path(run, task)
    if task['arm'] == 'parent':
        return parent, task['parent_sha256'], task['parent_state_sha256']
    _, record = training_receipt(run, task, manifest)
    return task_directory(run, task) / 'training' / 'final.pt', record['file_sha256'], record['state_sha256']


def validate_endpoint(saved, data_hash, task, manifest, state_sha):
    if task['arm'] == 'parent':
        validate_allocation_checkpoint(saved, data_hash, task['seed'])
        if tuple(saved.get(key) for key in ('mode', 'condition', 'stage')) != (
                'patternboost', 'patternboost', 'patternboost'):
            raise ValueError('Parent must be the saved PatternBoost endpoint')
    else:
        from finetune_train import validate_finetune_checkpoint
        validate_finetune_checkpoint(saved, data_hash, task['seed'], task['arm'], task['parent_sha256'])
        if (saved.get('base_training_seconds') != manifest['parent_training_seconds']
                or saved.get('additional_training_seconds') != task['training_seconds']
                or saved.get('parent_state_sha256') != task['parent_state_sha256']
                or saved.get('effective_training_settings', {}).get('learning_rate') != task['learning_rate']):
            raise ValueError('Fine-tuned checkpoint lineage does not match its task')
    if (saved.get('training_seconds') != task['lineage_training_seconds']
            or saved.get('threads') != manifest['threads'] or saved.get('state_sha256') != state_sha):
        raise ValueError('Checkpoint state, training budget or resources changed')
    return {key: saved.get(key) for key in (
        'kind', 'schema', 'arm', 'mode', 'stage', 'condition', 'seed', 'training_seconds',
        'base_training_seconds', 'additional_training_seconds', 'dataset_sha256',
        'state_sha256', 'threads', 'parent_file_sha256', 'parent_state_sha256',
        'segment_counters', 'diagnostics')}


def evaluate_one(run, task, manifest, sequence_id, output):
    run, output = Path(run).resolve(), Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite a sequence result')
    validate_runtime(manifest)
    if sequence_id not in manifest['validation_ids']:
        raise ValueError('Sequence is outside the frozen validation partition')
    data = json.loads((run / 'data.json').read_text())
    validate_dataset(data)
    validation = {row['id']: row for row in data['validation']}
    if sequence_id not in validation or sha(run / 'data.json') != manifest['hashes']['data.json']:
        raise ValueError('Frozen validation dataset changed')
    checkpoint, digest, state = endpoint(run, task, manifest)
    metadata = None

    def check(saved, data_hash):
        nonlocal metadata
        metadata = validate_endpoint(saved, data_hash, task, manifest, state)

    result = evaluate_sequence(validation[sequence_id], 'model', task['seed'],
                               task['evaluation_seconds'], manifest['threads'], checkpoint,
                               data_hash=manifest['hashes']['data.json'], checkpoint_validator=check)
    if metadata is None or sha(checkpoint) != digest:
        raise ValueError('Checkpoint changed during evaluation or was not validated')
    payload = {'schema': 1, 'arm': task['arm'], 'checkpoint_metadata': metadata,
               'checkpoint_sha256': digest, 'result': result}
    write_json(output, payload)
    return payload


def execute(args):
    run = Path(args.directory).resolve()
    manifest = verify(run)
    task = find_task(manifest, args.task)
    folder = task_directory(run, task)
    folder.mkdir(exist_ok=False)
    (folder / 'sequences').mkdir()
    started = time.monotonic()
    cpu_model = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    record = {'task': task, 'status': 'running', 'hostname': socket.gethostname(),
              'cpu_model': cpu_model, 'platform': platform.platform(), 'software': software(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'phases': []}
    payload = {'schema': 1, 'kind': 'finetune_endpoint_evaluation', 'status': 'running',
               'split': 'validation', 'final_test_access': False, 'selection': 'fixed_training_endpoint',
               'method': 'model', 'arm': task['arm'], 'seed': task['seed'],
               'threads': manifest['threads'], 'ids': manifest['validation_ids'],
               'checkpoint_sha256': None, 'checkpoint_metadata': None,
               'dataset_file_sha256': manifest['hashes']['data.json'],
               'evaluation_seconds_per_sequence_per_seed': task['evaluation_seconds'],
               'model_setup_charged_per_sequence': True, 'results': []}
    write_json(folder / 'launch.json', record)
    write_json(folder / 'evaluation.json', payload)

    def child_phase(name, command, timeout, log):
        start = time.monotonic()
        remaining = task['slurm_seconds'] - (start - started) - 30
        if remaining <= 0:
            raise TimeoutError('Fine-tuning task deadline')
        phase = {'name': name, 'exit_code': None}
        record['phases'].append(phase)
        write_json(folder / 'launch.json', record)
        try:
            with log.open('w') as handle:
                child = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                       timeout=min(timeout, remaining))
            phase['exit_code'] = child.returncode
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; inspect {log.name}')
        except subprocess.TimeoutExpired:
            phase['timed_out'] = True
            raise
        finally:
            phase['seconds'] = time.monotonic() - start
            write_json(folder / 'launch.json', record)

    try:
        validate_runtime(manifest)
        parent = parent_path(run, task)
        if task['arm'] != 'parent':
            command = [sys.executable, str(run / 'bundle' / 'folding_policy' / 'finetune_train.py'),
                       '--data', str(run / 'data.json'), '--parent', str(parent),
                       '--arm', task['arm'], '--seed', str(task['seed']),
                       '--seconds', str(task['training_seconds']), '--threads', str(manifest['threads']),
                       '--learning-rate', str(task['learning_rate']), '--output', str(folder / 'training')]
            child_phase('training', command, task['training_seconds'] + 180, folder / 'training.log')
        checkpoint, digest, _ = endpoint(run, task, manifest)
        payload['checkpoint_sha256'] = digest
        write_json(folder / 'evaluation.json', payload)
        for sequence_id in manifest['validation_ids']:
            output = folder / 'sequences' / (sequence_id + '.json')
            command = [sys.executable, str(run / 'bundle' / 'folding_policy' / 'finetune_worker.py'),
                       'sequence', str(run), '--task', str(task['task']), '--id', sequence_id,
                       '--output', str(output)]
            child_phase(sequence_id, command, task['evaluation_seconds'] + 60,
                        folder / 'sequences' / (sequence_id + '.log'))
            item = json.loads(output.read_text())
            row = item['result']
            if (item.get('arm') != task['arm'] or item.get('checkpoint_sha256') != digest
                    or row.get('id') != sequence_id or row.get('method') != 'model'
                    or row.get('seed') != task['seed'] or row.get('budget_s') != task['evaluation_seconds']):
                raise ValueError('Sequence result differs from its frozen task')
            metadata = item.get('checkpoint_metadata')
            if metadata is None or (payload['checkpoint_metadata'] is not None and payload['checkpoint_metadata'] != metadata):
                raise ValueError('Checkpoint metadata changed between sequence evaluations')
            payload['checkpoint_metadata'] = metadata
            payload['results'].append(row)
            write_json(folder / 'evaluation.json', payload)
        if sha(checkpoint) != digest:
            raise ValueError('Checkpoint changed during task execution')
        record['status'] = payload['status'] = 'complete'
    except Exception as exc:
        record['status'] = payload['status'] = 'failed'
        record['error'] = payload['error'] = str(exc)
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - started
        write_json(folder / 'launch.json', record)
        write_json(folder / 'evaluation.json', payload)
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
        expected = task_directory(run, task) / 'sequences' / (args.id + '.json')
        if args.output.resolve() != expected.resolve():
            parser.error('Sequence output must belong to its frozen task')
        evaluate_one(run, task, manifest, args.id, args.output)


if __name__ == '__main__':
    main()
