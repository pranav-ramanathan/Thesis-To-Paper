"""Evaluate declared training endpoints on validation, without checkpoint selection.

This separate entry point accepts the allocation pilot's complete training states.
The original pilot evaluator continues to require validation-selected checkpoints.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from dataset import validate_dataset
from evaluate import evaluate_sequence
from train import atomic_json, state_hash


def validate_allocation_checkpoint(saved, data_hash, seed=None):
    if saved.get('schema') != 1 or saved.get('kind') != 'allocation_training_state':
        raise ValueError('Expected a complete allocation training state')
    if (saved.get('final_test_enabled') is not False or saved.get('test_evaluated') is not False
            or saved.get('selection', {}).get('selected') is not False):
        raise ValueError('Expected a fixed endpoint with no final-test access or checkpoint selection')
    if saved.get('dataset_sha256') != data_hash:
        raise ValueError('Checkpoint belongs to a different dataset')
    if seed is not None and saved.get('seed') != seed:
        raise ValueError('Checkpoint seed differs from its declared evaluation seed')
    mode = saved.get('mode')
    expected = {'patternboost': ('patternboost', 'patternboost'),
                'rl_scratch': ('rl_scratch', 'rl'),
                'rl_branch': ('patternboost_rl', 'rl')}
    if mode not in expected or (saved.get('condition'), saved.get('stage')) != expected[mode]:
        raise ValueError('Checkpoint training mode, condition or stage is inconsistent')
    seconds = saved.get('training_seconds')
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Checkpoint lacks a positive cumulative training budget')
    for name in ('model_state', 'optimizer_state', 'rng_states', 'elite_pool'):
        if name not in saved:
            raise ValueError(f'Incomplete training state: missing {name}')
    if saved.get('state_sha256') != state_hash(saved['model_state']):
        raise ValueError('Checkpoint parameter hash does not match its weights')


def evaluate_endpoint(data_path, checkpoint, seed, seconds, threads, output, ids):
    data_path, checkpoint, output = map(Path, (data_path, checkpoint, output))
    if output.exists():
        raise ValueError('Refusing to overwrite an existing endpoint evaluation')
    if not math.isfinite(seconds) or seconds <= 0 or threads < 1:
        raise ValueError('Positive inference seconds and CPU thread count required')
    raw = data_path.read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    validation = {row['id']: row for row in data['validation']}
    if not ids or len(ids) != len(set(ids)) or any(name not in validation for name in ids):
        raise ValueError('Each requested ID must occur once in the validation partition')
    payload = {
        'schema': 1, 'kind': 'allocation_endpoint_evaluation', 'status': 'running',
        'split': 'validation', 'final_test_access': False,
        'selection': 'fixed_training_endpoint', 'method': 'model', 'seed': seed,
        'threads': threads, 'ids': list(ids), 'checkpoint': str(checkpoint.resolve()),
        'endpoint': checkpoint.stem,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'dataset_file_sha256': hashlib.sha256(raw).hexdigest(),
        'evaluation_seconds_per_sequence_per_seed': seconds,
        'model_setup_charged_per_sequence': True,
        'checkpoint_metadata': None, 'results': [],
    }

    def check(saved, data_hash):
        validate_allocation_checkpoint(saved, data_hash, seed)
        metadata = {key: saved.get(key) for key in
                    ('mode', 'stage', 'condition', 'seed', 'training_seconds',
                     'dataset_sha256', 'state_sha256', 'kind', 'schema')}
        if payload['checkpoint_metadata'] is not None and payload['checkpoint_metadata'] != metadata:
            raise ValueError('Checkpoint changed during endpoint evaluation')
        payload['checkpoint_metadata'] = metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in ids:
            result = evaluate_sequence(validation[name], 'model', seed, seconds, threads,
                                       checkpoint, data_hash=payload['dataset_file_sha256'],
                                       checkpoint_validator=check)
            payload['results'].append(result)
            atomic_json(output, payload)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != payload['checkpoint_sha256']:
            raise ValueError('Checkpoint file changed during evaluation')
        payload['status'] = 'complete'
        atomic_json(output, payload)
    except Exception as exc:
        payload['status'], payload['error'] = 'failed', str(exc)
        atomic_json(output, payload)
        raise
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    result = evaluate_endpoint(args.data, args.checkpoint, args.seed, args.seconds,
                               args.threads, args.output, args.ids)
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'sequences': len(result['results']), 'split': 'validation'}))


if __name__ == '__main__':
    main()
