"""Standalone inference checks on validation sequences only; no output repair."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time


def verified_score(sequence, positions):
    """Independent pairwise check, separate from training reward bookkeeping."""
    if len(positions) != len(sequence) or len(sequence) < 2 or set(sequence) - {'H', 'P'}:
        raise ValueError('A complete HP fold is required')
    points = [tuple(p) for p in positions]
    if any(len(p) != 3 or any(type(v) is not int for v in p) for p in points):
        raise ValueError('Integer coordinate triples required')
    if len(set(points)) != len(points):
        raise ValueError('Repeated occupied site')
    if points[:2] != [(0, 0, 0), (1, 0, 0)]:
        raise ValueError('Anchors differ from the common benchmark domain')
    if any(abs(v) > len(sequence) // 2 for p in points for v in p):
        raise ValueError('Fold outside the declared cube')
    if any(sum(abs(a-b) for a, b in zip(p, q)) != 1 for p, q in zip(points, points[1:])):
        raise ValueError('Non-unit backbone bond')
    return sum(sequence[i] == sequence[j] == 'H' and
               sum(abs(a-b) for a, b in zip(points[i], points[j])) == 1
               for i in range(len(points)) for j in range(i+2, len(points)))


def load_cp_solver():
    """Use the preserved CP model while isolating its bare geometry import."""
    folder = Path(__file__).resolve().parents[1] / 'cluster'
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    geometry = load('_policy_independent_cluster_geometry', folder / 'geometry.py')
    previous = sys.modules.get('geometry')
    sys.modules['geometry'] = geometry
    try:
        return load('_policy_cluster_cp_solver', folder / 'hp_solver.py').solve
    finally:
        if previous is None:
            del sys.modules['geometry']
        else:
            sys.modules['geometry'] = previous


def validate_checkpoint(saved, data_hash):
    if saved.get('dataset_sha256') != data_hash:
        raise ValueError('Checkpoint belongs to a different dataset')
    selection = saved.get('selection', {})
    if selection.get('split') != 'validation' or selection.get('validated') is not True:
        raise ValueError('Checkpoint has not completed validation selection')


def evaluate_sequence(row, method, seed, seconds, threads, checkpoint=None, batch_size=8, data_hash=None):
    start = time.monotonic()
    deadline = start + seconds
    sequence = row['sequence']
    result = {'id': row['id'], 'sequence': sequence, 'length': len(sequence), 'seed': seed,
              'method': method, 'budget_s': seconds, 'best_contacts': None, 'best_positions': None,
              'curve': [], 'attempted_folds': 0, 'generated_complete_folds': 0,
              'verified_witnesses': 0, 'trapped_folds': 0, 'interrupted_folds': 0,
              'discarded_after_budget': 0, 'solver_status': None, 'bound': None,
              'proof_within_budget': False, 'bound_recorded_s': None}

    def retain(positions, claimed=None):
        score = verified_score(sequence, positions)
        if claimed is not None and score != claimed:
            raise ValueError('Reported contacts disagree with independent verifier')
        elapsed = time.monotonic() - start
        if elapsed > seconds:
            result['discarded_after_budget'] += 1
            return
        result['verified_witnesses'] += 1
        if result['best_contacts'] is None or score > result['best_contacts']:
            result['best_contacts'] = score
            result['best_positions'] = positions
            result['curve'].append({'seconds': elapsed, 'contacts': score, 'positions': positions})

    if method == 'cp_sat':
        solve = load_cp_solver()
        result['setup_s'] = time.monotonic() - start
        result['attempted_folds'] = None
        result['generated_complete_folds'] = None
        def emit(event):
            if event['type'] == 'witness':
                retain(event['positions'], event['contacts'])
            elif event['type'] == 'solver_end':
                result['solver_status'] = event['status']
                result['bound'] = event.get('bound')
                result['bound_scope'] = event.get('scope')
                result['bound_recorded_s'] = time.monotonic() - start
                result['proof_within_budget'] = (event['status'] == 'OPTIMAL'
                    and event.get('scope') == 'full_declared_cube' and time.monotonic() <= deadline)
        if time.monotonic() < deadline:
            solve(sequence, seconds=max(0, deadline-time.monotonic()), threads=threads,
                  seed=seed, emit=emit, stopped=lambda: time.monotonic() >= deadline)
    else:
        if method == 'model':
            import torch
            from model import Policy, sample
            torch.set_num_threads(threads)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            validate_checkpoint(saved, data_hash)
            policy = Policy(saved['model_config'])
            policy.load_state_dict(saved['model_state'])
            policy.eval()
            generator = torch.Generator(device='cpu').manual_seed(seed)
            result['checkpoint_stage'] = saved.get('stage')
            result['checkpoint_condition'] = saved.get('condition')
            result['checkpoint_seed'] = saved.get('seed')
            result['checkpoint_selection'] = saved.get('selection')
            result['checkpoint_dataset_sha256'] = saved.get('dataset_sha256')
            def draw():
                return sample(policy, [sequence] * batch_size, generator=generator, deadline=deadline)
        else:
            from geometry import greedy
            rng = random.Random(seed)
            def draw():
                return [greedy(sequence, rng=rng, attempts=1, deadline=deadline)]
        result['setup_s'] = time.monotonic() - start
        while time.monotonic() < deadline:
            trajectories = draw()
            for trajectory in trajectories:
                result['attempted_folds'] += 1
                if trajectory and trajectory.get('complete'):
                    if time.monotonic() <= deadline:
                        result['generated_complete_folds'] += 1
                    retain(trajectory['positions'], trajectory.get('contacts'))
                elif (trajectory and trajectory.get('interrupted')) or time.monotonic() >= deadline:
                    result['interrupted_folds'] += 1
                else:
                    result['trapped_folds'] += 1
    result['elapsed_s'] = time.monotonic() - start
    result['valid_fold_rate'] = (result['generated_complete_folds'] / result['attempted_folds']
                                if result['attempted_folds'] else None)
    result['gap_contacts'] = (result['bound'] - result['best_contacts']
                             if result['bound'] is not None and result['best_contacts'] is not None else None)
    result['relative_gap'] = (result['gap_contacts'] / max(1, abs(result['best_contacts']))
                              if result['gap_contacts'] is not None else None)
    result['bound_recorded_after_budget'] = (result['bound_recorded_s'] > seconds
                                            if result['bound_recorded_s'] is not None else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['validation'], default='validation')
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--method', choices=['model', 'greedy', 'cp_sat'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads < 1:
        parser.error('Positive seconds and threads are required')
    if args.method == 'model' and args.checkpoint is None:
        parser.error('Model inference requires a validation-selected checkpoint')
    if args.output.exists():
        parser.error('Refusing to overwrite existing evaluation results')
    data = json.loads(args.data.read_text())
    from dataset import validate_dataset
    validate_dataset(data)
    rows = {row['id']: row for row in data[args.split]}
    if len(args.ids) != len(set(args.ids)) or any(name not in rows for name in args.ids):
        parser.error('Every requested sequence must occur once in the validation partition')
    result = {'schema': 1, 'split': 'validation', 'final_test_access': False,
              'method': args.method, 'seed': args.seed, 'threads': args.threads,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              'dataset_file_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'model_setup_charged_per_sequence': True,
              'generated_count_note': 'Model/greedy complete folds counted within budget; CP search nodes are not generated folds.',
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.ids:
        result['results'].append(evaluate_sequence(rows[name], args.method, args.seed,
                                args.seconds, args.threads, args.checkpoint,
                                data_hash=result['dataset_file_sha256']))
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'sequences': len(result['results']), 'split': 'validation'}))


if __name__ == '__main__':
    main()
