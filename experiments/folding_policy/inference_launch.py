"""Freeze saved allocation policies and submit validation-only inference searches."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from launch import sha, software, write_json

HERE = Path(__file__).resolve().parent
MODEL_ARMS = ('patternboost', 'balanced', 'more_pretrain', 'rl_scratch')
ARMS = MODEL_ARMS + ('cp_sat', 'greedy')
SEEDS = (0, 1, 2)
ARRAYS = {'models': {'array': '0-11%12', 'minutes': 180},
          'baselines': {'array': '12-17%6', 'minutes': 300}}
PROTOCOL = {
    'stage': 'validation-only-inference-comparison', 'arms': list(ARMS),
    'seeds': list(SEEDS), 'threads': 4, 'memory_gib': 16,
    'model_seconds_per_sequence_seed': 600,
    'baseline_seconds_per_sequence_seed': 1050,
    'inference_timepoints_s': [60, 300, 600],
    'amortization_sequences': 16, 'training_lineage_seconds': 7200,
    'arrays': ARRAYS, 'checkpoint_selection': 'fixed_training_endpoints',
    'new_training_enabled': False, 'final_test_enabled': False,
    'automatic_extension': False,
}
NEW_FILES = ('inference_launch.py', 'inference_worker.py', 'inference_report.py',
             'inference_protocol.json', 'inference.sbatch')


def task_directory(run, task):
    return Path(run) / 'tasks' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}"


def tasks(checkpoints):
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            model = arm in MODEL_ARMS
            info = checkpoints.get((arm, seed), {})
            rows.append({'task': len(rows), 'arm': arm, 'seed': seed,
                         'method': 'model' if model else arm,
                         'seconds': 600 if model else 1050,
                         'checkpoint': f'checkpoints/{arm}_seed{seed}.pt' if model else None,
                         'checkpoint_sha256': info.get('file_sha256') if model else None,
                         'state_sha256': info.get('state_sha256') if model else None,
                         'training_seconds': 7200 if model else 0,
                         'slurm_seconds': (180 if model else 300) * 60})
    return rows


def validate_protocol(config):
    if config != PROTOCOL:
        raise ValueError('Protocol differs from the bounded inference-only comparison')


def validate_source(source):
    """Read source receipts and checkpoints without writing to the finished campaign."""
    from allocation_launch import verify
    from allocation_report import _record, _training_errors
    source = Path(source).resolve()
    manifest = verify(source)
    if manifest.get('threads') != 4 or len(manifest.get('validation_ids', [])) != 16:
        raise ValueError('Source allocation must use four cores and all sixteen validation sequences')
    trained, by_key, receipts, endpoints = {}, {}, [], {}
    for task in manifest['tasks']:
        key = (task['arm'], task['seed'])
        by_key[key] = task
        folder = source / 'tasks' / f"{task['arm']}_seed{task['seed']}"
        result_path, launch_path = folder / 'training' / 'result.json', folder / 'launch.json'
        receipt = json.loads(result_path.read_text())
        launched = json.loads(launch_path.read_text())
        if (launched.get('status') != 'complete' or receipt.get('status') != 'complete'
                or _training_errors(task, receipt, manifest)):
            raise ValueError(f'Source training or allocation task did not complete correctly: {key}')
        if (receipt.get('threads') != 4 or receipt.get('training_seconds') != 7200
                or receipt.get('final_test_enabled') is not False
                or receipt.get('selection', {}).get('selected') is not False
                or launched.get('software') != manifest.get('software_at_freeze')):
            raise ValueError(f'Source training budget, settings or software changed: {key}')
        trained[key] = receipt
        receipts.extend((result_path, launch_path))
        stage = 'stage_7200' if task['arm'] == 'patternboost' else 'final'
        checkpoint = folder / 'training' / f'{stage}.pt'
        record = _record(receipt, checkpoint.name)
        if (record.get('file_sha256') != sha(checkpoint)
                or record.get('training_seconds') != 7200 or not record.get('state_sha256')
                or record.get('selected') is not False):
            raise ValueError(f'Source endpoint does not match its fixed training receipt: {key}')
        endpoints[key] = {**record, 'source_path': checkpoint}
    for key, task in by_key.items():
        if not task.get('parent_arm'):
            continue
        parent_key = (task['parent_arm'], task['seed'])
        parent_name = f"{task['resume_stage']}.pt"
        parent = _record(trained[parent_key], parent_name)
        checkpoint = source / 'tasks' / f'{parent_key[0]}_seed{parent_key[1]}' / 'training' / parent_name
        if (sha(checkpoint) != parent.get('file_sha256')
                or parent.get('training_seconds') != int(task['resume_stage'].split('_')[1])):
            raise ValueError(f'Shared parent checkpoint changed: {key}')
        receipts.append(checkpoint)
        child = trained[key]
        lineage, handoff = child.get('lineage', {}), child.get('handoff') or {}
        state, optimizer = parent.get('state_sha256'), parent.get('optimizer_sha256')
        if (not state or lineage.get('parent_file_sha256') != parent['file_sha256']
                or lineage.get('parent_state_sha256') != state
                or child.get('initial_state_sha256') != state
                or handoff.get('pretraining_sha256') != state
                or handoff.get('rl_initial_sha256') != state):
            raise ValueError(f'Branch does not retain its declared shared parent weights: {key}')
        if (not optimizer or lineage.get('parent_optimizer_sha256') != optimizer
                or handoff.get('pretraining_optimizer_sha256') != optimizer
                or handoff.get('rl_initial_optimizer_sha256') != optimizer
                or any(handoff.get(flag) is not True for flag in
                       ('optimizer_state_retained', 'rng_state_retained', 'elite_pool_retained'))):
            raise ValueError(f'Branch does not retain parent optimizer, RNG and elite pool: {key}')
    receipts.extend((source / 'allocation.json', source / 'allocation.sha256'))
    return manifest, endpoints, receipts


def submission_command(run, group):
    settings = ARRAYS[group]
    command = ['sbatch', '--parsable', f'--chdir={run}', f'--job-name=hp-infer-{group}',
               f"--array={settings['array']}", f"--time={settings['minutes']}",
               f'--output={run}/logs/{group}_%A_%a.out', f'--error={run}/logs/{group}_%A_%a.err']
    if os.environ.get('HP_MAIL_USER'):
        command.append('--mail-user=' + os.environ['HP_MAIL_USER'])
    return command + [str(run / 'bundle' / 'folding_policy' / 'inference.sbatch'), str(run)]


def submit_arrays(run, dry_run=False):
    """Keep accepted arrays visible and never retry a partial submission."""
    record = {'status': 'prepared', 'dry_run': dry_run, 'arrays': [], 'automatic_extension': False}
    for group in ARRAYS:
        command = submission_command(run, group)
        row = {'group': group, 'command': command, 'submitted': False}
        record['arrays'].append(row)
        write_json(run / 'submission.json', record)
        if dry_run:
            print('Prepared only: ' + shlex.join(command))
            continue
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            response = result.stdout.strip()
            row['sbatch_stderr'] = result.stderr
            match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', response)
            if not match:
                raise ValueError(f'Unexpected sbatch response: {response!r}')
            row.update(submitted=True, job_id=match.group(1), sbatch_response=response)
            write_json(run / 'job_ids.json', {item['group']: item['job_id'] for item in record['arrays']
                                            if item.get('submitted')})
            temporary = run / 'job_ids.txt.tmp'
            temporary.write_text(','.join(item['job_id'] for item in record['arrays']
                                          if item.get('submitted')) + '\n')
            temporary.replace(run / 'job_ids.txt')
            print(f"Submitted {group}: {row['job_id']}.")
        except Exception as exc:
            row['error'] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row.update(sbatch_output=exc.output, sbatch_stderr=exc.stderr)
            record['status'] = 'partial_submission' if any(item['submitted'] for item in record['arrays']) else 'failed'
            write_json(run / 'submission.json', record)
            accepted = ', '.join(item['job_id'] for item in record['arrays'] if item.get('submitted')) or 'none'
            raise RuntimeError(f'Submission failed for {group}; accepted arrays remain scheduled: {accepted}. '
                               f'Inspect {run / "submission.json"}; do not resubmit this directory.') from exc
        write_json(run / 'submission.json', record)
    record['status'] = 'prepared' if dry_run else 'submitted'
    write_json(run / 'submission.json', record)
    return record


def prepare(args):
    run, source = Path(args.directory).expanduser().resolve(), Path(args.source_allocation).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    if run.is_relative_to(source):
        raise ValueError('Inference output must be outside the completed source allocation')
    missing = [name for name in NEW_FILES if not (HERE / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete inference implementation: {missing}')
    validate_protocol(json.loads((HERE / 'inference_protocol.json').read_text()))
    source_manifest, endpoints, receipts = validate_source(source)
    current_software = software()
    if current_software != source_manifest['software_at_freeze']:
        raise ValueError('Python or package versions differ from source model training')
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python executable missing: {python}')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise ValueError('sbatch is unavailable; submit on Apocrita or use --dry-run')
    if not args.dry_run and any(current_software['packages'][name] is None for name in ('torch', 'numpy', 'ortools')):
        raise ValueError('The cluster virtual environment requires Torch, NumPy and OR-Tools')
    run.mkdir(parents=True)
    for name in ('logs', 'tasks', 'checkpoints', 'source_receipts'):
        (run / name).mkdir()
    # Original model, geometry, evaluator and CP-SAT solver stay byte-identical to training.
    shutil.copytree(source / 'bundle', run / 'bundle')
    policy = run / 'bundle' / 'folding_policy'
    for name in NEW_FILES:
        shutil.copy2(HERE / name, policy / name)
    for name in ('inference_pilot.sh', 'inference_report.sh'):
        if (HERE / name).is_file():
            shutil.copy2(HERE / name, policy / name)
    shutil.copy2(source / 'data.json', run / 'data.json')
    for (arm, seed), info in endpoints.items():
        target = run / 'checkpoints' / f'{arm}_seed{seed}.pt'
        shutil.copy2(info['source_path'], target)
        if sha(target) != info['file_sha256']:
            raise ValueError('Source checkpoint changed during copying')
    for path in receipts:
        target = run / 'source_receipts' / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Detect changes during the freeze, including source receipts not hashed by allocation.json.
    second_manifest, second_endpoints, _ = validate_source(source)
    if (second_manifest != source_manifest
            or {key: value['file_sha256'] for key, value in second_endpoints.items()}
            != {key: value['file_sha256'] for key, value in endpoints.items()}
            or any(sha(path) != sha(run / 'source_receipts' / path.relative_to(source)) for path in receipts)):
        raise ValueError('Source allocation changed while the inference run was frozen')
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    files = sorted(path for name in ('bundle', 'checkpoints', 'source_receipts')
                   for path in (run / name).rglob('*') if path.is_file()) + [run / 'data.json']
    manifest = {
        'schema': 1, 'kind': 'validation_inference_comparison',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': git.stdout.strip() if git.returncode == 0 else None,
        'source_allocation': str(source), 'source_allocation_sha256': sha(source / 'allocation.json'),
        'source_training_revision': source_manifest.get('source_revision'),
        'python_executable': str(python), 'software_at_freeze': current_software,
        'hashes': {str(path.relative_to(run)): sha(path) for path in files},
        'threads': 4, 'memory_gib': 16, 'seeds': list(SEEDS), 'arms': list(ARMS),
        'tasks': tasks(endpoints), 'validation_ids': source_manifest['validation_ids'],
        'inference_timepoints_s': [60, 300, 600], 'amortization_sequences': 16,
        'training_lineage_seconds': 7200, 'model_seconds_per_sequence_seed': 600,
        'baseline_seconds_per_sequence_seed': 1050, 'arrays': ARRAYS,
        'planned_inference_cpu_hours': 240, 'reserved_cpu_hours_ceiling': 264,
        'checkpoint_selection': 'fixed_training_endpoints',
        'new_training_enabled': False, 'final_test_access': False, 'no_automatic_followup': True,
    }
    write_json(run / 'inference.json', manifest)
    (run / 'inference.sha256').write_text(sha(run / 'inference.json') + '\n')
    verify(run)
    print(json.dumps({'directory': str(run), 'tasks': 18, 'validation_sequences': 16,
                      'planned_inference_cpu_hours': 240, 'reserved_cpu_hours_ceiling': 264,
                      'new_training': False, 'final_test_access': False}))
    submit_arrays(run, args.dry_run)
    return manifest


def verify(run):
    from allocation_launch import validation_ids
    run = Path(run).resolve()
    if (run / 'inference.sha256').read_text().strip() != sha(run / 'inference.json'):
        raise ValueError('Frozen inference manifest changed')
    manifest = json.loads((run / 'inference.json').read_text())
    expected = {'schema': 1, 'kind': 'validation_inference_comparison', 'threads': 4,
                'memory_gib': 16, 'seeds': list(SEEDS), 'arms': list(ARMS),
                'inference_timepoints_s': [60, 300, 600], 'amortization_sequences': 16,
                'training_lineage_seconds': 7200, 'model_seconds_per_sequence_seed': 600,
                'baseline_seconds_per_sequence_seed': 1050, 'arrays': ARRAYS,
                'checkpoint_selection': 'fixed_training_endpoints', 'new_training_enabled': False,
                'final_test_access': False, 'no_automatic_followup': True}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Inference plan differs from the frozen validation-only budgets')
    for relative, expected_hash in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != expected_hash:
            raise ValueError(f'Frozen artifact changed: {relative}')
    ids = validation_ids(json.loads((run / 'data.json').read_text()))
    if manifest.get('validation_ids') != ids or 'data.json' not in manifest['hashes']:
        raise ValueError('Frozen validation selection changed')
    validate_protocol(json.loads((run / 'bundle' / 'folding_policy' / 'inference_protocol.json').read_text()))
    provenance = run / 'source_receipts' / 'allocation.json'
    if sha(provenance) != manifest['source_allocation_sha256']:
        raise ValueError('Frozen source allocation receipt changed')
    source_manifest = json.loads(provenance.read_text())
    if source_manifest['hashes']['data.json'] != manifest['hashes']['data.json']:
        raise ValueError('Inference dataset differs from source training data')
    if source_manifest['software_at_freeze'] != manifest['software_at_freeze']:
        raise ValueError('Inference software differs from source training software')
    infos = {}
    for arm in MODEL_ARMS:
        for seed in SEEDS:
            path = run / 'source_receipts' / 'tasks' / f'{arm}_seed{seed}' / 'training' / 'result.json'
            trained = json.loads(path.read_text())
            filename = 'stage_7200.pt' if arm == 'patternboost' else 'final.pt'
            matches = [row for row in trained['checkpoint_records'] if row['filename'] == filename]
            if len(matches) != 1:
                raise ValueError('Frozen source lacks one unique endpoint receipt')
            infos[(arm, seed)] = matches[0]
    if manifest.get('tasks') != tasks(infos):
        raise ValueError('Frozen inference task plan or checkpoint provenance changed')
    for task in manifest['tasks']:
        if task['method'] == 'model' and manifest['hashes'].get(task['checkpoint']) != task['checkpoint_sha256']:
            raise ValueError('Checkpoint digest differs from source training receipt')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('directory')
    p.add_argument('--source-allocation', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('directory')
    p.add_argument('--task', required=True, type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        from inference_worker import execute
        execute(args)


if __name__ == '__main__':
    main()
