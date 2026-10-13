"""Freeze and submit the validation-only two-hour training-allocation pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

from launch import sha, software, write_json

HERE = Path(__file__).resolve().parent
ARMS = ('patternboost', 'rl_scratch', 'balanced', 'more_pretrain')
SEEDS = (0, 1, 2)
TRAINING_SECONDS = {'patternboost': 7200, 'rl_scratch': 7200,
                    'balanced': 3600, 'more_pretrain': 1800}
SLURM_MINUTES = {'patternboost': 210, 'rl_scratch': 150,
                 'balanced': 90, 'more_pretrain': 60}
RESUME_STAGES = {'balanced': 'stage_3600', 'more_pretrain': 'stage_5400'}
MILESTONES = [3600, 5400, 7200]


def validation_ids(data):
    from dataset import validate_dataset
    validate_dataset(data)
    rows = data['validation']
    if len(rows) != 16 or len({row['id'] for row in rows}) != 16:
        raise ValueError('Allocation pilot requires all 16 frozen validation sequences')
    return [row['id'] for row in rows]


def tasks():
    rows = []
    for arm in ARMS:
        stages = [f'stage_{seconds}' for seconds in MILESTONES] if arm == 'patternboost' else ['final']
        for seed in SEEDS:
            rows.append({
                'task': len(rows), 'arm': arm, 'seed': seed,
                'training_seconds': TRAINING_SECONDS[arm],
                'lineage_training_seconds': 7200,
                'slurm_seconds': SLURM_MINUTES[arm] * 60,
                'parent_arm': 'patternboost' if arm in RESUME_STAGES else None,
                'resume_stage': RESUME_STAGES.get(arm),
                'evaluation_stages': stages,
                'stage_lineage_seconds': {stage: int(stage.split('_')[1]) if stage != 'final' else 7200
                                         for stage in stages},
            })
    return rows


def validate_protocol(config):
    from train import DEFAULTS
    expected = {
        'stage': 'validation-only-allocation-pilot', 'conditions': list(ARMS),
        'seeds': list(SEEDS), 'total_training_seconds_per_condition_seed': 7200,
        'inference_seconds_per_sequence_seed': 60, 'threads': 4, 'memory_gib': 16,
        'array_concurrency': 3, 'slurm_minutes': SLURM_MINUTES,
        'final_test_enabled': False, 'automatic_extension': False,
        'model': {'width': 64, 'heads': 4, 'layers': 2, 'dropout': 0.0},
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Protocol differs from the bounded two-hour allocation pilot')
    if config.get('training') != {**DEFAULTS, 'validation_fraction': 0.05}:
        raise ValueError('Training objective settings differ from the preserved first pilot')
    allocation = config.get('allocation', {})
    if (allocation.get('milestones_seconds') != MILESTONES
            or allocation.get('initial_data_seconds') != 900
            or allocation.get('shared_prefix') is not True
            or allocation.get('checkpoint_selection') is not False):
        raise ValueError('Shared pretraining allocation changed')


def submission_command(run, arm, parent_job=None):
    command = ['sbatch', '--parsable', f'--chdir={run}',
               f'--job-name=hp-alloc-{arm}', f'--time={SLURM_MINUTES[arm]}',
               f'--output={run}/logs/{arm}_%A_%a.out',
               f'--error={run}/logs/{arm}_%A_%a.err']
    if arm in RESUME_STAGES:
        if parent_job is None:
            raise ValueError('An RL branch requires its PatternBoost array dependency')
        command.append(f'--dependency=aftercorr:{parent_job}')
    if os.environ.get('HP_MAIL_USER'):
        command.append('--mail-user=' + os.environ['HP_MAIL_USER'])
    return command + [str(run / 'bundle' / 'folding_policy' / 'allocation.sbatch'), str(run), arm]


def submit_arrays(run, dry_run=False):
    """Keep accepted jobs and their identifiers visible if a later submit fails."""
    record = {'status': 'prepared', 'dry_run': dry_run, 'arrays': [],
              'automatic_extension': False}
    parent = None
    for arm in ARMS:
        command = submission_command(run, arm, parent)
        row = {'arm': arm, 'command': command, 'submitted': False,
               'dependency': f'aftercorr:{parent}' if arm in RESUME_STAGES else None}
        record['arrays'].append(row)
        write_json(run / 'submission.json', record)
        if dry_run:
            print('Prepared only: ' + shlex.join(command))
            if arm == 'patternboost':
                parent = 'PARENT_ARRAY_JOB_ID'
            continue
        try:
            submitted = subprocess.run(command, text=True, capture_output=True, check=True)
            response = submitted.stdout.strip()
            row['sbatch_stderr'] = submitted.stderr
            match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', response)
            if not match:
                raise ValueError(f'Unexpected sbatch response: {response!r}')
            row.update(submitted=True, job_id=match.group(1), sbatch_response=response)
            if arm == 'patternboost':
                parent = row['job_id']
            print(f"Submitted {arm}: {row['job_id']} (three seeds).")
            write_json(run / 'job_ids.json', {item['arm']: item['job_id']
                                            for item in record['arrays'] if item.get('submitted')})
            (run / 'job_ids.txt').write_text(','.join(item['job_id'] for item in record['arrays']
                                                   if item.get('submitted')) + '\n')
        except Exception as exc:
            row['error'] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row['sbatch_output'] = exc.output
                row['sbatch_stderr'] = exc.stderr
            record['status'] = 'partial_submission' if any(item['submitted'] for item in record['arrays']) else 'failed'
            write_json(run / 'submission.json', record)
            accepted = ', '.join(item['job_id'] for item in record['arrays'] if item.get('submitted')) or 'none'
            raise RuntimeError(f'Submission failed for {arm}; accepted arrays remain scheduled: {accepted}. '
                               f'Inspect {run / "submission.json"}; do not resubmit this directory.') from exc
        write_json(run / 'submission.json', record)
    record['status'] = 'prepared' if dry_run else 'submitted'
    write_json(run / 'submission.json', record)
    return record


def prepare(args):
    run = Path(args.directory).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    required = ('dataset.py', 'train.py', 'model.py', 'geometry.py', 'evaluate.py', 'launch.py',
                'allocation_train.py', 'allocation_evaluate.py', 'allocation_report.py',
                'allocation_launch.py', 'allocation_protocol.json', 'allocation.sbatch')
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete allocation implementation: {missing}')
    config = json.loads((HERE / 'allocation_protocol.json').read_text())
    validate_protocol(config)
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python executable missing: {python}')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise ValueError('sbatch is unavailable; submit on Apocrita or use --dry-run')
    if not args.dry_run and software()['packages']['torch'] is None:
        raise ValueError('The cluster virtual environment must contain Torch before submission')
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    (run / 'tasks').mkdir()
    policy = run / 'bundle' / 'folding_policy'
    policy.mkdir(parents=True)
    for source in HERE.iterdir():
        if source.is_file() and source.suffix in {'.py', '.json', '.sh', '.sbatch', '.md', '.txt'}:
            shutil.copy2(source, policy / source.name)
    cluster = run / 'bundle' / 'cluster'
    cluster.mkdir()
    for name in ('hp_solver.py', 'geometry.py', 'protocol.json'):
        shutil.copy2(HERE.parent / 'cluster' / name, cluster / name)
    subprocess.run([str(python), str(policy / 'dataset.py'), '--output', str(run / 'data.json')], check=True)
    ids = validation_ids(json.loads((run / 'data.json').read_text()))
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(HERE), 'status', '--porcelain'], capture_output=True, text=True)
    files = sorted(path for path in (run / 'bundle').rglob('*') if path.is_file()) + [run / 'data.json']
    manifest = {
        'schema': 1, 'kind': 'validation_allocation_pilot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': git.stdout.strip() if git.returncode == 0 else None,
        'source_had_uncommitted_files': bool(dirty.stdout.strip()),
        'python_executable': str(python), 'software_at_freeze': software(),
        'hashes': {str(path.relative_to(run)): sha(path) for path in files},
        'threads': 4, 'memory_gib': 16, 'array_concurrency': 3,
        'seeds': list(SEEDS), 'arms': list(ARMS), 'tasks': tasks(),
        'evaluation_seconds_per_sequence_per_seed': 60,
        'evaluation_seeds': len(SEEDS), 'validation_ids': ids,
        'total_evaluation_seconds_per_sequence_per_method': 180,
        'stage_selection': 'fixed_training_endpoints',
        'lineage_training_seconds_per_method_seed': 7200,
        'physical_training_cpu_hours': 66,
        'planned_endpoint_evaluation_cpu_hours': 19.2,
        'reserved_cpu_hours_ceiling': 102,
        'final_test_access': False, 'no_automatic_followup': True,
        'dependency': 'aftercorr on matching PatternBoost seed; branches begin after parent evaluation',
    }
    write_json(run / 'allocation.json', manifest)
    (run / 'allocation.sha256').write_text(sha(run / 'allocation.json') + '\n')
    print(json.dumps({'directory': str(run), 'tasks': len(manifest['tasks']),
                      'validation_sequences': len(ids), 'lineage_training_hours_per_condition_seed': 2,
                      'planned_physical_cpu_hours_before_overhead': 85.2,
                      'reserved_cpu_hours_ceiling': 102}))
    submit_arrays(run, args.dry_run)
    if not args.dry_run:
        print('Validation allocation pilot only. No final test or longer campaign follows automatically.')
    return manifest


def verify(run):
    run = Path(run).resolve()
    if (run / 'allocation.sha256').read_text().strip() != sha(run / 'allocation.json'):
        raise ValueError('Frozen allocation manifest changed')
    manifest = json.loads((run / 'allocation.json').read_text())
    if manifest.get('kind') != 'validation_allocation_pilot' or manifest.get('final_test_access') is not False:
        raise ValueError('Allocation pilot may only evaluate validation sequences')
    if manifest.get('no_automatic_followup') is not True or manifest.get('tasks') != tasks():
        raise ValueError('Frozen allocation task plan changed')
    for relative, expected in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Frozen artifact changed: {relative}')
    ids = validation_ids(json.loads((run / 'data.json').read_text()))
    if ids != manifest.get('validation_ids'):
        raise ValueError('Validation selection changed')
    validate_protocol(json.loads((run / 'bundle' / 'folding_policy' / 'allocation_protocol.json').read_text()))
    return manifest


def execute(args):
    run = Path(args.directory).resolve()
    manifest = verify(run)
    task = next((row for row in manifest['tasks'] if row['arm'] == args.arm and row['seed'] == args.seed), None)
    if task is None:
        raise ValueError('Arm and seed are outside the frozen allocation pilot')
    folder = run / 'tasks' / f'{args.arm}_seed{args.seed}'
    folder.mkdir(exist_ok=False)
    (folder / 'evaluations').mkdir()
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
    try:
        if record['software'] != manifest['software_at_freeze']:
            raise ValueError('Python or package versions changed since the allocation was frozen')
        if int(os.environ.get('SLURM_CPUS_PER_TASK', manifest['threads'])) != manifest['threads']:
            raise ValueError('CPU allocation does not match frozen pilot')
        mode = 'rl_branch' if args.arm in RESUME_STAGES else args.arm
        train_command = [sys.executable, str(policy / 'allocation_train.py'), '--data', str(run / 'data.json'),
                         '--mode', mode, '--seed', str(args.seed), '--seconds', str(task['training_seconds']),
                         '--threads', str(manifest['threads']), '--output', str(folder / 'training'),
                         '--config', str(policy / 'allocation_protocol.json')]
        if task['parent_arm']:
            parent = run / 'tasks' / f"{task['parent_arm']}_seed{args.seed}"
            if not (parent / 'launch.json').is_file() or json.loads((parent / 'launch.json').read_text()).get('status') != 'complete':
                raise ValueError('PatternBoost parent did not finish successfully')
            checkpoint = parent / 'training' / (task['resume_stage'] + '.pt')
            if not checkpoint.is_file():
                raise ValueError(f'Parent training checkpoint missing: {checkpoint}')
            parent_result = json.loads((parent / 'training' / 'result.json').read_text())
            saved_records = [row for row in parent_result.get('checkpoint_records', [])
                             if row.get('filename') == checkpoint.name]
            expected_parent = {'status': 'complete', 'mode': 'patternboost', 'seed': args.seed,
                               'threads': manifest['threads'], 'budget_s': 7200,
                               'training_seconds': 7200, 'dataset_sha256': sha(run / 'data.json')}
            if (any(parent_result.get(key) != value for key, value in expected_parent.items())
                    or len(saved_records) != 1
                    or saved_records[0].get('file_sha256') != sha(checkpoint)
                    or saved_records[0].get('training_seconds') != int(task['resume_stage'].split('_')[1])):
                raise ValueError('Parent checkpoint differs from the completed training record')
            train_command += ['--resume', str(checkpoint)]
            record.update(parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=sha(checkpoint))
        phases = [('training', train_command, task['training_seconds'] + 120)]
        for stage in task['evaluation_stages']:
            command = [sys.executable, str(policy / 'allocation_evaluate.py'), '--data', str(run / 'data.json'),
                       '--checkpoint', str(folder / 'training' / (stage + '.pt')),
                       '--seed', str(args.seed), '--seconds', str(manifest['evaluation_seconds_per_sequence_per_seed']),
                       '--threads', str(manifest['threads']), '--output', str(folder / 'evaluations' / (stage + '.json')),
                       '--ids', *manifest['validation_ids']]
            timeout = len(manifest['validation_ids']) * manifest['evaluation_seconds_per_sequence_per_seed'] + 90
            phases.append(('evaluation_' + stage, command, timeout))
        for name, command, timeout in phases:
            phase_start = time.monotonic()
            remaining = task['slurm_seconds'] - (phase_start - start) - 15
            if remaining <= 0:
                raise TimeoutError('Allocation task deadline')
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
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else execute(args)


if __name__ == '__main__':
    main()
