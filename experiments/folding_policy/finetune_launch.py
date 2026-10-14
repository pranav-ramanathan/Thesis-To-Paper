"""Freeze a bounded validation-only fine-tuning screen from shared PB parents."""
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

from inference_launch import validate_source
from launch import sha, software, write_json

HERE = Path(__file__).resolve().parent
ARMS = ('patternboost', 'grpo', 'dr_grpo', 'reinforce', 'dpo')
SEEDS = (0, 1, 2)
LEARNING_RATES = {'patternboost': 0.0003, 'grpo': 0.0003, 'dr_grpo': 0.00003,
                  'reinforce': 0.00003, 'dpo': 0.00003, 'parent': 0.0}
PROTOCOL = {
    'stage': 'validation-only-finetuning-screen', 'arms': list(ARMS),
    'seeds': list(SEEDS), 'threads': 4, 'memory_gib': 16,
    'model': {'width': 64, 'heads': 4, 'layers': 2, 'dropout': 0.0},
    'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
    'evaluation_seconds_per_sequence_per_seed': 60,
    'learning_rates': LEARNING_RATES, 'dpo_beta': 0.1,
    'array': '0-17%18', 'slurm_minutes': 75,
    'checkpoint_selection': 'fixed_training_endpoints',
    'comparison': 'fixed_configuration_screen_not_algorithm_ranking',
    'learning_rate_tuning': False, 'final_test_enabled': False,
    'automatic_extension': False,
}
NEW_FILES = ('finetune_launch.py', 'finetune_worker.py', 'finetune_train.py',
             'finetune_report.py', 'finetune_protocol.json', 'finetune.sbatch',
             'finetune_pilot.sh', 'finetune_report.sh',
             'inference_launch.py', 'inference_report.py')


def task_directory(run, task):
    return Path(run) / 'tasks' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}"


def tasks(parents):
    result = []
    for arm in ARMS + ('parent',):
        for seed in SEEDS:
            parent = parents[seed]
            training = 0 if arm == 'parent' else 1800
            result.append({'task': len(result), 'arm': arm, 'seed': seed,
                           'parent': f'parents/patternboost_seed{seed}.pt',
                           'parent_sha256': parent['file_sha256'],
                           'parent_state_sha256': parent['state_sha256'],
                           'parent_optimizer_sha256': parent['optimizer_sha256'],
                           'training_seconds': training,
                           'lineage_training_seconds': 7200 + training,
                           'learning_rate': LEARNING_RATES[arm],
                           'evaluation_seconds': 60, 'slurm_seconds': 4500})
    return result


def validate_protocol(config):
    if config != PROTOCOL:
        raise ValueError('Protocol differs from the bounded fine-tuning screen')


def submission_command(run):
    run = Path(run)
    command = ['sbatch', '--parsable', f'--chdir={run}', '--job-name=hp-finetune',
               '--array=0-17%18', '--time=75', f'--output={run}/logs/%A_%a.out',
               f'--error={run}/logs/%A_%a.err']
    if os.environ.get('HP_MAIL_USER'):
        command.append('--mail-user=' + os.environ['HP_MAIL_USER'])
    return command + [str(run / 'bundle/folding_policy/finetune.sbatch'), str(run)]


def submit_array(run, dry_run=False):
    """Submit once; ambiguous responses remain recorded and are never retried."""
    run = Path(run)
    path = run / 'submission.json'
    if path.exists():
        raise ValueError('Submission was already attempted; do not resubmit this directory')
    command = submission_command(run)
    record = {'status': 'prepared', 'dry_run': dry_run, 'command': command,
              'submitted': False, 'automatic_extension': False}
    write_json(path, record)
    if dry_run:
        print('Prepared only: ' + shlex.join(command))
        return record
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        response = result.stdout.strip()
        record.update(sbatch_response=response, sbatch_stderr=result.stderr)
        match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', response)
        if not match:
            record['status'] = 'submission_uncertain'
            raise ValueError(f'Unexpected sbatch response: {response!r}')
        record.update(status='submitted', submitted=True, job_id=match.group(1))
        write_json(run / 'job_ids.json', {'finetune': record['job_id']})
        temporary = run / 'job_ids.txt.tmp'
        temporary.write_text(record['job_id'] + '\n')
        temporary.replace(run / 'job_ids.txt')
        print(f"Submitted fine-tuning array {record['job_id']} (18 tasks).")
    except Exception as exc:
        record['error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            record.update(status='failed', sbatch_output=exc.output, sbatch_stderr=exc.stderr)
        elif record['status'] == 'prepared':
            record['status'] = 'submission_uncertain'
        write_json(path, record)
        raise RuntimeError(f'Submission did not finish cleanly. Inspect {path}; '
                           'check Slurm before any new submission. This directory cannot be resubmitted.') from exc
    write_json(path, record)
    return record


def _source_parents(source):
    manifest, endpoints, _ = validate_source(source)
    config = json.loads((source / 'bundle/folding_policy/allocation_protocol.json').read_text())
    if config.get('model') != PROTOCOL['model']:
        raise ValueError('Source PatternBoost architecture differs from the frozen screen')
    parents = {seed: endpoints[('patternboost', seed)] for seed in SEEDS}
    if any(not item.get('optimizer_sha256') for item in parents.values()):
        raise ValueError('Source parent optimizer provenance is missing')
    receipts = [source / 'allocation.json', source / 'allocation.sha256']
    for seed in SEEDS:
        receipts.extend(source / 'tasks' / f'patternboost_seed{seed}' / name
                        for name in ('launch.json', 'training/result.json'))
    return manifest, parents, receipts


def prepare(args):
    run = Path(args.directory).expanduser().resolve()
    source = Path(args.source_allocation).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    if run.is_relative_to(source):
        raise ValueError('Fine-tuning output must be outside the completed source allocation')
    missing = [name for name in NEW_FILES if not (HERE / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete fine-tuning implementation: {missing}')
    validate_protocol(json.loads((HERE / 'finetune_protocol.json').read_text()))
    source_manifest, parents, receipts = _source_parents(source)
    versions = software()
    if versions != source_manifest['software_at_freeze']:
        raise ValueError('Python or package versions differ from source model training')
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python executable missing: {python}')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise ValueError('sbatch is unavailable; submit on Apocrita or use --dry-run')
    if not args.dry_run and any(versions['packages'][name] is None for name in ('torch', 'numpy', 'ortools')):
        raise ValueError('The cluster virtual environment requires Torch, NumPy and OR-Tools')
    run.mkdir(parents=True)
    for name in ('logs', 'tasks', 'parents', 'source_receipts'):
        (run / name).mkdir()
    # Freeze the source code used to train parents; only declared new helpers change.
    shutil.copytree(source / 'bundle', run / 'bundle')
    for name in NEW_FILES:
        shutil.copy2(HERE / name, run / 'bundle/folding_policy' / name)
    shutil.copy2(source / 'data.json', run / 'data.json')
    for seed, parent in parents.items():
        target = run / 'parents' / f'patternboost_seed{seed}.pt'
        shutil.copy2(parent['source_path'], target)
        if sha(target) != parent['file_sha256']:
            raise ValueError('Source parent changed during copying')
    for path in receipts:
        target = run / 'source_receipts' / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    second_manifest, second_parents, _ = _source_parents(source)
    if (second_manifest != source_manifest
            or {seed: row['file_sha256'] for seed, row in second_parents.items()}
            != {seed: row['file_sha256'] for seed, row in parents.items()}
            or any(sha(path) != sha(run / 'source_receipts' / path.relative_to(source)) for path in receipts)):
        raise ValueError('Source allocation changed while the fine-tuning run was frozen')
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    files = sorted(path for name in ('bundle', 'parents', 'source_receipts')
                   for path in (run / name).rglob('*') if path.is_file()) + [run / 'data.json']
    manifest = {
        'schema': 1, 'kind': 'validation_finetune_pilot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': git.stdout.strip() if git.returncode == 0 else None,
        'source_allocation': str(source), 'source_allocation_sha256': sha(source / 'allocation.json'),
        'source_training_revision': source_manifest.get('source_revision'),
        'python_executable': str(python), 'software_at_freeze': versions,
        'hashes': {str(path.relative_to(run)): sha(path) for path in files},
        'threads': 4, 'memory_gib': 16, 'seeds': list(SEEDS), 'arms': list(ARMS),
        'tasks': tasks(parents), 'validation_ids': source_manifest['validation_ids'],
        'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
        'training_lineage_seconds': 9000, 'evaluation_seconds_per_sequence_per_seed': 60,
        'planned_new_training_cpu_hours': 30.0, 'planned_evaluation_cpu_hours': 19.2,
        'planned_new_cpu_hours': 49.2, 'reserved_cpu_hours_ceiling': 90.0,
        'historical_shared_parent_cpu_hours': 24.0,
        'historical_parent_training_repeated': False,
        'checkpoint_selection': 'fixed_training_endpoints',
        'comparison': PROTOCOL['comparison'], 'learning_rate_tuning': False,
        'final_test_access': False, 'no_automatic_followup': True,
    }
    write_json(run / 'finetune.json', manifest)
    (run / 'finetune.sha256').write_text(sha(run / 'finetune.json') + '\n')
    verify(run)
    print(json.dumps({'directory': str(run), 'tasks': 18, 'new_training_tasks': 15,
                      'validation_sequences': 16, 'planned_new_cpu_hours': 49.2,
                      'reserved_cpu_hours_ceiling': 90.0, 'final_test_access': False}))
    submit_array(run, args.dry_run)
    return manifest


def verify(run):
    from allocation_launch import validation_ids
    from allocation_report import _record, _training_errors
    run = Path(run).resolve()
    if (run / 'finetune.sha256').read_text().strip() != sha(run / 'finetune.json'):
        raise ValueError('Frozen fine-tuning manifest changed')
    manifest = json.loads((run / 'finetune.json').read_text())
    expected = {'schema': 1, 'kind': 'validation_finetune_pilot', 'threads': 4,
                'memory_gib': 16, 'seeds': list(SEEDS), 'arms': list(ARMS),
                'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
                'training_lineage_seconds': 9000, 'evaluation_seconds_per_sequence_per_seed': 60,
                'planned_new_training_cpu_hours': 30.0, 'planned_evaluation_cpu_hours': 19.2,
                'planned_new_cpu_hours': 49.2, 'reserved_cpu_hours_ceiling': 90.0,
                'historical_shared_parent_cpu_hours': 24.0, 'historical_parent_training_repeated': False,
                'checkpoint_selection': 'fixed_training_endpoints',
                'comparison': PROTOCOL['comparison'], 'learning_rate_tuning': False,
                'final_test_access': False, 'no_automatic_followup': True}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Fine-tuning plan differs from the frozen validation-only budgets')
    for relative, digest in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen artifact changed: {relative}')
    for name in NEW_FILES:
        if f'bundle/folding_policy/{name}' not in manifest['hashes']:
            raise ValueError(f'Frozen implementation digest is missing: {name}')
    validate_protocol(json.loads((run / 'bundle/folding_policy/finetune_protocol.json').read_text()))
    ids = validation_ids(json.loads((run / 'data.json').read_text()))
    if manifest.get('validation_ids') != ids or 'data.json' not in manifest['hashes']:
        raise ValueError('Frozen validation selection changed')
    provenance = run / 'source_receipts/allocation.json'
    if sha(provenance) != manifest['source_allocation_sha256']:
        raise ValueError('Source allocation receipt changed')
    source_manifest = json.loads(provenance.read_text())
    if (run / 'source_receipts/allocation.sha256').read_text().strip() != sha(provenance):
        raise ValueError('Source allocation checksum differs from receipt')
    if (source_manifest['hashes']['data.json'] != manifest['hashes']['data.json']
            or source_manifest['software_at_freeze'] != manifest['software_at_freeze']
            or source_manifest.get('threads') != 4
            or source_manifest.get('validation_ids') != ids):
        raise ValueError('Dataset, hardware or software differs from source training')
    overlay = {f'bundle/folding_policy/{name}' for name in NEW_FILES}
    for relative, digest in source_manifest['hashes'].items():
        if relative.startswith('bundle/') and relative not in overlay and manifest['hashes'].get(relative) != digest:
            raise ValueError(f'Source model or environment code changed: {relative}')
    parents = {}
    for seed in SEEDS:
        folder = run / 'source_receipts/tasks' / f'patternboost_seed{seed}'
        trained = json.loads((folder / 'training/result.json').read_text())
        launched = json.loads((folder / 'launch.json').read_text())
        source_task = next(row for row in source_manifest['tasks']
                           if row['arm'] == 'patternboost' and row['seed'] == seed)
        if (trained.get('status') != 'complete' or launched.get('status') != 'complete'
                or _training_errors(source_task, trained, source_manifest)
                or trained.get('threads') != 4 or trained.get('training_seconds') != 7200
                or trained.get('final_test_enabled') is not False
                or trained.get('selection', {}).get('selected') is not False
                or launched.get('software') != manifest['software_at_freeze']):
            raise ValueError('Frozen parent training receipt is incomplete or changed')
        parent = _record(trained, 'stage_7200.pt')
        if (parent.get('training_seconds') != 7200 or parent.get('selected') is not False
                or not parent.get('state_sha256') or not parent.get('optimizer_sha256')
                or manifest['hashes'].get(f'parents/patternboost_seed{seed}.pt') != parent['file_sha256']):
            raise ValueError('Parent checkpoint digest differs from its training receipt')
        parents[seed] = parent
    if manifest.get('tasks') != tasks(parents):
        raise ValueError('Frozen fine-tuning task plan or parent provenance changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('directory')
    prepare_parser.add_argument('--source-allocation', required=True)
    prepare_parser.add_argument('--python', default=sys.executable)
    prepare_parser.add_argument('--dry-run', action='store_true')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('directory')
    run_parser.add_argument('--task', required=True, type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        from finetune_worker import execute
        execute(args)


if __name__ == '__main__':
    main()
