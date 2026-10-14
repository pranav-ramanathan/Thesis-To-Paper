"""Freeze and submit the final standalone-policy comparison, with per-seed dependencies."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from dataset import make_dataset, validate_dataset
from launch import sha, software, write_json

HERE = Path(__file__).resolve().parent
BRANCHES = ('patternboost', 'grpo', 'dr_grpo', 'reinforce', 'dpo')
SCRATCH = ('scratch_grpo', 'scratch_dr_grpo', 'scratch_reinforce')
BASELINES = ('cp_sat', 'greedy')
ARMS = BRANCHES + SCRATCH + BASELINES
SEEDS = (0, 1, 2, 3, 4)
CONTRASTS = ([(arm, 'patternboost') for arm in BRANCHES[1:]]
             + [(arm, 'scratch_' + arm) for arm in ('grpo', 'dr_grpo', 'reinforce')]
             + [(arm, reference) for arm in BRANCHES + SCRATCH for reference in BASELINES])
PROTOCOL = {
    'schema': 1, 'stage': 'final-frozen-policy-comparison', 'arms': list(ARMS),
    'seeds': list(SEEDS), 'threads': 4, 'memory_gib': 16,
    'model': {'width': 64, 'heads': 4, 'layers': 2, 'dropout': 0.0},
    'dataset_seed': 20261013, 'parent_training_seconds': 7200,
    'additional_training_seconds': 1800, 'training_lineage_seconds': 9000,
    'parent_learning_rate': 0.0003, 'endpoint_learning_rate': 0.00003,
    'evaluation_seconds_per_sequence_per_seed': 600,
    'inference_timepoints_s': [60, 300, 600], 'attempt_checkpoints': [1024],
    'array_concurrency': 5, 'maximum_concurrent_endpoint_jobs': 50,
    'campaign_deadline_days': 10, 'slurm_minutes': {'parent': 180, 'branch': 960, 'scratch': 1080, 'baseline': 960},
    'evaluation_splits': ['test', 'benchmark'], 'checkpoint_selection': 'fixed_training_endpoints',
    'final_test_enabled': True, 'automatic_extension': False, 'fresh_training': True,
    'inference_solver_repair': False,
    'contrasts': [list(pair) for pair in CONTRASTS],
}


def task_directory(run, task):
    return Path(run) / 'evaluations' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}_{task['split']}"


def training_directory(run, task):
    return Path(run) / 'training' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}"


def job_directory(run, task):
    return Path(run) / 'jobs' / f"{task['task']:03d}_{task['group']}_seed{task['seed']}"


def build_tasks(data):
    jobs, evaluations = [], []
    for seed in SEEDS:
        jobs.append(dict(task=len(jobs), group='parent', arm='parent', seed=seed, index=seed,
                         job_kind='parent', method=None, training_seconds=7200, parent_task=None,
                         evaluation_tasks=[], slurm_minutes=180))
    for arm in ARMS:
        method = arm if arm in BASELINES else 'model'
        seconds = 1800 if arm in BRANCHES else 9000 if arm in SCRATCH else 0
        for seed in SEEDS:
            job_id = len(jobs)
            ids = []
            for split in ('test', 'benchmark'):
                ids.append(len(evaluations))
                evaluations.append(dict(task=len(evaluations), arm=arm, method=method, seed=seed,
                                        split=split, sequence_ids=[row['id'] for row in data[split]],
                                        seconds=600, training_task=job_id if method == 'model' else None,
                                        job_task=job_id, training_seconds=9000 if method == 'model' else 0))
            jobs.append(dict(task=job_id, group=arm, arm=arm, seed=seed, index=seed,
                             job_kind='endpoint', method=method, training_seconds=seconds,
                             parent_task=seed if arm in BRANCHES else None, evaluation_tasks=ids,
                             slurm_minutes=1080 if arm in SCRATCH else 960))
    return jobs, evaluations


def find_job(manifest, group, index):
    matches = [t for t in manifest['tasks'] if t['group'] == group and t['index'] == index]
    if len(matches) != 1:
        raise ValueError('Job is outside the frozen final campaign')
    return matches[0]


def verify(run):
    run = Path(run).resolve()
    if (run / 'campaign.sha256').read_text().strip() != sha(run / 'campaign.json'):
        raise ValueError('Frozen final campaign manifest changed')
    manifest = json.loads((run / 'campaign.json').read_text())
    if manifest.get('kind') != 'final_policy_comparison' or manifest.get('protocol') != PROTOCOL:
        raise ValueError('Final campaign protocol changed')
    for relative, expected in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Frozen campaign artifact changed: {relative}')
    data = json.loads((run / 'data.json').read_text())
    validate_dataset(data)
    expected_data = make_dataset(PROTOCOL['dataset_seed'], run / 'bundle/cluster/protocol.json')
    if data != expected_data:
        raise ValueError('Dataset differs from the original frozen train/validation/test partition')
    jobs, evaluations = build_tasks(data)
    if manifest.get('tasks') != jobs or manifest.get('evaluation_tasks') != evaluations:
        raise ValueError('Final task matrix changed')
    if manifest.get('held_out_training_lengths') != data['held_out_training_lengths']:
        raise ValueError('Final length-transfer strata changed')
    for key in ('threads', 'memory_gib', 'arms', 'seeds', 'training_lineage_seconds',
                'inference_timepoints_s', 'attempt_checkpoints', 'contrasts'):
        if manifest.get(key) != PROTOCOL[key]:
            raise ValueError(f'Final manifest {key} changed')
    created = datetime.fromisoformat(manifest['created_utc'])
    if datetime.fromisoformat(manifest['deadline_utc']) != created + timedelta(days=10):
        raise ValueError('Final deadline differs from the frozen ten-day ceiling')
    expected = dict(planned_training_cpu_hours=240.0, planned_inference_cpu_hours=8800 / 3,
                    scientific_training_cpu_hours=400.0, reserved_cpu_hours_ceiling=3380.0,
                    final_test_access=True, checkpoint_selection='fixed_training_endpoints',
                    inference_solver_repair=False, automatic_extension=False)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Final comparison accounting or access rules changed')
    return manifest


def verified_training(run, task, manifest=None):
    """Check a full endpoint and its parent chain before inference or reporting."""
    import torch
    from final_train import validate_final_checkpoint
    run = Path(run).resolve()
    manifest = manifest or verify(run)
    launched = json.loads((job_directory(run, task) / 'launch.json').read_text())
    if (launched.get('task') != task or launched.get('software') != manifest['software_at_freeze']
            or launched.get('status') not in ('running', 'complete', 'failed')):
        raise ValueError('Training job provenance differs from the frozen campaign')
    folder = training_directory(run, task)
    receipt = json.loads((folder / 'result.json').read_text())
    checkpoint = folder / 'final.pt'
    file_hash = sha(checkpoint)
    parent = None
    if task['parent_task'] is not None:
        parent_task = manifest['tasks'][task['parent_task']]
        if parent_task['job_kind'] != 'parent' or parent_task['seed'] != task['seed']:
            raise ValueError('Training parent belongs to a different seed or stage')
        parent = verified_training(run, parent_task, manifest)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_final_checkpoint(saved, manifest['hashes']['data.json'], seed=task['seed'],
                              arm=task['arm'], parent_sha=parent['file_sha256'] if parent else None)
    lineage = 7200 if task['job_kind'] == 'parent' else 9000
    expected = dict(status='complete', arm=task['arm'], seed=task['seed'],
                    threads=manifest['threads'], dataset_sha256=manifest['hashes']['data.json'],
                    training_seconds=lineage, segment_training_seconds=task['training_seconds'],
                    checkpoint_sha256=file_hash, training_data_split='train',
                    final_test_enabled=False, test_evaluated=False)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Training receipt differs from its frozen complete endpoint')
    elapsed = receipt.get('compute_elapsed_s')
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < task['training_seconds']:
        raise ValueError('Training receipt lacks full elapsed-budget coverage')
    if (saved.get('model_config') != PROTOCOL['model'] or saved.get('training_seconds') != lineage
            or saved.get('segment_training_seconds') != task['training_seconds']
            or saved.get('threads') != manifest['threads']
            or receipt.get('final_state_sha256') != saved.get('state_sha256')):
        raise ValueError('Checkpoint architecture, resources, lineage or weights differ from receipt')
    if parent and (saved.get('parent_state_sha256') != parent['state_sha256']
                   or saved.get('initial_state_sha256') != parent['state_sha256']):
        raise ValueError('Branch did not begin from its same-seed PatternBoost parent')
    keys = ('schema', 'kind', 'role', 'arm', 'seed', 'threads', 'stage', 'condition', 'model_config',
            'dataset_sha256', 'training_seconds', 'segment_training_seconds', 'parent_file_sha256',
            'parent_state_sha256', 'initial_state_sha256', 'state_sha256', 'selection',
            'training_data_split', 'final_test_enabled', 'test_evaluated', 'final_campaign', 'status')
    metadata = {key: saved.get(key) for key in keys}
    for key in ('kind', 'role', 'final_campaign', 'parent_file_sha256', 'parent_state_sha256',
                'initial_state_sha256', 'selection'):
        if receipt.get(key) != saved.get(key):
            raise ValueError(f'Training receipt provenance differs from checkpoint: {key}')
    return dict(checkpoint=str(checkpoint), file_sha256=file_hash, state_sha256=saved['state_sha256'],
                metadata=metadata, training_seconds=lineage, receipt=receipt)


def verified_endpoint(run, task):
    manifest = verify(run)
    if task not in manifest['evaluation_tasks'] or task['method'] != 'model':
        raise ValueError('Endpoint lookup requires a frozen model evaluation task')
    training_task = manifest['tasks'][task['training_task']]
    if training_task['arm'] != task['arm'] or training_task['seed'] != task['seed']:
        raise ValueError('Evaluation does not match its training task')
    return verified_training(run, training_task, manifest)


def submission_commands(run, parent_id=None):
    run = Path(run).resolve()
    commands = []
    # Parents and independent work submit first; every dependent task uses its same index.
    for group in ('parent',) + SCRATCH + BASELINES + BRANCHES:
        minutes = 180 if group == 'parent' else 1080 if group in SCRATCH else 960
        cmd = ['sbatch', '--parsable', f'--chdir={run}', f'--job-name=hp-final-{group}',
               '--array=0-4%5', f'--time={minutes}',
               f'--output={run}/logs/%A_%a.out', f'--error={run}/logs/%A_%a.err']
        if group in BRANCHES:
            cmd += [f'--dependency=aftercorr:{parent_id or "PARENT_ARRAY_ID"}', '--kill-on-invalid-dep=yes']
        if os.environ.get('HP_MAIL_USER'):
            cmd.append('--mail-user=' + os.environ['HP_MAIL_USER'])
        commands.append((group, cmd + [str(run / 'bundle/folding_policy/final.sbatch'), str(run), group]))
    return commands


def submit(run, dry_run=False):
    """Record each accepted ID immediately; never silently retry partial submission."""
    run = Path(run).resolve()
    verify(run)
    path = run / 'submission.json'
    if path.exists():
        raise ValueError('Submission was already attempted; inspect saved IDs before any further action')
    state = dict(status='prepared', dry_run=dry_run, arrays=[], automatic_extension=False)
    write_json(path, state)
    if dry_run:
        for group, command in submission_commands(run):
            print(group + ': ' + shlex.join(command))
        return state
    ids = {}
    try:
        for group, _ in submission_commands(run):
            command = dict(submission_commands(run, ids.get('parent')))[group]
            entry = dict(group=group, command=command, status='submitting')
            state['arrays'].append(entry)
            write_json(path, state)
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            entry.update(stdout=result.stdout, stderr=result.stderr)
            match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', result.stdout.strip())
            if not match:
                entry['status'] = 'submission_uncertain'
                raise ValueError('Unexpected sbatch response; check Slurm before retrying')
            ids[group] = match.group(1)
            entry.update(status='submitted', job_id=ids[group])
            write_json(run / 'job_ids.json', ids)
            (run / 'job_ids.txt').write_text(','.join(ids.values()) + '\n')
            write_json(path, state)
            print(f'Submitted {group}: {ids[group]} (five tasks)', flush=True)
        state['status'] = 'submitted'
    except Exception as exc:
        state.update(status='submission_incomplete', error=str(exc))
        write_json(path, state)
        raise RuntimeError(f'Partial or uncertain submission; inspect {path} and job_ids.txt. '
                           'Accepted jobs are preserved; no automatic resubmission occurs.') from exc
    write_json(path, state)
    return state


def prepare(args):
    run = Path(args.directory).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    if json.loads((HERE / 'final_protocol.json').read_text()) != PROTOCOL:
        raise ValueError('Final protocol file differs from the approved comparison')
    required = ('final_train.py', 'final_launch.py', 'final_worker.py', 'final_report.py',
                'final.sbatch', 'final_campaign.sh', 'final_report.sh', 'FINAL_PLAN.md')
    if any(not (HERE / name).is_file() for name in required):
        raise ValueError('Final campaign implementation is incomplete')
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError('Python executable does not exist')
    versions = software()
    if not args.dry_run and (not shutil.which('sbatch') or any(v is None for v in versions['packages'].values())):
        raise ValueError('Submit on Apocrita with the existing Torch/NumPy/OR-Tools environment')
    run.mkdir(parents=True)
    for name in ('logs', 'jobs', 'training', 'evaluations', 'bundle/folding_policy', 'bundle/cluster'):
        (run / name).mkdir(parents=True)
    for source in HERE.iterdir():
        if source.is_file() and source.suffix in {'.py', '.json', '.sh', '.sbatch', '.md', '.txt'}:
            shutil.copy2(source, run / 'bundle/folding_policy' / source.name)
    for name in ('hp_solver.py', 'geometry.py', 'protocol.json'):
        shutil.copy2(HERE.parent / 'cluster' / name, run / 'bundle/cluster' / name)
    data = make_dataset(PROTOCOL['dataset_seed'], run / 'bundle/cluster/protocol.json')
    write_json(run / 'data.json', data)
    # Exact settings are frozen with the source graph; no test-based tuning is possible here.
    config = json.loads((HERE / 'allocation_protocol.json').read_text())
    config['model'] = PROTOCOL['model']
    write_json(run / 'training_config.json', {key: config[key] for key in ('model', 'training', 'allocation')})
    jobs, evaluations = build_tasks(data)
    created = datetime.now(timezone.utc)
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    files = sorted(p for p in (run / 'bundle').rglob('*') if p.is_file())
    files += [run / 'data.json', run / 'training_config.json']
    manifest = dict(schema=1, kind='final_policy_comparison', protocol=PROTOCOL,
                    created_utc=created.isoformat(), deadline_utc=(created + timedelta(days=10)).isoformat(),
                    source_revision=git.stdout.strip() if git.returncode == 0 else None,
                    python_executable=str(python), software_at_freeze=versions,
                    hashes={str(p.relative_to(run)): sha(p) for p in files},
                    tasks=jobs, evaluation_tasks=evaluations,
                    held_out_training_lengths=data['held_out_training_lengths'],
                    planned_training_cpu_hours=240.0, planned_inference_cpu_hours=8800 / 3,
                    scientific_training_cpu_hours=400.0, reserved_cpu_hours_ceiling=3380.0,
                    final_test_access=True, checkpoint_selection='fixed_training_endpoints',
                    inference_solver_repair=False, automatic_extension=False,
                    **{key: PROTOCOL[key] for key in ('threads', 'memory_gib', 'arms', 'seeds',
                       'training_lineage_seconds', 'inference_timepoints_s', 'attempt_checkpoints', 'contrasts')})
    write_json(run / 'campaign.json', manifest)
    (run / 'campaign.sha256').write_text(sha(run / 'campaign.json') + '\n')
    verify(run)
    print(json.dumps(dict(directory=str(run), scheduler_tasks=len(jobs), evaluation_tasks=len(evaluations),
                          planned_cpu_hours=240 + 8800 / 3, deadline_utc=manifest['deadline_utc'])), flush=True)
    return submit(run, args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('directory')
    prep.add_argument('--python', default=sys.executable)
    prep.add_argument('--dry-run', action='store_true')
    work = commands.add_parser('run')
    work.add_argument('directory')
    work.add_argument('--group', required=True)
    work.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        from final_worker import run_job
        run_job(Path(args.directory), args.group, args.index)


if __name__ == '__main__':
    main()
