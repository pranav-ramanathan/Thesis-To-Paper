"""Apocrita resource screen; disposable states, never a learning comparison.

Each cell has a fresh process so allocator history does not pollute its RSS.
The outer launch guardian owns the whole process group and the 50-minute cutoff.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from campaign import save, verify

HERE = Path(__file__).resolve().parent
REPRESENTATIVES = ('3d4', '3d6', '3d8')
COUNTS = (1, 2, 4, 8)
# Per-cell wall caps include Python/library startup. Sized from the first EHC
# pilot; total RL caps plus 5-second kill grace fit the 50-minute application cap.
RL_BUDGETS = {
    1: {'3d4':160, '3d6':400, '3d8':230},
    2: {'3d4':130, '3d6':250, '3d8':170},
    4: {'3d4':110, '3d6':200, '3d8':145},
    8: {'3d4':100, '3d6':180, '3d8':130},
}


def cell_budget(kind, seq_id, threads):
    if kind=='rl':
        return RL_BUDGETS[threads][seq_id]
    if kind=='compatibility':
        return 120
    return 125 if seq_id in REPRESENTATIVES else 30


def peak_gib():
    # Linux reports KiB; macOS reports bytes. Cluster measurements use Linux.
    divisor = 1024**3 if sys.platform == 'darwin' else 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def populate(agent, capacity, stopped=lambda: False):
    """Valid transitions with distinct owned arrays, matching resident replay size.

    Repeated transition values are intentional; sharing 128 array objects would
    drastically understate full-buffer RAM and checkpoint storage.
    """
    from worker import np_valid
    import random
    samples = []
    state = agent.env.reset()
    while len(samples) < 128:
        if stopped():
            return False
        valid = np_valid(agent.env)
        if not valid:
            state = agent.env.reset()
            continue
        action = random.choice(valid)
        new, reward, done, _ = agent.env.step(action)
        samples.append((state, action, reward, new, done))
        state = agent.env.reset() if done else new
    for i in range(capacity):
        if i % 1024 == 0 and stopped():
            return False
        s, a, r, ns, d = samples[i % len(samples)]
        agent.memory.memory.append((s.copy(), a, r, ns.copy(), d))
    agent.memory.priorities[:] = 1
    agent.memory.pos = 0
    return True


def rollout(agent, epsilon, *, training, recorder, stopped, store=True):
    from geometry import contacts
    started = time.monotonic()
    was_training = agent.policy_net.training
    agent.policy_net.train(training)
    steps = 0
    try:
        state = agent.env.reset()
        for _ in range(2 * agent.env.length):
            if stopped():
                return None
            action = agent.select_action(state, epsilon)
            new, reward, done, _ = agent.env.step(action)
            if training and store:
                agent.store_transition(state, action, reward, new, done)
            state = new
            steps += 1
            if done:
                break
        complete = len(agent.env.positions) == agent.env.length
        if complete:
            fold = [list(map(int, p)) for p in agent.env.positions]
            value = contacts(agent.env.sequence, fold, agent.env.radius)
            if value != agent.env._calculate_hh_bonds():
                raise RuntimeError('Independent contact check failed')
            recorder.emit(dict(type='witness', seq=agent.env.sequence, positions=fold,
                               contacts=value, engineering_only=True))
        return dict(rollout_s=time.monotonic()-started, steps=steps, complete=complete)
    finally:
        agent.policy_net.train(was_training)


def cell(args):
    # Only reached on a compute node via run(), never by report generation.
    import signal
    import torch
    from worker import Recorder, checkpoint, make_agent, seed_all
    from hp_solver import solve
    interrupted = False

    def stop_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop_signal)
    stopped = lambda: interrupted or time.time() >= args.end_epoch
    started = time.monotonic()
    cpu_start = time.process_time()
    protocol = json.loads((HERE/'protocol.json').read_text())
    config = protocol['configurations'][args.seq_id]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(80610)
    recorder = Recorder(args.out, config['sequence'])
    result = dict(kind=args.kind, seq_id=args.seq_id, threads=args.threads,
                  status='partial', phase='initialization', scientific_results=False,
                  measurement_profile='cp_v1' if args.kind=='cp' else 'integrated_v2',
                  profiling_overhead_s=0,
                  warmups_s=[], updates_s=[], episodes=[], rollouts=[], evaluations=[])

    def persist():
        profiling_started = time.monotonic()
        result.update(peak_rss_gib=peak_gib(), elapsed_s=time.monotonic()-started,
                      process_cpu_s=time.process_time()-cpu_start)
        save(args.out/'measurement.json', result)
        elapsed = time.monotonic()-profiling_started
        result['profiling_overhead_s'] += elapsed
        return elapsed

    try:
        persist()
        if args.kind == 'cp':
            from geometry import contacts
            result.update(full=[], repair=[])
            for repetition in range(5 if args.seq_id in REPRESENTATIVES else 1):
                for mode, seconds in (('full', 15), ('repair', 5)):
                    if stopped():
                        return
                    seq = config['sequence']
                    radius = len(seq)//2
                    # A valid anchored two-row snake inside the common cube.
                    incumbent = [[x, 0, 0] for x in range(radius+1)]
                    incumbent += [[x, 1, 0] for x in range(radius, -1, -1)]
                    incumbent = incumbent[:len(seq)]
                    result['phase'] = f'cp_{mode}'; persist()
                    t0 = time.monotonic()
                    fold, status = solve(seq, seconds=min(seconds, max(0, args.end_epoch-time.time())),
                                         threads=args.threads, seed=repetition, emit=recorder.emit,
                                         incumbent=incumbent if mode == 'repair' else None,
                                         movable=set(range(len(seq)-10, len(seq))) if mode == 'repair' else None,
                                         stopped=stopped)
                    result[mode].append(dict(seed=repetition, status=status, wall_s=time.monotonic()-t0,
                                             contacts=contacts(seq, fold, radius) if fold is not None else None,
                                             returned_witness=fold is not None))
                    persist()
            if not stopped():
                result['status'] = 'complete'
        else:
            result['phase'] = 'model_initialization'; persist()
            agent = make_agent(config)
            result['phase'] = 'replay_population'; persist()
            t0 = time.monotonic()
            if not populate(agent, config['memory_size'], stopped):
                return
            result.update(replay_entries=len(agent.memory), replay_fill_s=time.monotonic()-t0,
                          replay_storage='distinct FP32 state and next-state arrays; repeated valid values')
            persist()
            repetitions = 5 if args.kind == 'rl' else 1
            for _ in range(2 if args.kind == 'rl' else 1):
                if stopped():
                    return
                result['phase'] = 'warmup_update'; persist()
                t0 = time.monotonic()
                agent.update()
                result['warmups_s'].append(time.monotonic()-t0)
                persist()
            for repetition in range(repetitions):
                if stopped():
                    return
                result['phase'] = 'training_rollout_epsilon_0.25'; persist()
                t0 = time.monotonic()
                row = rollout(agent, .25, training=True, recorder=recorder, stopped=stopped)
                if row is None or stopped():
                    return
                result['phase'] = 'episode_update'; phase_write_s = persist()
                u0 = time.monotonic()
                loss = agent.update()
                update_s = time.monotonic()-u0
                result['updates_s'].append(update_s)
                result['last_loss'] = loss
                row.update(epsilon=.25, repetition=repetition, update_s=update_s)
                recorder.emit(dict(type='episode', engineering_only=True, loss=loss, **row))
                row['total_s'] = time.monotonic()-t0-phase_write_s
                row['profiling_write_s'] = phase_write_s
                result['episodes'].append(row)
                persist()
                # Isolated exploratory/greedy rollout timings. These do not write
                # replay or trigger more updates; full operations are timed above.
                for epsilon in (1.0, 0.0):
                    result['phase'] = f'rollout_only_epsilon_{epsilon}'; persist()
                    extra = rollout(agent, epsilon, training=True, store=False,
                                    recorder=recorder, stopped=stopped)
                    if extra is None or stopped():
                        return
                    result['rollouts'].append(dict(epsilon=epsilon, repetition=repetition, **extra))
                    persist()
                result['phase'] = 'greedy_evaluation'; persist()
                row = rollout(agent, 0.0, training=False, recorder=recorder, stopped=stopped)
                if row is None:
                    return
                result['evaluations'].append(row)
                persist()
            if stopped():
                return
            result['phase'] = 'checkpoint_write'; persist()
            checkpoint(agent, len(result['episodes']), recorder, 3600, 0)
            record = json.loads((args.out/'events.jsonl').read_text().splitlines()[-1])
            result['checkpoint'] = dict(bytes=(args.out/'checkpoint.pt').stat().st_size,
                                        write_s=record['write_s'])
            # Test storage, retain sizes/timings, free pilot checkpoints immediately.
            (args.out/'checkpoint.pt').unlink()
            result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if result['status']=='partial' and stopped():
            result['stop_reason'] = 'signal' if interrupted else 'cell_deadline'
        persist()
        recorder.handle.close()


def recommend(rows, *, compatibility_required=True, pilot_arm='rl'):
    """Choose fewest cores within 10% of fastest normalized representative rate."""
    timings = {}
    for row in rows:
        if row.get('kind') != 'rl' or row.get('status') != 'complete':
            continue
        values = [e['total_s'] for e in row['episodes'] if e['epsilon'] == .25]
        if len(values) == 5:
            timings[row['seq_id'], row['threads']] = statistics.median(values)
    counts = [c for c in COUNTS if all((s, c) in timings for s in REPRESENTATIVES)]
    scores = {c: statistics.geometric_mean([min(timings[s, k] for k in counts)/timings[s, c]
                                            for s in REPRESENTATIVES]) for c in counts}
    selected = min((c for c in counts if scores[c] >= .9*max(scores.values())), default=None)
    compatibility = {r['seq_id'] for r in rows if r.get('kind') == 'compatibility'
                     and r.get('status') == 'complete' and r.get('threads') == selected}
    cp_cells = {(r['seq_id'], r['threads']) for r in rows
                if r.get('kind') == 'cp' and r.get('status') == 'complete' and r['seq_id'] in REPRESENTATIVES}
    cp_other = {r['seq_id'] for r in rows if r.get('kind')=='cp'
                and r.get('status')=='complete' and r['seq_id'] not in REPRESENTATIVES}
    checked = selected is not None and (not compatibility_required or len(compatibility) == 5)
    rss = [r['peak_rss_gib'] for r in rows if r.get('peak_rss_gib') is not None]
    return dict(pilot_arm=pilot_arm, recommended_common_cores=selected, normalized_rates=scores,
                cp_candidate_workers=8 if len(cp_cells)==12 else None,
                provisional_ram_gib=max(4, math.ceil(1.5*max(rss))) if rss else None,
                memory_headroom_factor=1.5,
                all_remaining_configurations_checked=len(cp_other)==5 if pilot_arm=='cp_sat' else len(compatibility)==5,
                all_thread_counts_measured=len(counts)==4,
                all_cp_cells_complete=len(cp_cells)==12,
                ready_to_select_budget=checked and len(counts)==4 if pilot_arm=='rl' else False,
                limitations='Short-run RSS is not a long-run upper bound, particularly for CP-SAT. '
                            'Core selection uses RL throughput; short CP outcomes do not establish solver scaling.')


def report(task_dir):
    rows = []
    for directory in sorted((task_dir/'cells').glob('*')):
        path = directory/'measurement.json'
        row = json.loads(path.read_text()) if path.exists() else dict(status='missing')
        launched = directory/'cell_launch.json'
        if launched.exists():
            info = json.loads(launched.read_text())
            row.update(kind=info['kind'], seq_id=info['seq_id'], threads=info['threads'])
            if info.get('exit_code') != 0 or info.get('timed_out'):
                row['status'] = 'partial_or_failed'
            if row.get('error'):
                row['failure_reason'] = row['error']
            elif info.get('timed_out'):
                row['failure_reason'] = 'cell_deadline'
            elif info.get('exit_code') not in (0, None):
                row['failure_reason'] = f"process_exit_{info['exit_code']}"
            elif row['status']!='complete':
                row['failure_reason'] = row.get('stop_reason') or 'unfinished_measurement'
            else:
                row['failure_reason'] = None
            row['sampled_peak_rss_gib'] = info.get('sampled_peak_rss_gib')
            row['peak_rss_gib'] = max(row.get('peak_rss_gib', 0), info.get('sampled_peak_rss_gib', 0))
        rows.append(row)
    scope_path = task_dir/'resource_scope.json'
    pilot_arm = json.loads(scope_path.read_text())['pilot_arm'] if scope_path.exists() else (
        'cp_sat' if rows and all(r.get('kind')=='cp' for r in rows) else 'rl')
    recommendation = recommend(rows, pilot_arm=pilot_arm)
    from campaign import file_hash
    data = dict(report_schema_version=2, analyser_sha256=file_hash(Path(__file__)),
                pilot_arm=pilot_arm, scientific_results=False, measurements=rows, recommendation=recommendation)
    save(task_dir/'resource_report.json', data)
    core_message = (f"Candidate RL CPU request: **{recommendation['recommended_common_cores']} cores** (None means insufficient data)."
                    if pilot_arm=='rl' else f"CP-SAT tested candidate: **{recommendation['cp_candidate_workers']} workers**; "
                    'eight remains provisional. Short solver calls cannot select the best long-run worker count.')
    lines = [f'# Apocrita {pilot_arm} resource pilot', '', 'Disposable engineering measurements; no learning or method-superiority claims.', '',
             core_message,
             f"Provisional RAM request: **{recommendation['provisional_ram_gib']} GiB per job**, including 50% headroom.",
             recommendation['limitations'], '',
             '| Workload | Sequence | Cores | Status | Peak GiB | CPU / wall | Checkpoint MiB | Reason |',
             '|---|---|---:|---|---:|---:|---:|---|']
    for row in rows:
        cpu = row.get('process_cpu_s', 0)/max(.001, row.get('elapsed_s', .001))
        checkpoint_mb = f"{row['checkpoint']['bytes']/1024**2:.1f}" if row.get('checkpoint') else '—'
        reason = (row.get('failure_reason') or row.get('error') or '—').replace('|', '\\|')
        lines.append(f"| {row.get('kind')} | {row.get('seq_id')} | {row.get('threads')} | {row['status']} | "
                     f"{row.get('peak_rss_gib', 0):.2f} | {cpu:.2f} | {checkpoint_mb} | {reason} |")
    if pilot_arm=='rl':
        lines += ['', '## Saved timings, including unfinished cells', '',
                  'Each value uses only operations that finished. Partial cells retain their counts, '
                  'and remain excluded from core selection and full-run extrapolations.', '',
                  '| Sequence | Cores | Status | Updates measured | Median update seconds | '
                  'Epsilon 0.25 episodes measured | Median episode seconds |',
                  '|---|---:|---|---:|---:|---:|---:|']
        for row in rows:
            updates = row.get('updates_s', [])
            episodes = [e['total_s'] for e in row.get('episodes', []) if e['epsilon']==.25]
            update_value = f'{statistics.median(updates):.3f}' if updates else '—'
            episode_value = f'{statistics.median(episodes):.3f}' if episodes else '—'
            lines.append(f"| {row.get('seq_id')} | {row.get('threads')} | {row['status']} | "
                         f"{len(updates)} | {update_value} | {len(episodes)} | {episode_value} |")
    if pilot_arm=='rl':
        lines += ['', '## Extrapolated RL throughput at the candidate core count', '',
              'Ranges are the minimum/maximum of five measured episodes at epsilon 0.25; '
              'they are descriptive variability, not confidence intervals. Filled replay, '
              'disposable model weights and short episodes do not predict convergence. '
              'Greedy/exploratory timings are in the JSON. Periodic evaluation, saving and '
              'hybrid repairs add costs; estimates below exclude those periodic costs.', '',
              '| Sequence | Episodes / hour (range) | 200,000 episodes: hours (range) |',
              '|---|---:|---:|']
    for row in rows:
        if row.get('kind') != 'rl' or row['status'] != 'complete' or row['threads'] != recommendation['recommended_common_cores']:
            continue
        values = [e['total_s'] for e in row['episodes'] if e['epsilon'] == .25]
        if len(values) != 5:
            continue
        low, high = min(values), max(values)
        lines.append(f"| {row['seq_id']} | {3600/statistics.median(values):.0f} ({3600/high:.0f}–{3600/low:.0f}) | "
                     f"{200000*statistics.median(values)/3600:.1f} ({200000*low/3600:.1f}–{200000*high/3600:.1f}) |")
    if pilot_arm=='cp_sat':
        lines += ['', '## Short CP-SAT search and repair checks', '',
                  'Five fresh searches per representative and worker count; one check on each other string. '
                  'Missing witnesses remain missing. These caps do not predict time to optimality.', '',
                  '| Sequence | Workers | Mode | Returned / calls | Median wall seconds |',
                  '|---|---:|---|---:|---:|']
        for row in rows:
            for mode in ('full', 'repair'):
                trials = row.get(mode, [])
                if trials:
                    lines.append(f"| {row['seq_id']} | {row['threads']} | {mode} | "
                                 f"{sum(t['returned_witness'] for t in trials)}/{len(trials)} | "
                                 f"{statistics.median(t['wall_s'] for t in trials):.2f} |")
    lines += ['', '## Campaign resource scenarios', '',
              '120 primary jobs assuming the candidate core count is allocated to every method. '
              'Compare the other pilot before selecting a common allocation. Queue time and early CP proofs excluded. '
              'These scenarios do not select a scientific budget.', '',
              '| Hours / run | Allocated CPU-hours | Waves at concurrency 24 | Compute days |',
              '|---:|---:|---:|---:|']
    cores = recommendation['recommended_common_cores'] if pilot_arm=='rl' else recommendation['cp_candidate_workers']
    if cores is not None:
        for hours in (1, 2, 12, 24, 48):
            lines.append(f'| {hours} | {120*cores*hours:,} | 5 | {5*hours/24:.2f} |')
    coverage = recommendation['all_thread_counts_measured'] if pilot_arm=='rl' else recommendation['all_cp_cells_complete']
    lines += ['', f"All 1/2/4/8-core representative measurements complete: {coverage}.",
              f"All five remaining configurations checked: {recommendation['all_remaining_configurations_checked']}.",
              'Before freezing a campaign: inspect failures/timeouts, confirm all eight configurations, '
              'inspect Slurm MaxRSS/CPU efficiency in accounting.txt, choose the common elapsed budget, '
              'and check persistent storage quota. The optional pretrained encoder/controller study '
              'is not included in this memory request. Longer solver runs may need more RAM. '
              'A follow-up bounded stability test is appropriate if cells time out or RAM approaches the 32-GiB allocation.']
    (task_dir/'resource_report.md').write_text('\n'.join(lines)+'\n')
    return data


def diagnose(campaign):
    """Read existing frozen results; no numerical imports or new computation."""
    campaign = Path(campaign).resolve()
    manifest = verify(campaign)
    if manifest['mode']!='pilot':
        raise ValueError('Diagnostics expect a pilot directory')
    tasks = list((campaign/'tasks').glob('*_pilot_*'))
    if len(tasks)!=1:
        raise ValueError('Pilot task has not started yet')
    print(f"Pilot: {manifest.get('pilot_arm')}  Source revision: {manifest.get('git_revision')}")
    for directory in sorted((tasks[0]/'cells').glob('*')):
        load = lambda name: json.loads((directory/name).read_text()) if (directory/name).exists() else {}
        row, launched = load('measurement.json'), load('cell_launch.json')
        updates = row.get('updates_s', [])
        episodes = [e['total_s'] for e in row.get('episodes', []) if e['epsilon']==.25]
        median = lambda values: f'{statistics.median(values):.3f}s' if values else 'unmeasured'
        print(f"{directory.name}: status={row.get('status', 'missing')} exit={launched.get('exit_code')} "
              f"timed_out={launched.get('timed_out')} elapsed={row.get('elapsed_s', 0):.1f}s "
              f"phase={row.get('phase', 'not recorded by older pilot')}")
        print(f"  updates={len(updates)} median={median(updates)}; epsilon-.25 episodes={len(episodes)} median={median(episodes)}")
        if row.get('error'):
            print(f"  error: {row['error']}")
        log = directory/'worker.log'
        if launched.get('exit_code') not in (0, None) and not launched.get('timed_out') and log.exists():
            print('  worker log tail:')
            for line in log.read_text(errors='replace').splitlines()[-10:]:
                print('    '+line)


def run(args, protocol, recorder, *, stopped):
    root = args.out/'cells'
    root.mkdir()
    pilot_arm = 'cp_sat' if args.seq_id=='cp_sat_resources' else 'rl'
    save(args.out/'resource_scope.json', dict(pilot_arm=pilot_arm))
    kind = 'cp' if pilot_arm=='cp_sat' else 'rl'
    plan = [(kind, seq, cores) for cores in COUNTS for seq in REPRESENTATIVES]
    plan += [('cp' if pilot_arm=='cp_sat' else 'compatibility', seq, 8)
             for seq in protocol['configurations'] if seq not in REPRESENTATIVES]
    for index, (kind, seq_id, threads) in enumerate(plan):
        if stopped() or args.end_epoch-time.time() < 45:
            break
        if kind == 'compatibility':
            # Exercise other saved rows at the chosen common thread count if available.
            threads = report(args.out)['recommendation']['recommended_common_cores'] or 8
        out = root/f'{index:02d}_{kind}_{seq_id}_{threads}cores'
        out.mkdir()
        budget = cell_budget(kind, seq_id, threads)
        end_epoch = min(args.end_epoch-15, time.time()+budget)
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = str(threads)
        command = [sys.executable, str(HERE/'resource_pilot.py'), '--cell', '--kind', kind,
                   '--seq-id', seq_id, '--threads', str(threads), '--out', str(out),
                   '--end-epoch', str(end_epoch)]
        info = dict(kind=kind, seq_id=seq_id, threads=threads, budget_s=budget, sampled_peak_rss_gib=0)
        info['start_epoch'] = time.time()
        save(out/'cell_launch.json', info)
        recorder.emit(dict(type='resource_cell_start', **info))
        with (out/'worker.log').open('w') as log:
            # Inherit the outer guardian's process group so hard cutoff kills descendants.
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            sent_at = None
            while child.poll() is None:
                status = Path(f'/proc/{child.pid}/status')
                if status.exists():
                    try:
                        for line in status.read_text().splitlines():
                            if line.startswith('VmHWM:'):
                                info['sampled_peak_rss_gib'] = max(info['sampled_peak_rss_gib'], int(line.split()[1])/1024**2)
                    except FileNotFoundError:
                        pass
                if sent_at is None and (stopped() or time.time() >= end_epoch):
                    child.terminate()
                    sent_at = time.time()
                if sent_at is not None and time.time()-sent_at >= 5:
                    child.kill()
                time.sleep(.25)
            info.update(exit_code=child.wait(), timed_out=sent_at is not None,
                        elapsed_s=time.time()-info['start_epoch'])
        save(out/'cell_launch.json', info)
        # Remove any killed checkpoint write, preventing cumulative scratch growth.
        for path in out.glob('checkpoint.pt*'):
            path.unlink()
        data = report(args.out)
        recorder.emit(dict(type='resource_cell_end', **info))
    data = report(args.out)
    complete = len(data['measurements']) == len(plan) and all(r['status']=='complete' for r in data['measurements'])
    return dict(status='complete' if complete else 'partial', scientific_results=False,
                recommendation=data['recommendation'], resource_report=str(args.out/'resource_report.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', action='store_true')
    parser.add_argument('--kind', choices=('rl', 'cp', 'compatibility'))
    parser.add_argument('--seq-id')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--end-epoch', type=float)
    parser.add_argument('--report', type=Path, help='Frozen pilot directory; no numerical imports')
    parser.add_argument('--diagnose', type=Path, help='Read existing pilot timings and failure details; no numerical imports')
    args = parser.parse_args()
    if args.diagnose:
        diagnose(args.diagnose)
    elif args.report:
        manifest = verify(args.report.resolve())
        if manifest['mode'] != 'pilot':
            parser.error('--report expects a pilot directory')
        tasks = list((args.report/'tasks').glob('*_pilot_*'))
        if len(tasks) != 1:
            parser.error('Pilot task has not started yet')
        report(tasks[0])
        print(tasks[0]/'resource_report.md')
    elif args.cell and all(v is not None for v in (args.kind, args.seq_id, args.threads, args.out, args.end_epoch)):
        cell(args)
    else:
        parser.error('Use --report PILOT_DIR, or the internal --cell arguments')
