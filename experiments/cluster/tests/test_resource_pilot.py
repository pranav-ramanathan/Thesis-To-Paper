"""Resource-script checks with synthetic records; no local numerical workloads."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from campaign import create, save, verify
from resource_pilot import COUNTS, REPRESENTATIVES, cell, cell_budget, diagnose, populate, recommend, report, run


def measured(seq, cores, seconds):
    return dict(kind='rl', seq_id=seq, threads=cores, status='complete', peak_rss_gib=5,
                episodes=[dict(epsilon=.25, total_s=seconds) for _ in range(5)])


class ResourceTests(unittest.TestCase):
    def test_rl_budgets_include_startup_and_fit_the_one_hour_job(self):
        budgets = [cell_budget('rl', s, c) for c in COUNTS for s in REPRESENTATIVES]
        total = sum(budgets)+5*cell_budget('compatibility', '3d1', 8)
        self.assertLessEqual(total+17*5+15+45, 3000)
        self.assertEqual(cell_budget('rl', '3d6', 1), 400)
        self.assertEqual(cell_budget('compatibility', '3d5', 8), 120)

    def test_representative_cell_uses_seven_updates_and_five_complete_operation_samples(self):
        # Mock numerical operations: test control flow, never benchmark on laptop.
        fake_torch = SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None)
        agent = SimpleNamespace(memory=[None], update=MagicMock(return_value=.1))
        fake_rollout = MagicMock(side_effect=lambda *a, **k: dict(rollout_s=.001, steps=20, complete=True))

        def fake_checkpoint(agent, episode, recorder, next_evaluation, repairs):
            (recorder.out/'checkpoint.pt').write_bytes(b'mock')
            recorder.emit(dict(type='checkpoint', write_s=.001))

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), kind='rl', seq_id='3d4', threads=8, end_epoch=time.time()+60)
            with patch.dict(sys.modules, {'torch':fake_torch, 'hp_solver':SimpleNamespace(solve=MagicMock())}), \
                    patch('worker.seed_all'), patch('worker.make_agent', return_value=agent), \
                    patch('worker.checkpoint', side_effect=fake_checkpoint), patch('signal.signal'), \
                    patch('resource_pilot.populate', return_value=True), patch('resource_pilot.rollout', fake_rollout):
                cell(args)
            result = json.loads((Path(directory)/'measurement.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(agent.update.call_count, 7)
            self.assertEqual(len(result['warmups_s']), 2)
            self.assertEqual(len(result['updates_s']), 5)
            self.assertEqual(len(result['episodes']), 5)
            self.assertTrue(all(e['epsilon']==.25 for e in result['episodes']))
            self.assertEqual(len(result['rollouts']), 10)
            self.assertEqual(len(result['evaluations']), 5)
            self.assertEqual(result['checkpoint']['bytes'], 4)
            self.assertFalse((Path(directory)/'checkpoint.pt').exists())
            isolated = [c for c in fake_rollout.call_args_list if c.kwargs.get('store') is False]
            self.assertEqual(len(isolated), 10)

    def test_cp_and_rl_are_independent_frozen_pilot_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ('cp_sat', 'rl'):
                out = root/arm
                manifest = create(out, mode='pilot', pilot_arm=arm)
                self.assertEqual(manifest['pilot_arm'], arm)
                self.assertEqual(manifest['tasks'], [dict(arm='pilot', seed=0, seq_id=f'{arm}_resources')])
                self.assertEqual(manifest['run_seconds'], 3000)
                self.assertEqual(verify(out), manifest)

    def test_pilot_orchestration_never_mixes_cp_and_rl_workloads(self):
        protocol = json.loads((HERE/'protocol.json').read_text())
        for arm in ('cp_sat', 'rl'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                commands = []

                def fake_process(command, **kwargs):
                    commands.append(command)
                    get = lambda name: command[command.index(name)+1]
                    kind, seq, cores = get('--kind'), get('--seq-id'), int(get('--threads'))
                    row = measured(seq, cores, 1) if kind=='rl' else dict(
                        kind=kind, seq_id=seq, threads=cores, status='complete', peak_rss_gib=2)
                    if kind=='cp':
                        trial = dict(returned_witness=False, wall_s=15, status='UNKNOWN', contacts=None)
                        row.update(full=[trial]*5, repair=[trial]*5)
                    save(Path(get('--out'))/'measurement.json', row)
                    return SimpleNamespace(poll=lambda: 0, wait=lambda: 0)

                args = SimpleNamespace(out=Path(directory), seq_id=f'{arm}_resources', end_epoch=__import__('time').time()+3000)
                with patch('resource_pilot.subprocess.Popen', side_effect=fake_process):
                    result = run(args, protocol, MagicMock(), stopped=lambda: False)
                kinds = {c[c.index('--kind')+1] for c in commands}
                self.assertEqual(kinds, {'cp'} if arm=='cp_sat' else {'rl', 'compatibility'})
                self.assertEqual(len(commands), 17)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(json.loads((Path(directory)/'resource_report.json').read_text())['pilot_arm'], arm)

    def test_core_request_prefers_fewer_cores_within_ten_percent(self):
        rows = [measured(seq, cores, seconds) for seq in REPRESENTATIVES
                for cores, seconds in ((1, 4), (2, 2), (4, 1.05), (8, 1))]
        answer = recommend(rows)
        self.assertEqual(answer['recommended_common_cores'], 4)
        self.assertEqual(answer['provisional_ram_gib'], 8)
        self.assertFalse(answer['ready_to_select_budget'])
        rows += [dict(kind='compatibility', seq_id=s, threads=4, status='complete')
                 for s in ('3d1', '3d2', '3d3', '3d5', '3d7')]
        rows += [dict(kind='cp', seq_id=s, threads=c, status='complete')
                 for s in REPRESENTATIVES for c in COUNTS]
        self.assertTrue(recommend(rows)['ready_to_select_budget'])

    def test_missing_repetitions_and_failed_cells_cannot_select_cores(self):
        rows = [measured(seq, 8, 1) for seq in REPRESENTATIVES]
        rows[0]['episodes'].pop()
        self.assertIsNone(recommend(rows)['recommended_common_cores'])
        rows[0] = measured('3d4', 8, 1)
        rows[0]['status'] = 'partial_or_failed'
        self.assertIsNone(recommend(rows)['recommended_common_cores'])

    def test_report_rejects_a_success_record_from_a_timed_out_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root/'cells'/'00_rl_3d4_8cores'
            cell.mkdir(parents=True)
            save(cell/'measurement.json', measured('3d4', 8, 1))
            save(cell/'cell_launch.json', dict(kind='rl', seq_id='3d4', threads=8,
                                              exit_code=-9, timed_out=True, sampled_peak_rss_gib=12))
            data = report(root)
            self.assertEqual(data['measurements'][0]['status'], 'partial_or_failed')
            self.assertEqual(data['measurements'][0]['failure_reason'], 'cell_deadline')
            self.assertEqual(data['recommendation']['provisional_ram_gib'], 18)
            self.assertIsNone(data['recommendation']['recommended_common_cores'])
            self.assertTrue((root/'resource_report.md').is_file())

    def test_diagnostics_read_legacy_partial_samples_without_running_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'pilot'
            create(root, mode='pilot', pilot_arm='rl')
            cell = root/'tasks'/'000_pilot_rl_resources_seed0'/'cells'/'10_rl_3d6_8cores'
            cell.mkdir(parents=True)
            row = measured('3d6', 8, 10)
            row.update(status='partial', episodes=row['episodes'][:2], updates_s=[9, 11], elapsed_s=175)
            save(cell/'measurement.json', row)
            save(cell/'cell_launch.json', dict(kind='rl', seq_id='3d6', threads=8,
                                              exit_code=-9, timed_out=True, sampled_peak_rss_gib=11))
            before = (cell/'measurement.json').read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                diagnose(root)
            self.assertIn('timed_out=True', output.getvalue())
            self.assertIn('updates=2 median=10.000s', output.getvalue())
            self.assertIn('epsilon-.25 episodes=2 median=10.000s', output.getvalue())
            self.assertEqual((cell/'measurement.json').read_bytes(), before)
            self.assertFalse((cell/'worker.log').exists())
            data = report(cell.parent.parent)
            self.assertIsNone(data['recommendation']['recommended_common_cores'])
            self.assertIn('Saved timings, including unfinished cells', (cell.parent.parent/'resource_report.md').read_text())

    def test_exception_reason_is_distinct_from_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root/'cells'/'failed'
            cell.mkdir(parents=True)
            row = measured('3d4', 8, 1)
            row.update(status='failed', error='ValueError: invalid fold')
            save(cell/'measurement.json', row)
            save(cell/'cell_launch.json', dict(kind='rl', seq_id='3d4', threads=8,
                                              exit_code=1, timed_out=False, sampled_peak_rss_gib=5))
            data = report(root)
            self.assertEqual(data['measurements'][0]['failure_reason'], 'ValueError: invalid fold')
            self.assertIsNone(data['recommendation']['recommended_common_cores'])

    def test_full_replay_owns_distinct_state_arrays(self):
        env = SimpleNamespace(reset=lambda: [0, 1], step=lambda action: ([1, 2], 0, True, {}))
        memory = SimpleNamespace(memory=[], priorities=MagicMock(), pos=None)
        agent = SimpleNamespace(env=env, memory=memory)
        with patch('worker.np_valid', return_value=[0]):
            self.assertTrue(populate(agent, 256))
        self.assertEqual(len(memory.memory), 256)
        self.assertEqual(len({id(row[0]) for row in memory.memory}), 256)
        self.assertEqual(len({id(row[3]) for row in memory.memory}), 256)
        memory.memory[0][0][0] = 999
        self.assertEqual(memory.memory[128][0][0], 0)

    def test_stopped_population_does_not_claim_a_full_buffer(self):
        agent = SimpleNamespace(env=SimpleNamespace(reset=lambda: [0]),
                                memory=SimpleNamespace(memory=[]))
        self.assertFalse(populate(agent, 256, stopped=lambda: True))
        self.assertEqual(agent.memory.memory, [])

    def test_cluster_wrappers_submit_one_hour_and_report_without_numerical_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root/'venv'; (venv/'bin').mkdir(parents=True)
            (venv/'bin/python').symlink_to(sys.executable)
            commands = root/'commands'; commands.mkdir()
            capture = root/'submission.txt'
            sbatch = commands/'sbatch'
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HP_CAPTURE_FILE"\nprintf "45678\\n"\n')
            sbatch.chmod(0o755)
            sacct = commands/'sacct'
            sacct.write_text('#!/bin/bash\nprintf "JobID|State|MaxRSS\\n45678.0|COMPLETED|8G\\n"\n')
            sacct.chmod(0o755)
            env = dict(os.environ, HP_VENV_PATH=str(venv), HP_NODE_CONSTRAINT='verified_test_feature',
                       HP_PILOT_PARTITION='compute', HP_CAPTURE_FILE=str(capture),
                       PATH=str(commands)+os.pathsep+os.environ['PATH'])
            out = root/'pilot with spaces'
            result = subprocess.run(['bash', str(HERE/'rl-resource-test.sh'), str(out)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = capture.read_text().splitlines()
            self.assertIn('--partition=compute', args)
            self.assertFalse(any(a.startswith('--array') for a in args))
            template = Path(args[-2]).read_text()
            self.assertIn('#SBATCH --time=01:00:00', template)
            self.assertIn('#SBATCH --mem=32G', template)
            manifest = json.loads((out/'campaign.json').read_text())
            self.assertEqual(manifest['run_seconds'], 3000)
            self.assertEqual(manifest['threads'], 8)
            self.assertEqual(len(manifest['tasks']), 1)
            self.assertEqual(manifest['pilot_arm'], 'rl')
            task = out/'tasks'/'000_pilot_rl_resources_seed0'
            (task/'cells').mkdir(parents=True)
            result = subprocess.run(['bash', str(HERE/'resource-report.sh'), str(out)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((task/'resource_report.md').is_file())
            self.assertIn('45678.0', (out/'accounting.txt').read_text())
            cp_out = root/'cp pilot'
            result = subprocess.run(['bash', str(HERE/'cp-resource-test.sh'), str(cp_out)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            cp_manifest = json.loads((cp_out/'campaign.json').read_text())
            self.assertEqual(cp_manifest['pilot_arm'], 'cp_sat')
            self.assertIn('--job-name=hp-cp_sat-resources', capture.read_text().splitlines())


if __name__ == '__main__':
    unittest.main()
