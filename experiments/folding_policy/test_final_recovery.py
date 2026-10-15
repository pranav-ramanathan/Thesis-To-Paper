"""Recovery preserves failed-run evidence and uses fresh, fully budgeted cells.

Only the original frozen-campaign loader is replaced with a disposable fixture.
All geometries below are four-residue synthetic folds, never study test data;
the real witness, elapsed-budget and recovery hash checks remain active.
"""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import final_launch
import final_recovery as recovery
from final_report import summarize
from launch import sha, software


OPEN = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]]
SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
FAILURE = 'ValueError: Search returned before its budget without a timely optimality proof'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def row(source, score=1, elapsed=600.02):
    positions = copy.deepcopy(SQUARE if score else OPEN)
    return dict(source, seed=4, method='cp_sat', budget_s=600, elapsed_s=elapsed,
                setup_s=.275824, best_contacts=score, best_positions=positions,
                status='complete', generated_complete_folds=None, attempted_folds=None,
                proof_within_budget=False, solver_status='FEASIBLE', bound=2,
                bound_recorded_s=elapsed-.0001, bound_scope='full_declared_cube',
                bound_recorded_after_budget=elapsed > 600,
                curve=[dict(seconds=2, contacts=score, positions=copy.deepcopy(positions))])


def fixture(source):
    data = dict(test=[dict(id=f'test_{i:03d}', sequence='HHHH', length=4) for i in range(80)],
                benchmark=[dict(id=f'3d{i}', sequence='HHHH', length=4) for i in range(1, 9)],
                held_out_training_lengths=[])
    write(source / 'data.json', data)
    jobs, evaluations = final_launch.build_tasks(data)
    now = datetime.now(timezone.utc)
    manifest = dict(final_launch.PROTOCOL, kind='final_policy_comparison',
                    protocol=copy.deepcopy(final_launch.PROTOCOL), tasks=jobs,
                    evaluation_tasks=evaluations, python_executable=sys.executable,
                    software_at_freeze=software(), hashes={'data.json': sha(source/'data.json')},
                    created_utc=now.isoformat(), deadline_utc=(now+timedelta(days=10)).isoformat(),
                    held_out_training_lengths=[])
    write(source/'campaign.json', manifest)
    (source/'campaign.sha256').write_text(sha(source/'campaign.json')+'\n')
    task = evaluations[88]
    folder = final_launch.task_directory(source, task)
    payload = dict(schema=1, kind='final_policy_evaluation', split='test', final_test_access=True,
                   selection='fixed_training_endpoint', arm='cp_sat', method='cp_sat', seed=4,
                   threads=4, ids=task['sequence_ids'], seconds_per_sequence=600,
                   dataset_file_sha256=manifest['hashes']['data.json'],
                   model_setup_charged_per_sequence=True, checkpoint_sha256=None,
                   checkpoint_metadata=None, status='failed', error=FAILURE,
                   results=[row(r) for r in data['test'][:75]])
    write(folder/'evaluation.json', payload)
    write(folder/'launch.json', dict(task=task, software=manifest['software_at_freeze'],
                                     status='failed', error=FAILURE))
    for result in payload['results']:
        write(folder/'sequences'/f"{result['id']}.json", result)
    write(folder/'sequences/test_075.json', row(data['test'][75], elapsed=599.9213819764555))
    write(final_launch.job_directory(source,jobs[49])/'launch.json',
          dict(task=jobs[49], software=manifest['software_at_freeze'], status='failed', error=FAILURE))
    return manifest, data


def source_hashes(source):
    return {str(path.relative_to(source)): sha(path) for path in source.rglob('*') if path.is_file()}


class FinalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hp-recovery-fixture-only-')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source, self.destination = root/'source', root/'recovery'
        self.manifest, self.data = fixture(self.source)
        self.loader = patch.object(final_launch, 'verify', return_value=self.manifest)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return recovery.prepare(self.source, self.destination)

    def complete_cells(self, plan, score=0):
        for cell in plan['cells']:
            folder = self.destination/'cells'/f"{cell['task']:03d}"
            source = next(s for s in self.data[cell['split']] if s['id'] == cell['sequence_id'])
            write(folder/'result.json', row(source, score=score))
            write(folder/'launch.json', dict(cell=cell, software=self.manifest['software_at_freeze'],
                                              status='complete', result_sha256=sha(folder/'result.json')))

    def overlay(self):
        return recovery.load_recovery_overlay(self.source, self.destination, self.manifest)

    def test_plan_pins_original_evidence_and_exactly_thirteen_fresh_cells(self):
        before = source_hashes(self.source)
        plan = self.prepare()
        self.assertEqual(source_hashes(self.source), before)
        expected = [(88, f'test_{i:03d}') for i in range(75,80)]
        expected += [(89, f'3d{i}') for i in range(1,9)]
        self.assertEqual([(c['evaluation_task'], c['sequence_id']) for c in plan['cells']], expected)
        self.assertEqual(len(plan['cells']), 13)
        self.assertEqual((self.destination/'recovery.sha256').read_text().strip(),
                         sha(self.destination/'recovery.json'))
        self.assertTrue(plan['source_hashes'])
        for relative, expected_hash in plan['source_hashes'].items():
            self.assertEqual(sha(self.source/relative), expected_hash)
        orphan = self.source/'evaluations/088_cp_sat_seed4_test/sequences/test_075.json'
        self.assertIn(str(orphan.relative_to(self.source)), plan['source_hashes'])
        self.assertEqual(json.loads(orphan.read_text())['elapsed_s'], 599.9213819764555)

    def test_recovery_never_pools_the_short_run_with_the_replacement(self):
        before = source_hashes(self.source)
        plan = self.prepare()
        self.complete_cells(plan, score=0)
        overlay = self.overlay()
        test = overlay['payloads'][88]
        self.assertEqual(test['status'], 'complete')
        self.assertEqual(len(test['results']), 80)
        self.assertTrue(all(r['best_contacts'] == 1 for r in test['results'][:75]))
        self.assertTrue(all(r['best_contacts'] == 0 for r in test['results'][75:]))
        replaced = next(r for r in test['results'] if r['id']=='test_075')
        self.assertEqual(replaced['best_contacts'], 0)  # Original short search found one.
        self.assertEqual(replaced['elapsed_s'], 600.02)
        self.assertEqual(len(overlay['payloads'][89]['results']), 8)
        self.assertEqual(source_hashes(self.source), before)

    def test_original_evidence_mutation_is_rejected(self):
        self.prepare()
        path = self.source/'evaluations/088_cp_sat_seed4_test/sequences/test_075.json'
        result = json.loads(path.read_text())
        result['elapsed_s'] = 600
        write(path, result)
        with self.assertRaises(ValueError):
            recovery.verify_recovery_plan(self.destination)

    def test_frozen_recovery_plan_mutation_is_rejected(self):
        self.prepare()
        path = self.destination/'recovery.json'
        value = json.loads(path.read_text())
        value['cells'][0]['sequence_id'] = 'test_000'
        write(path,value)
        with self.assertRaises(ValueError):
            recovery.verify_recovery_plan(self.destination)

    def test_source_requires_exact_accepted_prefix_and_valid_witnesses(self):
        path = self.source/'evaluations/088_cp_sat_seed4_test/evaluation.json'
        original = json.loads(path.read_text())
        for scenario in ('missing', 'extra', 'corrupt'):
            with self.subTest(scenario=scenario):
                self.destination = Path(self.temporary.name)/f'recovery-{scenario}'
                payload = copy.deepcopy(original)
                if scenario=='missing':
                    payload['results'].pop()
                elif scenario=='extra':
                    payload['results'].append(row(self.data['test'][75]))
                else:
                    payload['results'][10]['best_positions'][3] = [0,0,0]
                write(path,payload)
                with self.assertRaises(ValueError):
                    self.prepare()
        write(path,original)

    def test_incomplete_recovery_cannot_complete_the_report_panel(self):
        plan = self.prepare()
        self.complete_cells(plan)
        (self.destination/'cells/012/result.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.overlay()

    def test_short_or_corrupt_recovery_result_is_rejected_even_with_matching_hash(self):
        plan = self.prepare()
        self.complete_cells(plan)
        result_path = self.destination/'cells/000/result.json'
        launch_path = result_path.parent/'launch.json'
        original = json.loads(result_path.read_text())
        for scenario in ('short','corrupt'):
            with self.subTest(scenario=scenario):
                changed = copy.deepcopy(original)
                if scenario=='short':
                    changed.update(elapsed_s=599.99,bound_recorded_s=599.98)
                else:
                    changed['best_positions'][3] = [0,0,0]
                write(result_path,changed)
                launched = json.loads(launch_path.read_text())
                launched['result_sha256'] = sha(result_path)
                write(launch_path,launched)
                with self.assertRaises(ValueError):
                    self.overlay()

    def test_report_overlay_writes_separate_artifacts_and_preserves_originals(self):
        before = source_hashes(self.source)
        plan = self.prepare()
        self.complete_cells(plan)
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = summarize(self.source, recovery_directory=self.destination)
        point = next(p for p in analysis['timepoints'] if
                     (p['arm'],p['seed'],p['split'],p['id'],p['comparison'],p['budget']) ==
                     ('cp_sat',4,'test','test_075','time',600))
        self.assertTrue(point['complete'])
        self.assertEqual(point['contacts'],0)
        self.assertEqual(analysis['complete_tasks'],2)
        self.assertTrue((self.destination/'report/report.md').is_file())
        self.assertEqual(source_hashes(self.source),before)

    def test_cell_runs_the_original_frozen_worker_and_refuses_a_second_attempt(self):
        self.prepare()
        before = source_hashes(self.source)
        calls = []
        def child(command, **kwargs):
            calls.append((command,kwargs))
            output = Path(command[command.index('--output')+1])
            write(output,row(self.data['test'][75],score=0))
            return types.SimpleNamespace(returncode=0)
        with patch.object(recovery.subprocess,'run',side_effect=child), \
                patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'4'}):
            recovery.run_cell(self.destination,0)
            with self.assertRaises(FileExistsError):
                recovery.run_cell(self.destination,0)
        self.assertEqual(len(calls),1)
        command = calls[0][0]
        self.assertEqual(Path(command[1]),(self.source/'bundle/folding_policy/final_worker.py').resolve())
        self.assertEqual(command[command.index('--evaluation')+1],'88')
        self.assertEqual(command[command.index('--sequence')+1],'test_075')
        saved = json.loads((self.destination/'cells/000/launch.json').read_text())
        self.assertEqual(saved['status'],'complete')
        self.assertEqual(saved['result_sha256'],sha(self.destination/'cells/000/result.json'))
        self.assertEqual(source_hashes(self.source),before)

    def test_short_fresh_cell_is_saved_as_failed_without_retry(self):
        self.prepare()
        def child(command, **kwargs):
            output = Path(command[command.index('--output')+1])
            write(output,row(self.data['test'][75],elapsed=599.99))
            return types.SimpleNamespace(returncode=0)
        with patch.object(recovery.subprocess,'run',side_effect=child) as mocked, \
                patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'4'}):
            with self.assertRaisesRegex(ValueError,'budget|coverage'):
                recovery.run_cell(self.destination,0)
        self.assertEqual(mocked.call_count,1)
        saved = json.loads((self.destination/'cells/000/launch.json').read_text())
        self.assertEqual(saved['status'],'failed')
        self.assertNotIn('result_sha256',saved)
        self.assertEqual(json.loads((self.destination/'cells/000/result.json').read_text())['elapsed_s'],599.99)

    def test_uncertain_submission_is_recorded_and_never_automatically_repeated(self):
        self.prepare()
        response = types.SimpleNamespace(returncode=0,stdout='unrecognized scheduler response',stderr='')
        with patch.object(recovery.subprocess,'run',return_value=response) as mocked, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'sbatch response'):
                recovery.submit(self.destination)
            with self.assertRaisesRegex(ValueError,'already attempted'):
                recovery.submit(self.destination)
        self.assertEqual(mocked.call_count,1)
        saved = json.loads((self.destination/'submission.json').read_text())
        self.assertEqual(saved['status'],'submission_uncertain')
        self.assertFalse((self.destination/'job_ids.txt').exists())

    def test_one_array_submission_saves_job_id_and_refuses_duplicate(self):
        self.prepare()
        response = types.SimpleNamespace(returncode=0,stdout='123456;fixture-cluster\n',stderr='')
        with patch.object(recovery.subprocess,'run',return_value=response) as mocked, \
                contextlib.redirect_stdout(io.StringIO()):
            recovery.submit(self.destination)
            with self.assertRaisesRegex(ValueError,'already attempted'):
                recovery.submit(self.destination)
        self.assertEqual(mocked.call_count,1)
        command = mocked.call_args.args[0]
        self.assertEqual(command[0],'sbatch')
        self.assertIn('--array=0-12%13',command)
        self.assertIn('--time=20',command)
        self.assertEqual((self.destination/'job_ids.txt').read_text(),'123456\n')
        saved = json.loads((self.destination/'submission.json').read_text())
        self.assertEqual(saved['status'],'submitted')


if __name__ == '__main__':
    unittest.main()
