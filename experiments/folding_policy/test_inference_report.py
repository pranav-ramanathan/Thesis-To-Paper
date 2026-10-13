"""Budget, witness and provenance checks for the inference-only comparison."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_launch import task_directory
from inference_report import at_time, summarize, verify_row

SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
OPEN = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]]
SOURCE = {'id': 'validation_000', 'sequence': 'HHHH', 'length': 4}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(run):
    write(run / 'data.json', {'validation': [SOURCE]})
    dataset_hash = digest(run / 'data.json')
    tasks = []
    for arm in ('patternboost', 'balanced', 'more_pretrain', 'rl_scratch', 'cp_sat', 'greedy'):
        model = arm not in ('cp_sat', 'greedy')
        task = {'task': len(tasks), 'arm': arm, 'seed': 0, 'method': 'model' if model else arm,
                'seconds': 600 if model else 1050, 'checkpoint': f'checkpoints/{arm}.pt' if model else None,
                'checkpoint_sha256': None, 'state_sha256': f'weights-{arm}' if model else None,
                'training_seconds': 7200 if model else 0, 'slurm_seconds': 10800 if model else 18000}
        metadata = None
        if model:
            checkpoint = run / task['checkpoint']
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(arm.encode())
            task['checkpoint_sha256'] = digest(checkpoint)
            mode = arm if arm in ('patternboost', 'rl_scratch') else 'rl_branch'
            metadata = {'schema': 1, 'kind': 'allocation_training_state', 'seed': 0,
                        'mode': mode, 'condition': 'patternboost_rl' if mode == 'rl_branch' else mode,
                        'stage': 'patternboost' if mode == 'patternboost' else 'rl',
                        'training_seconds': 7200, 'dataset_sha256': dataset_hash,
                        'state_sha256': task['state_sha256']}
        score = 0 if arm == 'rl_scratch' else 1
        position = OPEN if score == 0 else SQUARE
        row = {**SOURCE, 'method': task['method'], 'seed': 0, 'budget_s': task['seconds'],
               'elapsed_s': task['seconds'] + .01, 'setup_s': .1,
               'attempted_folds': None if arm == 'cp_sat' else 10,
               'generated_complete_folds': None if arm == 'cp_sat' else 9,
               'best_contacts': score, 'best_positions': position,
               'curve': [{'seconds': 2, 'contacts': score, 'positions': position}],
               'solver_status': 'FEASIBLE' if arm == 'cp_sat' else None,
               'proof_within_budget': False, 'bound': 2 if arm == 'cp_sat' else None,
               'bound_scope': 'full_declared_cube' if arm == 'cp_sat' else None,
               'bound_recorded_s': 1050.001 if arm == 'cp_sat' else None}
        if model:
            row.update(checkpoint_stage=metadata['stage'], checkpoint_condition=metadata['condition'],
                       checkpoint_seed=0, checkpoint_dataset_sha256=dataset_hash)
        payload = {'schema': 1, 'kind': 'inference_comparison_evaluation', 'status': 'complete',
                   'split': 'validation', 'final_test_access': False, 'arm': arm,
                   'method': task['method'], 'seed': 0, 'threads': 4, 'ids': [SOURCE['id']],
                   'seconds_per_sequence': task['seconds'], 'dataset_file_sha256': dataset_hash,
                   'model_setup_charged_per_sequence': True,
                   'checkpoint_sha256': task['checkpoint_sha256'], 'checkpoint_metadata': metadata,
                   'results': [row]}
        write(task_directory(run, task) / 'evaluation.json', payload)
        write(task_directory(run, task) / 'launch.json', {'status': 'complete'})
        tasks.append(task)
    return {'threads': 4, 'tasks': tasks, 'validation_ids': [SOURCE['id']],
            'hashes': {'data.json': dataset_hash}, 'inference_timepoints_s': [60, 300, 600],
            'amortization_sequences': 16, 'training_lineage_seconds': 7200}


class InferenceReportTests(unittest.TestCase):
    def report(self, run, manifest):
        with patch('inference_launch.verify', return_value=manifest), contextlib.redirect_stdout(io.StringIO()):
            return summarize(run)

    def modify(self, run, manifest, arm, update):
        task = next(t for t in manifest['tasks'] if t['arm'] == arm)
        path = task_directory(run, task) / 'evaluation.json'
        payload = json.loads(path.read_text())
        update(payload)
        write(path, payload)
        return task, payload

    def test_all_methods_and_training_cost_are_aligned_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            output = self.report(run, fixture(run))
            self.assertEqual(output['complete_tasks'], 6)
            self.assertEqual(output['verified_witness_records'], 12)
            self.assertEqual(output['costs']['new_training_seconds'], 0)
            self.assertEqual(output['costs']['model_training_charge_seconds_per_sequence'], 450)
            end = [r for r in output['timepoints'] if r['comparison'] == 'amortized_end_to_end']
            self.assertTrue(all(r['charged_seconds'] == 1050 for r in end))
            model = next(r for r in end if r['arm'] == 'patternboost')
            baseline = next(r for r in end if r['arm'] == 'cp_sat')
            self.assertEqual((model['inference_seconds'], model['training_charge_seconds']), (600, 450))
            self.assertEqual((baseline['inference_seconds'], baseline['training_charge_seconds']), (1050, 0))
            self.assertIsNone(baseline['generated_complete_folds'])
            self.assertIsNone(baseline['upper_bound'])  # Recorded after its cutoff.
            for name in ('analysis.json', 'report.md', 'summary.csv', 'curves.csv'):
                self.assertTrue((run / name).is_file())

    def test_prefix_timepoints_do_not_inherit_folds_generated_later(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            output = self.report(run, fixture(run))
            model = [p for p in output['timepoints'] if p['arm'] == 'patternboost']
            self.assertIsNone(model[0]['generated_complete_folds'])
            self.assertIsNone(model[1]['attempted_folds'])
            self.assertEqual(model[2]['generated_complete_folds'], 9)
            greedy = [p for p in output['timepoints'] if p['arm'] == 'greedy']
            self.assertIsNone(greedy[2]['generated_complete_folds'])
            self.assertEqual(greedy[3]['generated_complete_folds'], 9)
            self.assertEqual(greedy[0]['actual_full_search_elapsed_s'], 1050.01)

    def test_missing_folds_and_missing_tasks_remain_missing_but_zero_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def no_folds(payload):
                payload['results'][0].update(best_contacts=None, best_positions=None,
                    curve=[], generated_complete_folds=0)
            self.modify(run, manifest, 'balanced', no_folds)
            missing = next(t for t in manifest['tasks'] if t['arm'] == 'more_pretrain')
            (task_directory(run, missing) / 'evaluation.json').unlink()
            output = self.report(run, manifest)
            rows = {r['arm']: r for r in output['summary'] if r['comparison'] == 'amortized_end_to_end'}
            self.assertEqual(rows['rl_scratch']['mean_contacts'], 0)
            self.assertIsNone(rows['balanced']['mean_contacts'])
            self.assertEqual(rows['balanced']['complete_runs'], 1)
            self.assertEqual(rows['balanced']['valid_runs'], 0)
            self.assertIsNone(rows['more_pretrain']['mean_contacts'])
            self.assertEqual(rows['more_pretrain']['recorded_runs'], 0)

    def test_running_task_progress_retained_without_entering_final_means(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def partial(payload):
                payload['status'] = 'running'
                payload['results'][0]['elapsed_s'] = 100
            self.modify(run, manifest, 'balanced', partial)
            output = self.report(run, manifest)
            points = [p for p in output['timepoints'] if p['arm'] == 'balanced']
            self.assertEqual(points[0]['contacts'], 1)
            self.assertTrue(points[0]['covered'])
            self.assertFalse(points[0]['complete'])
            self.assertIsNone(points[1]['contacts'])
            self.assertTrue(all(r['mean_contacts'] is None for r in output['summary'] if r['arm'] == 'balanced'))

    def test_early_proof_carries_forward_but_not_backward_or_from_later_witnesses(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def proof(payload):
                payload['results'][0].update(elapsed_s=120.1, solver_status='OPTIMAL',
                                            proof_within_budget=True, bound=1, bound_recorded_s=120)
                payload['results'][0]['curve'] = [
                    {'seconds': 2, 'contacts': 0, 'positions': OPEN},
                    {'seconds': 119, 'contacts': 1, 'positions': SQUARE}]
            task, payload = self.modify(run, manifest, 'cp_sat', proof)
            output = self.report(run, manifest)
            points = [p for p in output['timepoints'] if p['arm'] == 'cp_sat']
            self.assertEqual(points[0]['contacts'], 0)
            self.assertFalse(points[0]['proven_optimal'])
            self.assertIsNone(points[0]['upper_bound'])
            self.assertEqual(points[1]['contacts'], 1)
            self.assertTrue(points[1]['proven_optimal'])
            self.assertTrue(points[-1]['complete'])
            self.assertEqual(points[-1]['actual_full_search_elapsed_s'], 120.1)
            checked = verify_row(payload['results'][0], SOURCE, task)
            checked['coverage_s'] = 20
            self.assertFalse(at_time(payload['results'][0], checked, 60)['covered'])

    def test_invalid_proof_scope_and_unwitnessed_optimum_are_rejected(self):
        for changes in ({'bound_scope': 'restricted_repair'}, {'bound': 2}, {'bound_recorded_s': 1100}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(payload):
                    payload['results'][0].update(solver_status='OPTIMAL', proof_within_budget=True,
                        bound=1, bound_recorded_s=100)
                    payload['results'][0].update(changes)
                self.modify(run, manifest, 'cp_sat', corrupt)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_all_witnesses_checked_and_integer_coordinates_required(self):
        for corrupt_curve in (True, False):
            with self.subTest(curve=corrupt_curve), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(payload):
                    row = payload['results'][0]
                    if corrupt_curve:
                        row['curve'].insert(0, {'seconds': 1, 'contacts': 0, 'positions': SQUARE})
                    else:
                        row['best_positions'][0][0] = 0.0
                self.modify(run, manifest, 'patternboost', corrupt)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)
                self.assertTrue(json.loads((run / 'analysis.json').read_text())['errors'])

    def test_checkpoint_provenance_and_setup_exclusion_are_rejected(self):
        for field, value in (('checkpoint_sha256', 'wrong'), ('model_setup_charged_per_sequence', False),
                             ('dataset_file_sha256', 'wrong')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                self.modify(run, manifest, 'balanced', lambda payload: payload.update({field: value}))
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            self.modify(run, manifest, 'balanced', lambda payload: payload['checkpoint_metadata'].update(training_seconds=3600))
            with self.assertRaises(SystemExit):
                self.report(run, manifest)

    def test_complete_task_requires_every_sequence_and_elapsed_or_proof_coverage(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def partial(payload):
                    if missing:
                        payload['results'] = []
                    else:
                        payload['results'][0]['elapsed_s'] = 599
                self.modify(run, manifest, 'patternboost', partial)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_failed_checkpoint_loading_has_visible_failure_without_fabricated_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            self.modify(run, manifest, 'balanced', lambda payload: payload.update(
                status='failed', checkpoint_metadata=None, results=[], error='load failed'))
            output = self.report(run, manifest)
            task = next(t for t in output['tasks'] if t['arm'] == 'balanced')
            self.assertEqual(task['status'], 'failed')
            self.assertEqual(task['error'], 'load failed')
            self.assertEqual(task['verified_sequences'], 0)


if __name__ == '__main__':
    unittest.main()
