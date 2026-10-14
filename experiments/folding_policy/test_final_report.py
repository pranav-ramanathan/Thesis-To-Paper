"""Final reporting preserves budgets, provenance, split separation and seed units."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from final_report import summarize
from finetune_followup_evaluate import prefix_snapshot

OPEN = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]]
SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def task_directory(run, task):
    return Path(run) / 'evaluations' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}_{task['split']}"


def fixture(run):
    data = {'test': [{'id': 'test_000', 'sequence': 'HHHH', 'length': 4},
                     {'id': 'test_001', 'sequence': 'HHHH', 'length': 4}],
            'benchmark': [{'id': '3d1', 'sequence': 'HHHH', 'length': 4,
                           'best_known_contacts': 1, 'reference': 'https://example.org/paper',
                           'optimum_status': 'fixture only'}],
            'held_out_training_lengths': [4]}
    write(run / 'data.json', data)
    arms, seeds = ['patternboost', 'grpo', 'scratch_grpo', 'cp_sat', 'greedy'], [0, 1]
    tasks, jobs, endpoints = [], [], {}
    for arm in arms:
        for seed in seeds:
            model = arm not in ('cp_sat', 'greedy')
            training_task = len(jobs) if model else None
            jobs.append({'task': len(jobs), 'arm': arm, 'seed': seed, 'training_seconds': 9000 if model else 0})
            metadata = {'schema': 1, 'kind': 'final_policy_training_state', 'threads': 4,
                        'seed': seed, 'stage': arm, 'condition': arm, 'arm': arm,
                        'training_seconds': 9000, 'dataset_sha256': 'dataset-hash',
                        'state_sha256': f'{arm}-{seed}'} if model else None
            if model:
                endpoints[training_task] = {'file_sha256': f'checkpoint-{arm}-{seed}',
                                            'state_sha256': metadata['state_sha256'],
                                            'metadata': metadata, 'training_seconds': 9000}
            for split in ('test', 'benchmark'):
                task = {'task': len(tasks), 'arm': arm, 'seed': seed, 'method': 'model' if model else arm,
                        'split': split, 'sequence_ids': [r['id'] for r in data[split]],
                        'seconds': 600, 'training_task': training_task}
                rows = []
                for source in data[split]:
                    score = 1 if arm in ('grpo', 'cp_sat', 'greedy') else 0
                    positions = SQUARE if score else OPEN
                    row = {**source, 'seed': seed, 'method': task['method'], 'budget_s': 600,
                           'elapsed_s': 600.01, 'setup_s': .1, 'best_contacts': score,
                           'best_positions': copy.deepcopy(positions), 'status': 'complete',
                           'generated_complete_folds': None if arm == 'cp_sat' else 2,
                           'attempted_folds': None if arm == 'cp_sat' else 2,
                           'proof_within_budget': False,
                           'solver_status': 'FEASIBLE' if arm == 'cp_sat' else None,
                           'bound': 2 if arm == 'cp_sat' else None,
                           'bound_recorded_s': 600.01 if arm == 'cp_sat' else None,
                           'bound_scope': 'full_declared_cube' if arm == 'cp_sat' else None,
                           'curve': [{'seconds': 2, 'contacts': score, 'positions': copy.deepcopy(positions)}]}
                    if model:
                        attempts = [{'attempt_index': i, 'seconds': i + 1, 'status': 'complete',
                                     'contacts': score, 'positions': copy.deepcopy(positions)} for i in (1, 2)]
                        row['curve'][0]['attempt_index'] = 1
                        row.update(attempt_checkpoints=[2], ordered_attempts=attempts,
                                   attempt_prefixes=[prefix_snapshot(attempts, 2, 600)],
                                   trapped_folds=0, interrupted_folds=0, discarded_after_budget=0,
                                   valid_fold_rate=1.0, checkpoint_stage=arm, checkpoint_condition=arm,
                                   checkpoint_seed=seed, checkpoint_dataset_sha256='dataset-hash',
                                   resource_usage={'hostname': 'test-host', 'process_cpu_s': 1200.02,
                                                   'setup_process_cpu_s': .1, 'process_cpu_per_wall': 2.0})
                    rows.append(row)
                payload = {'schema': 1, 'kind': 'final_policy_evaluation', 'status': 'complete',
                           'split': split, 'final_test_access': True, 'selection': 'fixed_training_endpoint',
                           'arm': arm, 'method': task['method'], 'seed': seed, 'threads': 4,
                           'ids': task['sequence_ids'], 'seconds_per_sequence': 600,
                           'dataset_file_sha256': 'dataset-hash', 'model_setup_charged_per_sequence': True,
                           'checkpoint_sha256': endpoints[training_task]['file_sha256'] if model else None,
                           'checkpoint_metadata': copy.deepcopy(metadata), 'results': rows}
                write(task_directory(run, task) / 'evaluation.json', payload)
                write(task_directory(run, task) / 'launch.json', {'task': task, 'status': 'complete',
                                                                'software': {'python': 'fixture-python'}})
                tasks.append(task)
    manifest = {'arms': arms, 'seeds': seeds, 'threads': 4, 'tasks': jobs,
                'software_at_freeze': {'python': 'fixture-python'},
                'evaluation_tasks': tasks, 'training_lineage_seconds': 9000,
                'inference_timepoints_s': [60, 300, 600], 'attempt_checkpoints': [2],
                'hashes': {'data.json': 'dataset-hash'},
                'contrasts': [('grpo', 'patternboost'), ('grpo', 'scratch_grpo'), ('grpo', 'cp_sat')]}
    return manifest, endpoints


class FinalReportTests(unittest.TestCase):
    def report(self, run, manifest, endpoints, endpoint_error=None):
        def endpoint(_, task):
            if endpoint_error:
                raise ValueError(endpoint_error)
            return endpoints[task['training_task']]
        module = types.SimpleNamespace(task_directory=task_directory, verify=lambda _: manifest,
                                       verified_endpoint=endpoint,
                                       training_directory=lambda root, task: Path(root) / 'training' / str(task['task']),
                                       verified_training=lambda root, task, manifest: endpoints[task['task']])
        with patch.dict(sys.modules, {'final_launch': module}), contextlib.redirect_stdout(io.StringIO()):
            return summarize(run)

    def modify(self, run, manifest, callback, arm='grpo', split='test', seed=0):
        task = next(t for t in manifest['evaluation_tasks'] if (t['arm'], t['split'], t['seed']) == (arm, split, seed))
        path = task_directory(run, task) / 'evaluation.json'
        payload = json.loads(path.read_text())
        callback(payload)
        write(path, payload)
        return task

    def test_split_separation_costs_and_seed_replications(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            result = self.report(run, manifest, endpoints)
            self.assertEqual(result['complete_tasks'], 20)
            pair = next(r for r in result['paired'] if (r['split'], r['stratum'], r['comparison'], r['budget'], r['arm'], r['reference']) == ('test', 'all', 'time', 600, 'grpo', 'patternboost'))
            self.assertEqual(pair['paired_seeds'], 2)  # Not four sequence-seed evaluations.
            self.assertEqual(pair['mean_difference'], 1)
            self.assertEqual(pair['sample_sd'], 0)
            self.assertEqual(result['costs']['scientific_model_training_lineage_allocated_cpu_hours'], 60)
            self.assertEqual(result['costs']['planned_inference_allocated_cpu_hours'], 20)
            self.assertEqual(result['costs']['amortization_scenarios'][0]['total_seconds_per_sequence'], 9600)
            public = [r for r in result['summary'] if r['split'] == 'benchmark']
            self.assertTrue(all(r['best_known_contacts'] == 1 for r in public))
            self.assertTrue(all(r['best_known_contacts'] is None for r in result['summary'] if r['split'] == 'test'))
            for name in ('report.md', 'analysis.json', 'summary.csv', 'timepoints.csv', 'paired.csv',
                         'seed_pairs.csv', 'seed_macros.csv', 'macro_summary.csv', 'curves.csv', 'throughput.csv'):
                self.assertTrue((run / name).is_file(), name)

    def test_missing_task_and_zero_scores_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            task = next(t for t in manifest['evaluation_tasks'] if (t['arm'], t['seed'], t['split']) == ('grpo', 0, 'test'))
            (task_directory(run, task) / 'evaluation.json').unlink()
            result = self.report(run, manifest, endpoints)
            self.assertEqual(result['complete_tasks'], 19)
            rows = [r for r in result['summary'] if r['split'] == 'test' and r['comparison'] == 'time' and r['budget'] == 600]
            pb = next(r for r in rows if r['arm'] == 'patternboost')
            self.assertEqual((pb['mean_contacts'], pb['valid_runs']), (0, 2))
            grpo = next(r for r in rows if r['arm'] == 'grpo')
            self.assertEqual((grpo['mean_contacts'], grpo['valid_runs'], grpo['planned_runs']), (1, 1, 2))
            macro = next(r for r in result['macro_summary'] if (r['split'], r['stratum'], r['comparison'], r['budget'], r['arm']) == ('test', 'all', 'time', 600, 'grpo'))
            self.assertEqual(macro['complete_seeds'], 1)

    def test_partial_task_progress_is_visible_but_not_final(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            self.modify(run, manifest, lambda p: p.update(status='running', results=p['results'][:1]))
            result = self.report(run, manifest, endpoints)
            point = next(r for r in result['timepoints'] if (r['split'], r['arm'], r['seed'], r['id'], r['comparison'], r['budget']) == ('test', 'grpo', 0, 'test_000', 'time', 600))
            self.assertEqual(point['contacts'], 1)
            self.assertTrue(point['covered'])
            self.assertFalse(point['complete'])
            macro = next(r for r in result['seed_macros'] if (r['split'], r['stratum'], r['arm'], r['seed'], r['comparison'], r['budget']) == ('test', 'all', 'grpo', 0, 'time', 600))
            self.assertIsNone(macro['mean_contacts'])

    def test_incomplete_valid_panel_excluded_without_imputing_missing_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            def no_fold(payload):
                payload['results'][0].update(best_contacts=None, best_positions=None, curve=[], generated_complete_folds=0)
            self.modify(run, manifest, no_fold, arm='greedy')
            result = self.report(run, manifest, endpoints)
            macro = next(r for r in result['seed_macros'] if (r['split'], r['stratum'], r['arm'], r['seed'], r['comparison'], r['budget']) == ('test', 'all', 'greedy', 0, 'time', 600))
            self.assertEqual(macro['valid_sequences'], 1)
            self.assertIsNone(macro['mean_contacts'])

    def test_later_proof_never_fills_earlier_scores_and_late_bound_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            def proof(payload):
                row = payload['results'][0]
                row.update(elapsed_s=120, solver_status='OPTIMAL', bound=1,
                           bound_recorded_s=120, proof_within_budget=True,
                           curve=[{'seconds': 2, 'contacts': 0, 'positions': OPEN},
                                  {'seconds': 119, 'contacts': 1, 'positions': SQUARE}])
            self.modify(run, manifest, proof, arm='cp_sat')
            result = self.report(run, manifest, endpoints)
            points = [p for p in result['timepoints'] if (p['split'], p['arm'], p['seed'], p['id']) == ('test', 'cp_sat', 0, 'test_000')]
            self.assertEqual([p['contacts'] for p in points], [0, 1, 1])
            self.assertFalse(points[0]['proven_optimal'])
            self.assertTrue(points[-1]['proven_optimal'])
            late = next(p for p in result['timepoints'] if (p['split'], p['arm'], p['seed'], p['id'], p['budget']) == ('test', 'cp_sat', 1, 'test_000', 600))
            self.assertIsNone(late['upper_bound'])

    def test_corrupt_metadata_endpoint_and_every_prefix_witness_are_rejected(self):
        changes = [lambda p: p.update(checkpoint_sha256='wrong'),
                   lambda p: p['checkpoint_metadata'].update(training_seconds=8000),
                   lambda p: p.update(split='validation'),
                   lambda p: p['results'][0]['ordered_attempts'][1]['positions'][0].__setitem__(0, .0),
                   lambda p: p['results'][0]['curve'][0].update(seconds=601),
                   lambda p: p.update(results=p['results'][:1])]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest, endpoints = fixture(run)
                self.modify(run, manifest, change)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest, endpoints)
                result = json.loads((run / 'analysis.json').read_text())
                self.assertEqual(result['complete_tasks'], 19)
                self.assertTrue(result['errors'])
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            with self.assertRaises(SystemExit):
                self.report(run, manifest, endpoints, endpoint_error='Training lineage is incomplete')

    def test_fixed_attempt_prefixes_cannot_use_later_improvements(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            def later(payload):
                row = payload['results'][0]
                for attempt in row['ordered_attempts']:
                    attempt.update(contacts=0, positions=copy.deepcopy(OPEN))
                row['attempt_prefixes'] = [prefix_snapshot(row['ordered_attempts'], 2, 600)]
                row.update(attempted_folds=3, generated_complete_folds=3,
                           curve=[{'seconds': 2, 'contacts': 0, 'positions': copy.deepcopy(OPEN), 'attempt_index': 1},
                                  {'seconds': 100, 'contacts': 1, 'positions': copy.deepcopy(SQUARE), 'attempt_index': 3}])
            self.modify(run, manifest, later)
            result = self.report(run, manifest, endpoints)
            points = [r for r in result['timepoints'] if (r['split'], r['arm'], r['seed'], r['id']) == ('test', 'grpo', 0, 'test_000')]
            self.assertEqual([p['contacts'] for p in points], [0, 1, 1, 0])

    def test_physical_training_counts_shared_parent_once_and_excludes_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            parent = {'task': len(manifest['tasks']), 'arm': 'parent', 'seed': 0,
                      'training_seconds': 7200}
            manifest['tasks'].append(parent)
            for task_id, seconds in ((0, 9001), (parent['task'], 7201)):
                receipt = {'status': 'complete', 'compute_elapsed_s': seconds, 'checkpoint_io_s': 2}
                write(run / 'training' / str(task_id) / 'result.json', receipt)
                endpoints.setdefault(task_id, {})['receipt'] = receipt
            write(run / 'training/1/result.json', {'status': 'failed', 'compute_elapsed_s': 10})
            result = self.report(run, manifest, endpoints)
            self.assertAlmostEqual(result['costs']['verified_complete_physical_training_allocated_cpu_hours'],
                                   (9001 + 7201) * 4 / 3600)
            self.assertEqual(result['costs']['scientific_model_training_lineage_allocated_cpu_hours'], 60)
            self.assertEqual(len(result['training']), 7)
            failed = next(row for row in result['training'] if row['task'] == 1)
            self.assertEqual(failed['status'], 'failed')
            self.assertIsNone(failed['verified_complete_seconds'])

    def test_interrupted_training_is_incomplete_not_a_verification_error(self):
        for status in ('interrupted', 'insufficient_training'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest, endpoints = fixture(run)
                write(run / 'training/0/result.json', {'status': status, 'compute_elapsed_s': 100,
                                                      'error': 'Budget did not complete'})
                result = self.report(run, manifest, endpoints)
                row = next(row for row in result['training'] if row['task'] == 0)
                self.assertEqual(row['status'], status)
                self.assertIsNone(row['verified_complete_seconds'])
                self.assertEqual(result['errors'], [])

    def test_launch_task_software_and_completion_must_match(self):
        for field, value in (('software', {'python': 'wrong-python'}), ('status', 'running'),
                             ('task', {'task': 999})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest, endpoints = fixture(run)
                task = manifest['evaluation_tasks'][0]
                path = task_directory(run, task) / 'launch.json'
                launch = json.loads(path.read_text())
                launch[field] = value
                write(path, launch)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest, endpoints)
                result = json.loads((run / 'analysis.json').read_text())
                self.assertEqual(result['tasks'][0]['status'], 'verification_failed')

    def test_running_launch_retains_progress_and_per_sequence_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, endpoints = fixture(run)
            task = self.modify(run, manifest, lambda p: p.update(status='running', results=p['results'][:1]))
            path = task_directory(run, task) / 'launch.json'
            launch = json.loads(path.read_text())
            launch['status'] = 'running'
            write(path, launch)
            result = self.report(run, manifest, endpoints)
            self.assertEqual(result['errors'], [])
            self.assertTrue(all(row['hydrophobic_fraction'] == 1 for row in result['summary']))
            self.assertFalse(any(row['stratum'].startswith('h_fraction') for row in result['seed_macros']))


if __name__ == '__main__':
    unittest.main()
