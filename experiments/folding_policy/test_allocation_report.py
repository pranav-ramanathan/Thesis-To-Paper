"""Reporting checks for endpoint provenance and paired training effects."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from allocation_report import summarize, verify_row


SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
OPEN = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(run):
    source = {'id': 'validation_000', 'sequence': 'HHHH', 'length': 4}
    write(run / 'data.json', {'validation': [source]})
    digest = sha(run / 'data.json')
    tasks = []
    for arm in ('patternboost', 'balanced', 'more_pretrain', 'rl_scratch'):
        parent = arm in ('balanced', 'more_pretrain')
        stages = ['stage_3600', 'stage_5400', 'stage_7200'] if arm == 'patternboost' else ['final']
        tasks.append({'arm': arm, 'seed': 0,
                      'training_seconds': 3600 if arm == 'balanced' else 1800 if arm == 'more_pretrain' else 7200,
                      'lineage_training_seconds': 7200, 'parent_arm': 'patternboost' if parent else None,
                      'resume_stage': 'stage_3600' if arm == 'balanced' else 'stage_5400' if parent else None,
                      'evaluation_stages': stages,
                      'stage_lineage_seconds': {stage: int(stage.split('_')[1]) if stage.startswith('stage_') else 7200
                                                for stage in stages}})
    manifest = {'threads': 4, 'tasks': tasks, 'validation_ids': [source['id']],
                'evaluation_seconds_per_sequence_per_seed': 60, 'hashes': {'data.json': digest}}
    parent_records = {}
    for task in tasks:
        arm = task['arm']
        mode = arm if arm in ('patternboost', 'rl_scratch') else 'rl_branch'
        algorithm_stage = 'patternboost' if arm == 'patternboost' else 'rl'
        condition = arm if arm in ('patternboost', 'rl_scratch') else 'patternboost_rl'
        folder = run / 'tasks' / f'{arm}_seed0'
        training = {'status': 'complete', 'mode': mode, 'seed': 0, 'dataset_sha256': digest,
                    'budget_s': task['training_seconds'], 'elapsed_s': task['training_seconds'] + .1,
                    'compute_elapsed_s': task['training_seconds'], 'test_evaluated': False,
                    'counters': {'supervised_updates': 3, 'rl_updates': 3}, 'checkpoint_records': []}
        for stage in task['evaluation_stages']:
            checkpoint = folder / 'training' / f'{stage}.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f'{arm}/{stage}/weights'.encode())
            record = {'filename': checkpoint.name, 'file_sha256': sha(checkpoint),
                      'state_sha256': f'model-{arm}-{stage}', 'optimizer_sha256': f'optimizer-{arm}-{stage}',
                      'training_seconds': task['stage_lineage_seconds'][stage]}
            training['checkpoint_records'].append(record)
            if arm == 'patternboost':
                parent_records[stage] = record
            contacts = 0 if arm in ('balanced', 'rl_scratch') else 1
            positions = SQUARE if contacts else OPEN
            result = {'id': source['id'], 'sequence': source['sequence'], 'seed': 0, 'method': 'model',
                      'budget_s': 60, 'elapsed_s': 60.01, 'best_contacts': contacts,
                      'best_positions': positions, 'attempted_folds': 10, 'generated_complete_folds': 9,
                      'checkpoint_stage': algorithm_stage, 'checkpoint_condition': condition,
                      'checkpoint_seed': 0, 'checkpoint_dataset_sha256': digest,
                      'curve': [{'seconds': 1, 'contacts': contacts, 'positions': positions}]}
            payload = {'kind': 'allocation_endpoint_evaluation', 'status': 'complete',
                       'split': 'validation', 'final_test_access': False, 'selection': 'fixed_training_endpoint',
                       'threads': 4, 'evaluation_seconds_per_sequence_per_seed': 60,
                       'dataset_file_sha256': digest, 'seed': 0, 'endpoint': stage,
                       'checkpoint_sha256': sha(checkpoint), 'checkpoint_metadata': {
                           'seed': 0, 'stage': algorithm_stage, 'mode': mode, 'condition': condition,
                           'dataset_sha256': digest, 'state_sha256': record['state_sha256'],
                           'training_seconds': record['training_seconds']}, 'results': [result]}
            write(folder / 'evaluations' / f'{stage}.json', payload)
        if task['parent_arm']:
            parent = parent_records[task['resume_stage']]
            training['initial_state_sha256'] = parent['state_sha256']
            training['lineage'] = {'parent_file_sha256': parent['file_sha256'],
                                   'parent_state_sha256': parent['state_sha256'],
                                   'parent_optimizer_sha256': parent['optimizer_sha256']}
            training['handoff'] = {'pretraining_sha256': parent['state_sha256'],
                                   'rl_initial_sha256': parent['state_sha256'],
                                   'pretraining_optimizer_sha256': parent['optimizer_sha256'],
                                   'rl_initial_optimizer_sha256': parent['optimizer_sha256'],
                                   'optimizer_state_retained': True, 'rng_state_retained': True,
                                   'elite_pool_retained': True}
        write(folder / 'training' / 'result.json', training)
        write(folder / 'launch.json', {'arm': arm, 'seed': 0, 'status': 'complete'})
    return manifest


class AllocationReportTests(unittest.TestCase):
    def report(self, run, manifest):
        with patch('allocation_launch.verify', return_value=manifest), contextlib.redirect_stdout(io.StringIO()):
            return summarize(run)

    def test_actual_post_rl_deterioration_is_retained_in_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            result = self.report(run, fixture(run))
            delta = next(row for row in result['contrast_summary']
                         if row['contrast'] == 'balanced_minus_its_pretraining')
            self.assertEqual(delta['mean_difference'], -1)
            self.assertEqual(delta['negative'], 1)
            endpoint = next(row for row in result['summary'] if row['arm'] == 'balanced')
            self.assertEqual(endpoint['mean_contacts'], 0)
            self.assertEqual(result['verified_witness_records'], 12)
            self.assertEqual(result['costs']['planned_shared_training_cpu_hours'], 22)
            self.assertEqual(result['costs']['planned_independent_lineage_training_cpu_hours'], 32)
            for filename in ('analysis.json', 'report.md', 'summary.csv', 'contrast.csv', 'curves.csv'):
                self.assertTrue((run / filename).exists())

    def test_partial_evaluation_remains_visible_but_excluded_from_final_means(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'balanced_seed0' / 'evaluations' / 'final.json'
            payload = json.loads(path.read_text())
            payload['status'] = 'running'
            write(path, payload)
            result = self.report(run, manifest)
            row = next(row for row in result['summary'] if row['arm'] == 'balanced')
            self.assertEqual(row['complete_runs'], 0)
            self.assertIsNone(row['mean_contacts'])
            endpoint = next(row for row in result['endpoints'] if row['arm'] == 'balanced')
            self.assertEqual(endpoint['status'], 'partial')
            self.assertEqual(endpoint['progress'][0]['best_contacts'], 0)

    def test_interrupted_training_does_not_become_a_complete_budget_result(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'rl_scratch_seed0' / 'training' / 'result.json'
            payload = json.loads(path.read_text())
            payload['status'] = 'interrupted'
            write(path, payload)
            result = self.report(run, manifest)
            row = next(row for row in result['summary'] if row['arm'] == 'rl_scratch')
            self.assertEqual(row['complete_runs'], 0)

    def test_corrupted_curve_witness_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'balanced_seed0' / 'evaluations' / 'final.json'
            payload = json.loads(path.read_text())
            payload['results'][0]['curve'][0]['contacts'] = 1
            write(path, payload)
            with self.assertRaises(SystemExit):
                self.report(run, manifest)
            result = json.loads((run / 'analysis.json').read_text())
            self.assertTrue(any('Curve contact count' in error for error in result['errors']))

    def test_wrong_endpoint_seed_or_checkpoint_hash_is_rejected(self):
        changes = [('endpoint', 'stage_3600'), ('seed', 9), ('checkpoint_sha256', 'changed')]
        for field, value in changes:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                path = run / 'tasks' / 'balanced_seed0' / 'evaluations' / 'final.json'
                payload = json.loads(path.read_text())
                payload[field] = value
                write(path, payload)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_wrong_branch_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'balanced_seed0' / 'training' / 'result.json'
            payload = json.loads(path.read_text())
            payload['handoff']['rl_initial_sha256'] = 'other-parent'
            write(path, payload)
            with self.assertRaises(SystemExit):
                self.report(run, manifest)

    def test_complete_evaluation_requires_all_planned_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'balanced_seed0' / 'evaluations' / 'final.json'
            payload = json.loads(path.read_text())
            payload['results'] = []
            write(path, payload)
            with self.assertRaises(SystemExit):
                self.report(run, manifest)

    def test_early_inference_cannot_claim_full_budget(self):
        row = {'id': 'v', 'sequence': 'HHHH', 'seed': 0, 'method': 'model', 'budget_s': 60, 'elapsed_s': 1}
        with self.assertRaisesRegex(ValueError, 'full budget'):
            verify_row(row, {'id': 'v', 'sequence': 'HHHH'}, 0, 60)

    def test_failed_model_loading_is_reported_without_claiming_an_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            path = run / 'tasks' / 'balanced_seed0' / 'evaluations' / 'final.json'
            payload = json.loads(path.read_text())
            payload.update(status='failed', checkpoint_metadata=None, results=[], error='Model loading failed')
            write(path, payload)
            result = self.report(run, manifest)
            endpoint = next(row for row in result['endpoints'] if row['arm'] == 'balanced')
            self.assertEqual(endpoint['status'], 'failed')
            self.assertEqual(endpoint['error'], 'Model loading failed')
            row = next(row for row in result['summary'] if row['arm'] == 'balanced')
            self.assertEqual(row['complete_runs'], 0)


if __name__ == '__main__':
    unittest.main()
