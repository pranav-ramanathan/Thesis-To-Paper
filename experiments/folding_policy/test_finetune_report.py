"""Verify fine-tuning endpoints, paired comparisons, budgets and stale outputs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finetune_launch import task_directory
from finetune_report import ARMS, UPDATE_KEYS, sha, summarize

SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
OPEN = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]]
SOURCE = {'id': 'validation_000', 'sequence': 'HHHH', 'length': 4}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + '\n')


def fixture(run):
    write(run / 'data.json', {'validation': [SOURCE]})
    dataset_hash = sha(run / 'data.json')
    parent = run / 'parents/patternboost_seed0.pt'
    parent.parent.mkdir()
    parent.write_bytes(b'parent')
    tasks = []
    for arm in ARMS:
        task = {'task': len(tasks), 'arm': arm, 'seed': 0, 'parent': str(parent.relative_to(run)),
                'parent_sha256': sha(parent), 'parent_state_sha256': 'parent-state',
                'parent_optimizer_sha256': 'parent-optimizer',
                'training_seconds': 0 if arm == 'parent' else 1800,
                'learning_rate': 0 if arm == 'parent' else .0003,
                'lineage_training_seconds': 7200 if arm == 'parent' else 9000,
                'evaluation_seconds': 60}
        folder = task_directory(run, task)
        folder.mkdir(parents=True)
        metadata = {'schema': 1, 'kind': 'allocation_training_state' if arm == 'parent' else 'finetune_training_state',
                    'arm': None if arm == 'parent' else arm, 'seed': 0, 'threads': 4,
                    'dataset_sha256': dataset_hash,
                    'state_sha256': 'parent-state' if arm == 'parent' else arm + '-state',
                    'training_seconds': task['lineage_training_seconds'],
                    'mode': 'patternboost' if arm == 'parent' else 'finetune',
                    'stage': 'patternboost' if arm == 'parent' else arm,
                    'condition': 'patternboost' if arm in ('parent', 'patternboost') else arm}
        digest = task['parent_sha256']
        if arm != 'parent':
            checkpoint = folder / 'training/final.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(arm.encode())
            digest = sha(checkpoint)
            counters = {UPDATE_KEYS[arm]: 3, 'generated_folds': 8, 'valid_folds': 7,
                        'rl_groups': 4, 'zero_advantage_groups': 1}
            record = {'filename': 'final.pt', 'file_sha256': digest, 'state_sha256': arm + '-state',
                      'training_seconds': 9000, 'selected': False}
            trained = {'schema': 1, 'status': 'complete', 'arm': arm, 'seed': 0, 'threads': 4,
                       'budget_s': 1800, 'base_training_seconds': 7200, 'additional_training_seconds': 1800,
                       'training_seconds': 9000, 'dataset_sha256': dataset_hash,
                       'initial_state_sha256': task['parent_state_sha256'],
                       'parent_state_sha256': task['parent_state_sha256'], 'parent_file_sha256': task['parent_sha256'],
                       'final_state_sha256': arm + '-state', 'checkpoint_records': [record],
                       'reference_state_sha256': 'parent-state',
                       'handoff': {'pretraining_sha256': 'parent-state', 'finetune_initial_sha256': 'parent-state',
                                   'pretraining_optimizer_sha256': 'parent-optimizer', 'optimizer_moments_retained': True},
                       'final_test_enabled': False, 'selection': {'selected': False},
                       'compute_elapsed_s': 1800.1, 'test_evaluated': False,
                       'segment_counters': counters, 'diagnostics': {'entropy': 1.5},
                       'training_settings': {'learning_rate': task['learning_rate']}}
            write(folder / 'training/result.json', trained)
            metadata.update(base_training_seconds=7200, additional_training_seconds=1800,
                            parent_file_sha256=task['parent_sha256'], parent_state_sha256=task['parent_state_sha256'],
                            segment_counters=copy.deepcopy(counters))
        score = 0 if arm in ('parent', 'grpo') else 1
        positions = copy.deepcopy(OPEN if score == 0 else SQUARE)
        row = {**SOURCE, 'method': 'model', 'seed': 0, 'budget_s': 60, 'elapsed_s': 60.01,
               'setup_s': .1, 'attempted_folds': 10, 'generated_complete_folds': 9,
               'valid_fold_rate': .9, 'best_contacts': score, 'best_positions': positions,
               'curve': [{'seconds': 2, 'contacts': score, 'positions': copy.deepcopy(positions)}],
               'checkpoint_stage': metadata['stage'], 'checkpoint_condition': metadata['condition'],
               'checkpoint_seed': 0, 'checkpoint_dataset_sha256': dataset_hash, 'proof_within_budget': False}
        payload = {'schema': 1, 'kind': 'finetune_endpoint_evaluation', 'status': 'complete',
                   'split': 'validation', 'final_test_access': False, 'selection': 'fixed_training_endpoint',
                   'arm': arm, 'method': 'model', 'seed': 0, 'threads': 4, 'ids': [SOURCE['id']],
                   'evaluation_seconds_per_sequence_per_seed': 60, 'model_setup_charged_per_sequence': True,
                   'dataset_file_sha256': dataset_hash, 'checkpoint_sha256': digest,
                   'checkpoint_metadata': metadata, 'results': [row]}
        write(folder / 'evaluation.json', payload)
        write(folder / 'launch.json', {'status': 'complete', 'software': {'python': '3.11', 'torch': '2.11'}})
        tasks.append(task)
    return {'kind': 'validation_finetune_pilot', 'threads': 4, 'tasks': tasks,
            'software_at_freeze': {'python': '3.11', 'torch': '2.11'},
            'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
            'evaluation_seconds_per_sequence_per_seed': 60, 'validation_ids': [SOURCE['id']],
            'hashes': {'data.json': dataset_hash}}


class FineTuneReportTests(unittest.TestCase):
    def report(self, run, manifest):
        with patch('finetune_launch.verify', return_value=manifest), contextlib.redirect_stdout(io.StringIO()):
            return summarize(run)

    def modify(self, run, manifest, arm, update, filename='evaluation.json'):
        task = next(t for t in manifest['tasks'] if t['arm'] == arm)
        path = task_directory(run, task) / filename
        value = json.loads(path.read_text())
        update(value)
        write(path, value)
        return task

    def test_shared_parent_and_equal_additional_cost_with_negative_paired_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            result = self.report(run, fixture(run))
            self.assertEqual(result['complete_tasks'], 6)
            self.assertEqual(result['verified_witness_records'], 12)
            self.assertEqual(result['costs']['planned_new_training_allocated_cpu_hours'], 10)
            self.assertEqual(result['costs']['training_lineage_allocated_cpu_hours'], 50)
            contrasts = {(r['arm'], r['reference']): r for r in result['paired']}
            self.assertEqual(contrasts['grpo', 'patternboost']['mean_difference'], -1)
            self.assertEqual(contrasts['grpo', 'parent']['mean_difference'], 0)
            self.assertEqual(contrasts['dpo', 'parent']['mean_difference'], 1)
            self.assertEqual(contrasts['grpo', 'patternboost']['negative'], 1)
            for filename in ('analysis.json', 'report.md', 'summary.csv', 'paired.csv'):
                self.assertTrue((run / filename).exists())

    def test_missing_folds_and_running_tasks_do_not_turn_into_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def missing(payload):
                payload['results'][0].update(best_contacts=None, best_positions=None, curve=[],
                                              generated_complete_folds=0, valid_fold_rate=0)
            self.modify(run, manifest, 'dpo', missing)
            self.modify(run, manifest, 'reinforce', lambda p: p.update(status='running'))
            result = self.report(run, manifest)
            rows = {row['arm']: row for row in result['summary']}
            self.assertIsNone(rows['dpo']['mean_contacts'])
            self.assertEqual((rows['dpo']['valid_runs'], rows['dpo']['complete_runs']), (0, 1))
            self.assertIsNone(rows['reinforce']['mean_contacts'])
            self.assertEqual(rows['grpo']['mean_contacts'], 0)
            self.assertEqual(next(r for r in result['points'] if r['arm'] == 'reinforce')['contacts'], 1)
            self.assertEqual(next(r for r in result['paired'] if r['arm'] == 'reinforce')['paired_runs'], 0)

    def test_active_training_with_initialized_empty_evaluation_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            task = self.modify(run, manifest, 'dpo', lambda p: p.update(status='running', results=[],
                                checkpoint_sha256=None, checkpoint_metadata=None))
            folder = task_directory(run, task)
            (folder / 'training/result.json').unlink()
            write(folder / 'launch.json', {'status': 'running'})
            result = self.report(run, manifest)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['complete_tasks'], 5)

    def test_incomplete_training_cannot_become_a_complete_endpoint(self):
        for field, value in (('status', 'interrupted'), ('compute_elapsed_s', 1799),
                             ('initial_state_sha256', 'wrong'), ('additional_training_seconds', 1700)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                self.modify(run, manifest, 'grpo', lambda t: t.update({field: value}), 'training/result.json')
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)
                self.assertTrue(json.loads((run / 'analysis.json').read_text())['errors'])

    def test_late_curve_wrong_contact_and_invalid_coordinates_fail(self):
        for corruption in ('late', 'score', 'float'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(payload):
                    row = payload['results'][0]
                    if corruption == 'late':
                        row['curve'][0]['seconds'] = 60.001
                    elif corruption == 'score':
                        row['curve'][0]['contacts'] = 0
                    else:
                        row['best_positions'][0][0] = .0
                self.modify(run, manifest, 'dpo', corrupt)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_stale_checkpoint_and_result_counter_mismatch_fail(self):
        for corruption in ('file', 'counters', 'state'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                task = next(t for t in manifest['tasks'] if t['arm'] == 'dpo')
                if corruption == 'file':
                    (task_directory(run, task) / 'training/final.pt').write_bytes(b'changed')
                elif corruption == 'counters':
                    self.modify(run, manifest, 'dpo', lambda t: t['segment_counters'].update(preference_updates=999),
                                'training/result.json')
                else:
                    self.modify(run, manifest, 'dpo', lambda t: t.update(final_state_sha256='wrong'), 'training/result.json')
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_provenance_resource_and_fold_counter_mismatches_fail(self):
        for corruption in ('test', 'setup', 'resource', 'ids', 'counts'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(payload):
                    if corruption == 'test':
                        payload['split'] = 'test'
                    elif corruption == 'setup':
                        payload['model_setup_charged_per_sequence'] = False
                    elif corruption == 'resource':
                        payload['threads'] = 8
                    elif corruption == 'ids':
                        payload['results'].append(copy.deepcopy(payload['results'][0]))
                    else:
                        payload['results'][0]['generated_complete_folds'] = 11
                self.modify(run, manifest, 'patternboost', corrupt)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_complete_evaluation_requires_all_sequences_and_full_budget(self):
        for corruption in ('missing', 'partial'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(payload):
                    if corruption == 'missing':
                        payload['results'] = []
                    else:
                        payload['results'][0]['elapsed_s'] = 59
                self.modify(run, manifest, 'parent', corrupt)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_completed_launch_and_frozen_software_are_required(self):
        for corruption in ('software', 'launch_status'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(launch):
                    if corruption == 'software':
                        launch['software']['torch'] = 'changed'
                    else:
                        launch['status'] = 'failed'
                self.modify(run, manifest, 'dpo', corrupt, 'launch.json')
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_final_test_selection_and_parent_optimizer_identity_are_enforced(self):
        for corruption in ('final_test', 'selection', 'reference', 'optimizer'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def corrupt(trained):
                    if corruption == 'final_test':
                        trained['final_test_enabled'] = True
                    elif corruption == 'selection':
                        trained['selection']['selected'] = True
                    elif corruption == 'reference':
                        trained['reference_state_sha256'] = 'wrong'
                    else:
                        trained['handoff']['pretraining_optimizer_sha256'] = 'wrong'
                self.modify(run, manifest, 'dpo', corrupt, 'training/result.json')
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_all_three_seed_pairs_are_preserved_without_sequence_pseudoreplication(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            for seed in (1, 2):
                for original in list(manifest['tasks'][:6]):
                    task = {**original, 'task': len(manifest['tasks']), 'seed': seed}
                    old, new = task_directory(run, original), task_directory(run, task)
                    new.mkdir()
                    for filename in ('launch.json', 'evaluation.json', 'training/result.json'):
                        if not (old / filename).exists():
                            continue
                        value = json.loads((old / filename).read_text())
                        if filename == 'evaluation.json':
                            value['seed'] = value['checkpoint_metadata']['seed'] = seed
                            value['results'][0]['seed'] = value['results'][0]['checkpoint_seed'] = seed
                        elif filename.endswith('result.json'):
                            value['seed'] = seed
                        write(new / filename, value)
                    if (old / 'training/final.pt').exists():
                        (new / 'training/final.pt').write_bytes((old / 'training/final.pt').read_bytes())
                    manifest['tasks'].append(task)
            result = self.report(run, manifest)
            self.assertEqual(result['complete_tasks'], 18)
            self.assertTrue(all(row['planned_runs'] == 3 for row in result['summary']))
            self.assertTrue(all(row['paired_runs'] == 3 for row in result['paired']))
            self.assertTrue(all(row['sd_contacts'] == 0 for row in result['summary']))


if __name__ == '__main__':
    unittest.main()
