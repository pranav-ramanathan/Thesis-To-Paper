"""Reject late, incomplete, malformed and misattributed follow-up measurements."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finetune_followup_launch import task_directory
from finetune_followup_report import ARMS, summarize
from finetune_report import sha
from test_finetune_report import fixture as old_fixture, OPEN, SOURCE, SQUARE, write


def fixture(run):
    source = run / 'source_finetune'
    source.mkdir()
    old = old_fixture(source)
    for task in old['tasks']:
        receipt = task_directory(source, task) / 'training/result.json'
        if receipt.exists():
            trained = json.loads(receipt.read_text())
            trained['handoff'].update(rng_state_retained=True, elite_pool_retained=True,
                                      learning_rate_override=task['learning_rate'])
            write(receipt, trained)
    (run / 'data.json').write_bytes((source / 'data.json').read_bytes())
    manifest = {**old, 'tasks': [], 'inference_timepoints_s': [60, 300, 600],
                'attempt_checkpoints': [2, 4, 8], 'evaluation_seconds_per_sequence_per_seed': 600}
    for arm in ARMS:
        algorithm = arm.removesuffix('_low_lr')
        previous = next(t for t in old['tasks'] if t['arm'] == algorithm)
        old_folder = task_directory(source, previous)
        new = arm.endswith('_low_lr')
        task = {**previous, 'task': len(manifest['tasks']), 'arm': arm, 'algorithm': algorithm,
                'new_training_seconds': 1800 if new else 0,
                'learning_rate': 3e-5 if new else previous['learning_rate'],
                'parent': 'source_finetune/' + previous['parent'], 'evaluation_seconds': 600,
                'source_task_directory': str(old_folder.relative_to(run)) if not new else None,
                'training_receipt': (str((old_folder / 'training/result.json').relative_to(run))
                                     if not new and algorithm != 'parent' else None)}
        checkpoint = source / previous['parent'] if algorithm == 'parent' else old_folder / 'training/final.pt'
        payload = json.loads((old_folder / 'evaluation.json').read_text())
        task.update(checkpoint=str(checkpoint.relative_to(run)) if not new else None,
                    checkpoint_sha256=sha(checkpoint) if not new else None,
                    checkpoint_state_sha256=payload['checkpoint_metadata']['state_sha256'] if not new else None)
        folder = task_directory(run, task)
        folder.mkdir(parents=True)
        if new:
            (folder / 'training').mkdir()
            (folder / 'training/final.pt').write_bytes(checkpoint.read_bytes())
            trained = json.loads((old_folder / 'training/result.json').read_text())
            trained['training_settings']['learning_rate'] = task['learning_rate']
            trained['handoff']['learning_rate_override'] = task['learning_rate']
            write(folder / 'training/result.json', trained)
        metadata = payload['checkpoint_metadata']
        if algorithm != 'parent':
            metadata.update(effective_training_settings={'learning_rate': task['learning_rate']},
                            reference_state_sha256=task['parent_state_sha256'])
        final_score = 0 if arm in ('parent', 'grpo') else 1
        row = payload['results'][0]
        attempts, curve = [], []
        for index in range(1, 9):
            when = index if index < 5 else index + 345
            score = 0 if index < 5 else final_score
            status = 'trapped' if index == 4 else 'complete'
            positions = copy.deepcopy(OPEN if score == 0 else SQUARE) if status == 'complete' else None
            attempts.append({'attempt_index': index, 'seconds': when, 'status': status,
                             'contacts': score if status == 'complete' else None, 'positions': positions})
            if status == 'complete' and (not curve or score > curve[-1]['contacts']):
                curve.append({'seconds': when, 'contacts': score, 'positions': positions, 'attempt_index': index})
        prefixes = []
        for count in manifest['attempt_checkpoints']:
            prefix = attempts[:count]
            valid = [p for p in prefix if p['status'] == 'complete']
            best = max(valid, key=lambda p: p['contacts'])
            prefixes.append({'attempts': count, 'reached': True, 'covered': True, 'observed_attempts': count,
                             'seconds': prefix[-1]['seconds'], 'complete_folds': len(valid),
                             'trapped_folds': count - len(valid), 'interrupted_folds': 0,
                             'discarded_after_budget': 0, 'best_contacts': best['contacts'],
                             'best_positions': copy.deepcopy(best['positions'])})
        row.update(status='complete', budget_s=600, elapsed_s=600.01, attempted_folds=8,
                   generated_complete_folds=7, trapped_folds=1, interrupted_folds=0, discarded_after_budget=0,
                   valid_fold_rate=7/8, best_contacts=final_score, best_positions=copy.deepcopy(curve[-1]['positions']),
                   curve=copy.deepcopy(curve), attempt_checkpoints=manifest['attempt_checkpoints'],
                   ordered_attempts=attempts, attempt_prefixes=prefixes,
                   resource_usage={'process_cpu_s': 1200.02, 'setup_process_cpu_s': .1,
                                   'process_cpu_per_wall': 2.0, 'threads': 4, 'hostname': 'test-host',
                                   'verification_s': .5, 'snapshot_write_s': .1})
        payload.update(kind='finetune_followup_endpoint_evaluation', algorithm=algorithm, arm=arm,
                       evaluation_seconds_per_sequence_per_seed=600)
        write(folder / 'evaluation.json', payload)
        write(folder / 'launch.json', {'status': 'complete', 'software': old['software_at_freeze'], 'task': task})
        manifest['tasks'].append(task)
    return manifest


class FollowupReportTests(unittest.TestCase):
    def report(self, run, manifest):
        with patch('finetune_followup_launch.verify', return_value=manifest), contextlib.redirect_stdout(io.StringIO()):
            return summarize(run)

    def modify(self, run, manifest, arm, callback, filename='evaluation.json'):
        task = next(t for t in manifest['tasks'] if t['arm'] == arm)
        path = task_directory(run, task) / filename
        payload = json.loads(path.read_text())
        callback(payload)
        write(path, payload)
        return task

    def point(self, result, arm, comparison, budget):
        return next(p for p in result['points'] if (p['arm'], p['comparison'], p['budget']) ==
                    (arm, comparison, budget))

    def test_reuse_controls_negative_pairs_timed_cutoffs_and_exact_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            result = self.report(run, fixture(run))
            self.assertEqual(result['complete_tasks'], 8)
            self.assertEqual(result['verified_witness_records'], 78)
            self.assertEqual(result['costs']['planned_new_training_allocated_cpu_hours'], 4)
            self.assertEqual(len(result['training']), 2)
            self.assertEqual(self.point(result, 'dpo', 'time', 300)['contacts'], 0)
            self.assertEqual(self.point(result, 'dpo', 'time', 600)['contacts'], 1)
            self.assertEqual(self.point(result, 'dpo', 'attempts', 4)['contacts'], 0)
            self.assertEqual(self.point(result, 'dpo', 'attempts', 8)['contacts'], 1)
            paired = next(p for p in result['paired'] if (p['arm'], p['reference'], p['comparison'], p['budget']) ==
                          ('grpo', 'patternboost', 'time', 600))
            self.assertEqual((paired['mean_difference'], paired['negative']), (-1, 1))
            self.assertEqual(result['throughput'][0]['process_cpu_per_wall'], 2)
            for name in ('analysis.json', 'report.md', 'summary.csv', 'paired.csv', 'throughput.csv'):
                self.assertTrue((run / name).exists())

    def test_missing_prefix_coverage_cannot_be_filled_by_a_later_fold(self):
        for corrupt in ('score', 'coordinates', 'count', 'trap_removed', 'covered'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def change(payload):
                    row = payload['results'][0]
                    if corrupt == 'score':
                        row['attempt_prefixes'][1]['best_contacts'] = 1
                    elif corrupt == 'coordinates':
                        row['attempt_prefixes'][1]['best_positions'] = copy.deepcopy(SQUARE)
                    elif corrupt == 'count':
                        row['attempt_prefixes'][1]['observed_attempts'] = 5
                    elif corrupt == 'trap_removed':
                        row['ordered_attempts'].pop(3)
                    else:
                        row['attempt_prefixes'][1]['covered'] = False
                self.modify(run, manifest, 'dpo', change)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_every_complete_prefix_witness_is_checked_even_when_not_an_improvement(self):
        for corrupt in ('score', 'geometry', 'time', 'curve'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def change(payload):
                    row = payload['results'][0]
                    if corrupt == 'score':
                        row['ordered_attempts'][1]['contacts'] = 1
                    elif corrupt == 'geometry':
                        row['ordered_attempts'][1]['positions'][0][0] = 0.0
                    elif corrupt == 'time':
                        row['ordered_attempts'][7]['seconds'] = 600.001
                    else:
                        row['curve'][1]['attempt_index'] = 6
                self.modify(run, manifest, 'dpo', change)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_partial_prefix_and_failed_tasks_are_retained_but_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def partial(payload):
                payload['status'] = 'failed'
                row = payload['results'][0]
                row.update(status='failed', elapsed_s=4.1, attempted_folds=4, generated_complete_folds=3,
                           valid_fold_rate=.75, best_contacts=0, best_positions=copy.deepcopy(OPEN))
                row['curve'] = row['curve'][:1]
                row['ordered_attempts'] = row['ordered_attempts'][:4]
                row['attempt_prefixes'][2] = {**row['attempt_prefixes'][1], 'attempts': 8,
                    'reached': False, 'covered': False, 'best_contacts': None, 'best_positions': None}
                row.pop('resource_usage')
            self.modify(run, manifest, 'dpo', partial)
            result = self.report(run, manifest)
            self.assertEqual(result['complete_tasks'], 7)
            self.assertEqual(result['errors'], [])
            self.assertIsNone(self.point(result, 'dpo', 'attempts', 8)['contacts'])
            self.assertFalse(self.point(result, 'dpo', 'attempts', 4)['complete'])
            self.assertEqual(self.point(result, 'dpo', 'attempts', 4)['contacts'], 0)

    def test_missing_valid_folds_stay_missing_but_zero_scores_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def missing(payload):
                row = payload['results'][0]
                row.update(best_contacts=None, best_positions=None, curve=[], generated_complete_folds=0,
                           trapped_folds=8, valid_fold_rate=0)
                for record in row['ordered_attempts']:
                    record.update(status='trapped', contacts=None, positions=None)
                for prefix in row['attempt_prefixes']:
                    prefix.update(complete_folds=0, trapped_folds=prefix['attempts'],
                                  best_contacts=None, best_positions=None)
            self.modify(run, manifest, 'dpo', missing)
            result = self.report(run, manifest)
            self.assertIsNone(self.point(result, 'dpo', 'attempts', 8)['contacts'])
            self.assertTrue(self.point(result, 'dpo', 'attempts', 8)['complete'])
            self.assertEqual(self.point(result, 'grpo', 'attempts', 8)['contacts'], 0)

    def test_control_labels_cannot_replace_underlying_algorithm_or_learning_rate(self):
        for corrupt in ('stage', 'rate', 'reference', 'seed', 'test'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                def change(payload):
                    metadata = payload['checkpoint_metadata']
                    if corrupt == 'stage':
                        metadata['stage'] = 'grpo_low_lr'
                    elif corrupt == 'rate':
                        metadata['effective_training_settings']['learning_rate'] = .0003
                    elif corrupt == 'reference':
                        metadata['reference_state_sha256'] = 'wrong'
                    elif corrupt == 'seed':
                        metadata['seed'] = 1
                    else:
                        payload['split'] = 'test'
                self.modify(run, manifest, 'grpo_low_lr', change)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_reused_complete_training_receipts_are_required_and_checkpoints_immutable(self):
        for corrupt in ('missing', 'budget', 'parent', 'checkpoint', 'rng', 'elite', 'rate_override'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                task = next(t for t in manifest['tasks'] if t['arm'] == 'reinforce')
                receipt = run / task['training_receipt']
                if corrupt == 'missing':
                    receipt.unlink()
                elif corrupt == 'checkpoint':
                    (run / task['checkpoint']).write_bytes(b'changed')
                else:
                    trained = json.loads(receipt.read_text())
                    if corrupt in ('rng', 'elite', 'rate_override'):
                        key = {'rng': 'rng_state_retained', 'elite': 'elite_pool_retained',
                               'rate_override': 'learning_rate_override'}[corrupt]
                        trained['handoff'][key] = False
                    else:
                        trained['compute_elapsed_s' if corrupt == 'budget' else 'initial_state_sha256'] = 1799 if corrupt == 'budget' else 'wrong'
                    write(receipt, trained)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_new_training_partial_and_empty_evaluation_is_valid_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            task = self.modify(run, manifest, 'grpo_low_lr', lambda p: p.update(status='running', results=[],
                checkpoint_sha256=None, checkpoint_metadata=None))
            (task_directory(run, task) / 'training/result.json').unlink()
            self.modify(run, manifest, 'grpo_low_lr', lambda p: p.update(status='running'), 'launch.json')
            result = self.report(run, manifest)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['complete_tasks'], 7)
            self.assertEqual(len(result['training']), 1)

    def test_full_time_coverage_software_and_resource_accounting_are_enforced(self):
        for corrupt in ('time', 'sequence', 'launch', 'software', 'cpu', 'counts', 'no_resources', 'task'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest = fixture(run)
                if corrupt in ('launch', 'software', 'task'):
                    self.modify(run, manifest, 'parent', lambda p: p.update(
                        **({'status': 'failed'} if corrupt == 'launch' else
                           {'software': {}} if corrupt == 'software' else {'task': {}})), 'launch.json')
                else:
                    def change(payload):
                        row = payload['results'][0]
                        if corrupt == 'time':
                            row['elapsed_s'] = 599
                        elif corrupt == 'sequence':
                            payload['results'] = []
                        elif corrupt == 'cpu':
                            row['resource_usage']['process_cpu_s'] = -1
                        elif corrupt == 'no_resources':
                            row.pop('resource_usage')
                        else:
                            row['trapped_folds'] = 0
                    self.modify(run, manifest, 'parent', change)
                with self.assertRaises(SystemExit):
                    self.report(run, manifest)

    def test_interrupted_attempt_prevents_count_coverage_even_if_target_was_reached(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = fixture(run)
            def interrupted(payload):
                row = payload['results'][0]
                row['ordered_attempts'][7].update(status='interrupted', contacts=None, positions=None, seconds=600.001)
                row.update(generated_complete_folds=6, interrupted_folds=1, valid_fold_rate=.75)
                row['attempt_prefixes'][2].update(covered=False, complete_folds=6, interrupted_folds=1,
                    seconds=600.001, best_contacts=None, best_positions=None)
            self.modify(run, manifest, 'dpo', interrupted)
            result = self.report(run, manifest)
            self.assertFalse(self.point(result, 'dpo', 'attempts', 8)['covered'])
            self.assertTrue(self.point(result, 'dpo', 'time', 600)['covered'])


if __name__ == '__main__':
    unittest.main()
