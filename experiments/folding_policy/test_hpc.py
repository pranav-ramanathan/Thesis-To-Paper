"""Bounded stdlib tests for split protection, snapshots and witness reporting."""
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import random
import json

from dataset import make_dataset
from evaluate import verified_score, validate_checkpoint
from launch import sha, validate_data, verify, write_json
from report import summarize


class PilotSafetyTests(unittest.TestCase):
    def test_checkpoint_requires_matching_data_and_complete_validation(self):
        saved = {'dataset_sha256': 'expected', 'selection': {'split': 'validation', 'validated': True}}
        validate_checkpoint(saved, 'expected')
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            validate_checkpoint(saved, 'changed')
        saved['selection']['validated'] = False
        with self.assertRaisesRegex(ValueError, 'completed validation'):
            validate_checkpoint(saved, 'expected')

    def test_reverse_leakage_is_rejected(self):
        data = make_dataset()
        data['validation'][0]['sequence'] = data['train'][0]['sequence'][::-1]
        with self.assertRaisesRegex(ValueError, 'shared'):
            validate_data(data)

    def test_four_validation_lengths_and_compositions_selected(self):
        data = make_dataset()
        selected = validate_data(data)
        self.assertEqual(selected, [data['validation'][i]['id'] for i in (0, 5, 10, 15)])
        self.assertFalse(set(selected) & {r['id'] for r in data['test'] + data['benchmark']})

    def test_frozen_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            data = make_dataset()
            write_json(run / 'data.json', data)
            (run / 'code.py').write_text('version = 1\n')
            manifest = {'final_test_access': False, 'validation_ids': validate_data(data),
                        'hashes': {name: sha(run / name) for name in ('data.json', 'code.py')}}
            write_json(run / 'pilot.json', manifest)
            verify(run)
            (run / 'code.py').write_text('version = 2\n')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                verify(run)

    def test_independent_scoring_and_invalid_folds(self):
        square = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        self.assertEqual(verified_score('HHHH', square), 1)
        self.assertEqual(verified_score('HPPH', square), 1)
        with self.assertRaisesRegex(ValueError, 'complete'):
            verified_score('HHHH', square[:-1])
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            verified_score('HHHH', square[:3] + [square[0]])
        with self.assertRaisesRegex(ValueError, 'Integer'):
            verified_score('HHHH', [[float(v) for v in p] for p in square])

    def test_report_rejects_corrupted_improvement_witness(self):
        from geometry import greedy
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            data = make_dataset()
            write_json(run / 'data.json', data)
            task = {'task': 0, 'condition': 'rl_scratch', 'seed': 0, 'controls': ['greedy']}
            manifest = {'final_test_access': False, 'validation_ids': validate_data(data),
                        'hashes': {'data.json': sha(run / 'data.json')}, 'tasks': [task],
                        'evaluation_seconds_per_sequence_per_seed': 60, 'evaluation_seeds': 2}
            write_json(run / 'pilot.json', manifest)
            folder = run / 'tasks' / '00_rl_scratch_seed0'
            folder.mkdir(parents=True)
            source = data['validation'][0]
            fold = greedy(source['sequence'], rng=random.Random(1), attempts=100)
            row = {'id': source['id'], 'sequence': source['sequence'], 'seed': 0, 'method': 'greedy',
                   'budget_s': 60, 'elapsed_s': 60, 'best_contacts': fold['contacts'],
                   'best_positions': fold['positions'], 'attempted_folds': 1, 'generated_complete_folds': 1,
                   'proof_within_budget': False, 'bound': None,
                   'curve': [{'seconds': 1, 'contacts': fold['contacts'], 'positions': fold['positions']}]}
            payload = {'split': 'validation', 'final_test_access': False,
                       'dataset_file_sha256': sha(run / 'data.json'), 'results': [row]}
            path = folder / 'greedy_evaluation.json'
            write_json(path, payload)
            with contextlib.redirect_stdout(io.StringIO()):
                result = summarize(run)
            self.assertEqual(result['summary'][0]['mean_contacts'], fold['contacts'])
            payload['results'][0]['curve'][0]['contacts'] += 1
            write_json(path, payload)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                summarize(run)
            analysis = json.loads((run / 'analysis.json').read_text())
            self.assertTrue(any('Curve contact count' in error for error in analysis['errors']))


if __name__ == '__main__':
    unittest.main()
