import copy
import unittest
from experiments.folding_policy.dataset import make_dataset, validate_dataset, digest


class DatasetTests(unittest.TestCase):
    def test_reproducible_disjoint_variable_length_splits(self):
        data = make_dataset()
        self.assertEqual(data, make_dataset())
        self.assertEqual([len(data[k]) for k in ('train', 'validation', 'test', 'benchmark')], [160, 16, 80, 8])
        self.assertTrue(validate_dataset(data))
        self.assertFalse(set(data['held_out_training_lengths']) & {r['length'] for r in data['train']})
        self.assertEqual([r['length'] for r in data['validation'][:4]], [24, 40, 60, 72])

    def test_reversed_test_sequence_cannot_be_training_data(self):
        data = make_dataset()
        data['train'][0] = copy.deepcopy(data['benchmark'][0])
        data['train'][0]['sequence'] = data['train'][0]['sequence'][::-1]
        data['split_sha256'] = digest({k: data[k] for k in ('train', 'validation', 'test', 'benchmark')})
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_dataset(data)


if __name__ == '__main__':
    unittest.main()
