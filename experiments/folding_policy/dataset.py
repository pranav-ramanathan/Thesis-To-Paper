"""Deterministic HP sequence partitions; never generates benchmark folds."""
import argparse
import hashlib
import json
import random
from pathlib import Path

TRAIN_LENGTHS = (16, 20, 24, 28, 32, 40, 46, 50, 60, 64)
VALIDATION_LENGTHS = (24, 40, 60, 72)
TEST_LENGTHS = (20, 24, 25, 36, 46, 48, 50, 58, 64, 72)
FRACTIONS = (0.25, 0.4, 0.6, 0.75)
BENCHMARK_COUNTS = (11, 13, 9, 18, 35, 31, 34, 44)
BENCHMARK_SOURCE = 'https://arxiv.org/html/2412.20329v1#S3'


def sequence_key(seq):
    """Exclude chain reversals as well as identical sequences across partitions."""
    return min(seq, seq[::-1])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_dataset(seed=20261013, benchmark_path=None):
    path = Path(benchmark_path) if benchmark_path else Path(__file__).resolve().parents[1] / 'cluster' / 'protocol.json'
    configs = json.loads(path.read_text())['configurations']
    benchmark = []
    for (name, row), count in zip(sorted(configs.items()), BENCHMARK_COUNTS):
        benchmark.append(dict(id=name, sequence=row['sequence'], length=len(row['sequence']),
                              best_known_contacts=count, reference=BENCHMARK_SOURCE,
                              optimum_status='reported optimum proved in our declared cube' if name in ('3d1', '3d2', '3d3') else 'best reported; no proof established here'))
    if len(benchmark) != 8:
        raise ValueError('Expected exactly eight historical benchmark sequences')
    rng = random.Random(seed)
    used = {sequence_key(row['sequence']) for row in benchmark}

    def rows(split, lengths, repeats):
        result = []
        for repeat in range(repeats):
            for fraction in FRACTIONS:
                for length in lengths:
                    for _ in range(10000):
                        chars = ['H'] * round(length * fraction) + ['P'] * (length-round(length * fraction))
                        rng.shuffle(chars)
                        sequence = ''.join(chars)
                        key = sequence_key(sequence)
                        # Avoid trivial near-copies of the already examined public strings.
                        near = any(len(b['sequence']) == length and min(
                            sum(a != c for a, c in zip(sequence, b['sequence'])),
                            sum(a != c for a, c in zip(sequence[::-1], b['sequence']))) <= 2 for b in benchmark)
                        if key not in used and not near:
                            used.add(key)
                            result.append(dict(id=f'{split}_{len(result):03d}', sequence=sequence,
                                               length=length, hydrophobic_fraction=sequence.count('H')/length))
                            break
                    else:
                        raise RuntimeError('Could not generate distinct split')
        return result

    data = dict(schema_version=1, seed=seed,
                provenance='Synthetic sequences generated before training; no fold or score filtering.',
                train=rows('train', TRAIN_LENGTHS, 4),
                validation=rows('validation', VALIDATION_LENGTHS, 1),
                test=rows('test', TEST_LENGTHS, 2), benchmark=benchmark,
                held_out_training_lengths=[25, 36, 48, 58, 72],
                public_benchmark_caveat='Historical scores already observed; fixed external evaluation, not a previously unseen research benchmark.')
    data['split_sha256'] = digest({key: data[key] for key in ('train', 'validation', 'test', 'benchmark')})
    validate_dataset(data)
    return data


def validate_dataset(data):
    seen = {}
    for split in ('train', 'validation', 'test', 'benchmark'):
        if not data.get(split):
            raise ValueError(f'Empty partition: {split}')
        for row in data[split]:
            sequence = row['sequence']
            if not sequence or set(sequence) - {'H', 'P'} or len(sequence) != row['length']:
                raise ValueError(f'Invalid sequence: {row["id"]}')
            key = sequence_key(sequence)
            if key in seen:
                raise ValueError(f'Sequence/reversal leakage: {split} and {seen[key]}')
            seen[key] = split
    train_lengths = {row['length'] for row in data['train']}
    if train_lengths.intersection(data.get('held_out_training_lengths', [])):
        raise ValueError('Held-out training length leaked')
    expected = digest({key: data[key] for key in ('train', 'validation', 'test', 'benchmark')})
    if data.get('split_sha256') != expected:
        raise ValueError('Split hash mismatch')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--benchmark-protocol', type=Path)
    args = parser.parse_args()
    data = make_dataset(args.seed, args.benchmark_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(path=str(args.output), split_sha256=data['split_sha256'],
                          counts={key: len(data[key]) for key in ('train', 'validation', 'test', 'benchmark')})))


if __name__ == '__main__':
    main()
