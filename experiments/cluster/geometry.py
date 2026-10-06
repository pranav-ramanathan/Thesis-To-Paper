"""Independent integer-lattice witness validation, without Torch or OR-Tools."""

def contacts(seq, positions, radius=None):
    if len(positions) != len(seq) or len(seq) < 2 or set(seq) - {'H', 'P'}:
        raise ValueError('Expected a complete HP fold')
    points = [tuple(p) for p in positions]
    if any(len(p) != 3 or any(type(v) is not int for v in p) for p in points):
        raise ValueError('Coordinates must be integer triples')
    if len(set(points)) != len(points):
        raise ValueError('Self intersection')
    if points[:2] != [(0, 0, 0), (1, 0, 0)]:
        raise ValueError('Incorrect anchoring')
    if any(sum(abs(a-b) for a, b in zip(p, q)) != 1 for p, q in zip(points, points[1:])):
        raise ValueError('Broken backbone')
    if radius is not None and any(abs(v) > radius for p in points for v in p):
        raise ValueError('Outside declared cube')
    occupied = {p: i for i, p in enumerate(points)}
    result = 0
    for i, (x, y, z) in enumerate(points):
        if seq[i] != 'H':
            continue
        for q in ((x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)):
            j = occupied.get(q)
            if j is not None and j > i+1 and seq[j] == 'H':
                result += 1
    return result
