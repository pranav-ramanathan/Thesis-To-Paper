"""Compact CP-SAT search in the common CPU benchmark cube."""
import math
import time
from ortools.sat.python import cp_model
from geometry import contacts


def solve(seq, *, seconds, threads, seed, emit, incumbent=None, movable=None, stopped=lambda:False):
    started = time.monotonic()
    n, radius = len(seq), len(seq)//2
    radix = 2*radius+1
    model = cp_model.CpModel()
    xyz = [[model.new_int_var(-min(i,radius), min(i,radius), f'p{i}_{a}') for a in range(3)] for i in range(n)]
    cells = [model.new_int_var(0, radix**3-1, f'cell{i}') for i in range(n)]
    for i in range(n):
        model.add(cells[i] == sum((xyz[i][a]+radius)*radix**a for a in range(3)))
    model.add_all_different(cells)
    for a, value in enumerate((1,0,0)):
        model.add(xyz[1][a] == value)

    def distance(i,j):
        ds = [model.new_int_var(0, 2*radius, f'd{i}_{j}_{a}') for a in range(3)]
        for a in range(3):
            model.add_abs_equality(ds[a], xyz[i][a]-xyz[j][a])
        return sum(ds)

    for i in range(n-1):
        model.add(distance(i,i+1) == 1)
    flags = []
    for i in range(n):
        for j in range(i+3,n,2):
            if seq[i] == seq[j] == 'H':
                flag = model.new_bool_var(f'contact{i}_{j}')
                d = distance(i,j)
                model.add(d == 1).only_enforce_if(flag)
                model.add(d >= 2).only_enforce_if(flag.Not())
                flags.append(flag)
    model.maximize(sum(flags))
    frozen = []
    if incumbent is not None:
        value = contacts(seq, incumbent, radius)
        model.add(sum(flags) >= value)
        for i, point in enumerate(incumbent):
            for a in range(3):
                model.add_hint(xyz[i][a], point[a])
                if movable is not None and i not in movable:
                    model.add(xyz[i][a] == point[a])
            if movable is not None and i not in movable:
                frozen.append(i)
    elif movable is not None:
        raise ValueError('Repair needs an incumbent')
    build_s = time.monotonic()-started
    remaining = seconds-build_s
    if remaining <= 0 or stopped():
        emit(dict(type='solver_end', status='NOT_STARTED', scope='repair' if movable is not None else 'full_cube', build_s=build_s))
        return None, 'NOT_STARTED'
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = threads
    solver.parameters.random_seed = seed
    scope = 'restricted_repair' if movable is not None else 'full_declared_cube'

    class Witness(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            fold = [[int(self.value(v)) for v in row] for row in xyz]
            score = contacts(seq, fold, radius)
            if score != round(self.objective_value):
                raise RuntimeError('Solver objective disagrees with independent contacts')
            if incumbent is not None and any(fold[i] != list(incumbent[i]) for i in frozen):
                raise RuntimeError('Repair changed a frozen residue')
            emit(dict(type='witness', source='cp_sat', seq=seq, positions=fold, contacts=score,
                      bound=float(self.best_objective_bound), bound_scope=scope))
            if stopped():
                self.stop_search()

    emit(dict(type='solver_start', seed=seed, workers=threads, scope=scope,
              seconds=remaining, build_s=build_s, movable=sorted(movable) if movable is not None else None))
    status = solver.solve(model, Witness())
    name = solver.status_name(status)
    fold = [[int(solver.value(v)) for v in row] for row in xyz] if status in (cp_model.FEASIBLE,cp_model.OPTIMAL) else None
    if fold is not None:
        if contacts(seq, fold, radius) != round(solver.objective_value):
            raise RuntimeError('Final solver witness mismatch')
    bound = float(solver.best_objective_bound)
    emit(dict(type='solver_end', status=name, scope=scope, wall_s=time.monotonic()-started,
              bound=bound if math.isfinite(bound) else None,
              contacts=contacts(seq,fold,radius) if fold is not None else None))
    return fold, name
