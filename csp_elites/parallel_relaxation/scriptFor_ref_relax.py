from csp_elites.parallel_relaxation.structure_optimizer import MultiprocessOptimizer
from csp_elites.parallel_relaxation.structure_to_use import atoms_for_ref

if __name__ == '__main__':
    optimizer = MultiprocessOptimizer()

    optimizer.relax(atoms_for_ref, fmax=0.2, steps=10)
