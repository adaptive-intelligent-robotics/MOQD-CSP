from typing import List, Dict

import numpy as np
from pymoo.indicators.hv import HV

from csp_elites.map_elites.elites_utils import Species


def calculate_front(
    current_front,
    point,
):
    """Given the fitness of the current front and a new point, calculate new boolean array of whether solutions are on the front"""
    cat_front = np.concatenate((current_front, [point]))  
    front_bool = np.ones(len(cat_front), dtype=bool)
    for i, point in enumerate(cat_front):
        front_bool[i] = np.all(np.any(cat_front[:i] < point, axis=1)) and np.all(np.any(cat_front[i+1:] < point, axis=1))
    return front_bool


def add_to_front(
    species,
    niche,
    max_front_size: int=10,
):
    """ Given a new species and list of species on current front return new front"""
    
    niche_fitnesses = [s.fitness for s in niche]
    front_candidates = np.concatenate([niche, [species]])
    
    front_bool = calculate_front(
        np.array(niche_fitnesses),
        species.fitness
    )
    new_front = [s for i, s in enumerate(front_candidates) if front_bool[i]]

    if len(new_front) > max_front_size:
        remove_idx = np.random.randint(0, len(new_front))
        new_front = np.delete(new_front, remove_idx)    
        
    return new_front
    
    
def mome_add_to_niche(species: Species,
    niche: int,
    archive: Dict[str, List[Species]],
):
    if niche in archive:
        new_niche = add_to_front(species, archive[niche])
        archive[niche] = new_niche
    else:
        archive[niche] = [species]
    return archive


def mome_uniform_selection_fn(
    archive: Dict[str, List[Species]],
    batch_size: int,
) -> Tuple[List[Species], List[Species]]:
    
    # Find which niches have been filled
    keys = list(archive.keys())
    
    # we select all the parents at the same time because randint is slow
    rand1 = np.random.randint(len(keys), size=batch_size)
    rand2 = np.random.randint(len(keys), size=batch_size)

    parents_x = []
    parents_y = []
    
    for n in range(0, batch_size):
        # niche selection
        x_niche = archive[keys[rand1[n]]]
        y_niche = archive[keys[rand2[n]]]
        # parent selection
        x = np.random.choice(x_niche)
        y = np.random.choice(y_niche)
        parents_x.append(x)
        parents_y.append(y)
    
    return parents_x, parents_y

def mome_metrics_fn(
    archive,
    config,
    n_evals,
):
    hypervolumes = []
    max_sum_scores = []
    num_solutions = 0
    hypervolume_fn = HV(ref_point=config.system.reference_point)
    
    for niche in archive.values():
        fitnesses = np.array([s.fitness for s in niche])
        niche_hypervolume = hypervolume_fn(fitnesses * -1)
        hypervolumes.append(niche_hypervolume)
        max_sum_scores.append(np.sum(fitnesses, axis=1).max())
        num_solutions += len(niche)
        
    metrics = {
        "evalutations": n_evals,
        "num_solutions": num_solutions,
        "max_sum_scores": np.max(max_sum_scores),
        "coverage": 100 * len(hypervolumes) / config.number_of_niches,
        "qd_score": np.sum(hypervolumes),
    }
    
    return metrics