from typing import List, Dict, Tuple
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
):
    """ Given a new species and list of species on current front return new front"""
    
    niche_fitnesses = [s.fitness for s in niche]
    front_candidates = np.concatenate([niche, [species]])
    
    front_bool = calculate_front(
        np.array(niche_fitnesses),
        species.fitness
    )
    new_front = [s for i, s in enumerate(front_candidates) if front_bool[i]]

    return new_front
    
    
def mome_add_to_niche(species: Species,
    niche: int,
    archive: Dict[str, List[Species]],
    max_front_size: int=10,
):
    if niche in archive:
        new_niche = add_to_front(
            species,
            archive[niche],
        )
        if len(new_niche) > max_front_size:
            remove_idx = np.random.randint(0, len(new_niche))
            new_niche = np.delete(new_niche, remove_idx) 
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

def calculate_crowding_distances(
    niche: List[Species],
)-> Tuple[float,...]:
    
    fitnesses = np.array([s.fitness for s in niche])
    
    if len(fitnesses) == 1:
        return [1], [0, 0]
    
    else:
        sorted_args = np.argsort(fitnesses, axis=0)[:, 0]
        sorted_fitnesses = fitnesses[sorted_args]
        sorted_distances = np.sum(np.abs(sorted_fitnesses[1:] - sorted_fitnesses[:-1]), axis=1)
        sorted_distances = np.insert(sorted_distances, 0, sorted_distances[0])
        sorted_distances = np.append(sorted_distances, sorted_distances[-1])    
        crowding_distances = [np.mean([sorted_distances[i], sorted_distances[i+1]]) for i in range(len(niche))]
        
        boundary_indices = sorted_args[0], sorted_args[-1]

        return np.take(crowding_distances, sorted_args), boundary_indices

def mome_crowding_selection_fn(
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
        
        # crowding distance calculation
        x_crowding_distances, _ = calculate_crowding_distances(x_niche)
        y_crowding_distances, _ = calculate_crowding_distances(y_niche)
        
        x_probs = x_crowding_distances / np.sum(x_crowding_distances)
        y_probs = y_crowding_distances / np.sum(y_crowding_distances)
        
        # parent selection
        x = np.random.choice(x_niche, p=x_probs)
        y = np.random.choice(y_niche, p=y_probs)
        parents_x.append(x)
        parents_y.append(y)
    
    return parents_x, parents_y

def mome_crowding_add_to_niche(species: Species,
    niche: int,
    archive: Dict[str, List[Species]],
    max_front_size: int=4,
):
    if niche in archive:
        new_niche = add_to_front(species, archive[niche])
        if len(new_niche) > max_front_size:
            crowding_distances, boundary_indices = calculate_crowding_distances(new_niche)
            crowding_distances[np.array(boundary_indices)] = np.inf
            remove_idx = np.argmin(crowding_distances)
            new_niche = np.delete(new_niche, remove_idx) 
        archive[niche] = new_niche
    else:
        archive[niche] = [species]
    return archive

def mome_metrics_fn(
    archive,
    config,
    n_evals,
):
    
    all_fitnesses = []
    hypervolumes = []
    num_solutions = 0
    hypervolume_fn = HV(ref_point=config.system.reference_point)
    
    for niche in archive.values():
        fitnesses = np.array([s.fitness for s in niche if np.all(s.fitness != -10000)])
        all_fitnesses.append(fitnesses)
        niche_hypervolume = hypervolume_fn(fitnesses * -1)
        hypervolumes.append(niche_hypervolume)
        num_solutions += len(niche)
        
    all_fitnesses = np.concatenate(all_fitnesses)

    global_front_bool = calculate_front(
        all_fitnesses,
        config.system.reference_point
    )
    
    global_front = np.array([s for i, s in enumerate(all_fitnesses) if global_front_bool[i]])
    
    global_hypervolume = hypervolume_fn(global_front * -1)
    metrics = {
        "evaluations": n_evals,
        "num_solutions": num_solutions,
        "max_sum_scores": np.max(np.sum(all_fitnesses, axis=1)),
        "max_hypervolume": np.max(hypervolumes),
        "max_energy_fitness": np.max(all_fitnesses[:,0]),
        "min_energy_fitness": np.min(all_fitnesses[:,0]),
        "energy_qd_score": np.sum(all_fitnesses[:,0]),
        "max_magmom_fitness": np.max(all_fitnesses[:,1]),
        "min_magmom_fitness": np.min(all_fitnesses[:,1]),
        "magmom_qd_score": np.sum(all_fitnesses[:,1]),
        "coverage": 100 * len(hypervolumes) / config.number_of_niches,
        "moqd_score": np.sum(hypervolumes),
        "global_hypervolume": global_hypervolume,

    }
    
    return metrics