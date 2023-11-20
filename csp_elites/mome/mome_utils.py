from typing import List, Dict

import numpy as np
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