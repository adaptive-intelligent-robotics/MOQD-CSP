#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
import pathlib
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Optional

# import numba
import numpy as np
from ase import Atom
# from numba import jit
# from numba.experimental import jitclass
from sklearn.cluster import KMeans

from csp_elites.crystal.materials_data_model import MaterialProperties


class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid

def __centroids_filename(
    k: int, dim: int, bd_names: List[MaterialProperties],
    bd_minimum_values: List[float], bd_maximum_values: List[float],
):
    bd_tag = ""
    for i, bd_name in enumerate(bd_names):
        bd_tag += f"_{bd_name.value}_{bd_minimum_values[i]}_{bd_maximum_values[i]}"
    return '/centroids/centroids_' + str(k) + '_' + str(dim) + bd_tag +'.dat'


def write_centroids(
    centroids, experiment_folder, bd_names: List[MaterialProperties],
        bd_minimum_values: List[float], bd_maximum_values: List[float],
):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim, bd_names, bd_minimum_values, bd_maximum_values)
    file_path = Path(experiment_folder).parent
    with open(f"{file_path}{filename}", 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def cvt(k, number_of_bd_dimensions, samples, bd_minimum_values, bd_maximum_values,
        experiment_folder, bd_names: List[MaterialProperties], cvt_use_cache=True,):
    # check if we have cached values
    fname = __centroids_filename(k, number_of_bd_dimensions, bd_names, bd_minimum_values, bd_maximum_values)
    file_location = pathlib.Path(experiment_folder).parent
    if cvt_use_cache:
        if Path(f"{file_location}/{fname}").is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(f"{file_location}/{fname}")
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)


    bd_dim_1 = np.random.uniform(bd_minimum_values[0], bd_maximum_values[0], size=(samples, 1))
    bd_dim_2 = np.random.uniform(bd_minimum_values[1], bd_maximum_values[1], size=(samples, 1))

    x = np.hstack((bd_dim_1, bd_dim_2))
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)#,algorithm="full") ##  n_jobs=-1,
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_, experiment_folder,
                    bd_names, bd_minimum_values, bd_maximum_values)

    return k_means.cluster_centers_


def make_hashable(array):
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

def parallel_eval_no_multiprocess():
    pass

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def save_archive(archive, gen, directory_path):
    storage = []
    for k in archive.values():
        one_individual = [k.fitness, k.centroid, k.desc, k.x]
        storage.append(one_individual)

    filename_pkl = str(directory_path) + '/archive_' + str(gen) + '.pkl'
    with open(filename_pkl, 'wb') as f:
        pickle.dump(storage, f)

def add_to_archive(s, centroid, archive, kdt) -> Tuple[bool, int]:
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    s.centroid = n
    if "data" in s.x["info"]:
        parent_id = s.x["info"]["data"]["parents"]
    else:
        parent_id = [None]
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return True, parent_id
        return False, parent_id
    else:
        archive[n] = s
        return True, parent_id

def evaluate_old(to_evaluate):
    really_relax = True
    z, cellbounds, behavioural_descriptors, n_relaxation_steps, f = to_evaluate
    fit, desc, kill = f(z, cellbounds, really_relax, behavioural_descriptors, n_relaxation_steps)
    if kill:
        return None
    else:
        return Species(z, desc, fit)


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
# todo: @jit(cache=True)
# @jit
def evaluate(z, cellbounds, behavioural_descriptors, n_relaxation_steps, f) -> Optional[Species]:
    really_relax = True
    z, fit, desc, kill = f(z, cellbounds, really_relax, behavioural_descriptors, n_relaxation_steps)
    if kill:
        return None
    else:
        return Species(z, desc, fit)

def make_experiment_folder(directory_name: str):
    path = Path(__file__).parent.parent.parent
    new_path = path / "experiments" / directory_name
    new_path.mkdir(exist_ok=True)
    return new_path

def make_current_time_string(with_time: bool = True):
    today = date.today().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H_%M") if with_time else ""
    return f"{today}_{time_now}"



def evaluate_parallel(to_evaluate) -> List[Optional[Species]]:
    s_list = []

    for i in range(len(to_evaluate)):
        z, cellbounds, behavioural_descriptors, n_relaxation_steps, f = to_evaluate[i]
        if z is None:
            continue
        s = evaluate(
            z, cellbounds, behavioural_descriptors, n_relaxation_steps, f
        )
        s_list.append(s)
    return s_list
