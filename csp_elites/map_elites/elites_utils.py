# | This file is based on the implementation map-elites implementation pymap_elites repo by resibots team https://github.com/resibots/pymap_elites
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#

import pathlib
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms
from sklearn.cluster import KMeans

from csp_elites.crystal.materials_data_model import MaterialProperties


class Species:
    def __init__(
        self,
        x,
        desc,
        fitness,
        centroid=None,
        fitness_gradient=None,
        descriptor_gradients=None,
    ):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid
        self.fitness_gradient = fitness_gradient
        self.descriptor_gradients = descriptor_gradients


def __centroids_filename(
    k: int,
    dim: int,
    bd_names: List[MaterialProperties],
    bd_minimum_values: List[float],
    bd_maximum_values: List[float],
    formula: str,
):
    if formula == "TiO2" or formula is None:
        bd_tag = ""
    else:
        bd_tag = "_" + formula

    for i, bd_name in enumerate(bd_names):
        bd_tag += f"_{bd_name.value}_{bd_minimum_values[i]}_{bd_maximum_values[i]}"

    return "/centroids_" + str(k) + "_" + str(dim) + bd_tag + ".dat"


def write_centroids(
    centroids,
    experiment_folder,
    bd_names: List[MaterialProperties],
    bd_minimum_values: List[float],
    bd_maximum_values: List[float],
    formula: str,
):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(
        k, dim, bd_names, bd_minimum_values, bd_maximum_values, formula
    )
    file_path = Path(experiment_folder)
    with open(f"{file_path}{filename}", "w") as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + " ")
            f.write("\n")


def cvt(
    k,
    number_of_bd_dimensions,
    samples,
    bd_minimum_values,
    bd_maximum_values,
    experiment_folder,
    bd_names: List[MaterialProperties],
    cvt_use_cache=True,
    formula: str = "",
    centroids_load_dir=None,
    centroids_save_dir=None,
):
    # check if we have cached values
    fname = __centroids_filename(
        k,
        number_of_bd_dimensions,
        bd_names,
        bd_minimum_values,
        bd_maximum_values,
        formula=formula,
    )
    if cvt_use_cache:
        if Path(f"{centroids_load_dir}/{fname}").is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(f"{centroids_load_dir}/{fname}")
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    bd_dim_1 = np.random.uniform(
        bd_minimum_values[0], bd_maximum_values[0], size=(samples, 1)
    )
    bd_dim_2 = np.random.uniform(
        bd_minimum_values[1], bd_maximum_values[1], size=(samples, 1)
    )
    x = np.hstack((bd_dim_1, bd_dim_2))
    if number_of_bd_dimensions == 3:
        bd_dim_3 = np.random.uniform(
            bd_minimum_values[1], bd_maximum_values[1], size=(samples, 1)
        )
        x = np.hstack((x, bd_dim_3))

    k_means = KMeans(
        init="k-means++", n_clusters=k, n_init=1, verbose=1
    )  # ,algorithm="full") ##  n_jobs=-1,
    k_means.fit(x)
    write_centroids(
        k_means.cluster_centers_,
        centroids_save_dir,
        bd_names,
        bd_minimum_values,
        bd_maximum_values,
        formula=formula,
    )

    return k_means.cluster_centers_


def make_hashable(array):
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params["parallel"] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)


def parallel_eval_no_multiprocess():
    pass


# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def save_archive(archive, gen, directory_path, name=""):
    storage = []
    for niche in archive.values():
        for s in niche:
            one_individual = [s.fitness, s.centroid, s.desc, s.x]
            storage.append(one_individual)

    filename_pkl = str(directory_path) + f"/{name}archive_" + str(gen) + ".pkl"
    with open(filename_pkl, "wb") as f:
        pickle.dump(storage, f)

def map_elites_add_to_niche(
    species: Species,
    niche: int,
    archive: Dict[str, List[Species]],
    objective_index: int = 0,
):
    if niche in archive:
        if species.fitness[objective_index] > archive[niche][0].fitness[objective_index]:
            archive[niche] = [species]
    else:
        archive[niche] = [species]
    return archive


def add_to_archive(
    s: Species,
    centroid: np.ndarray,
    archive: Dict[str, Species],
    kdt,
    add_to_niche_fn: Callable = map_elites_add_to_niche,
) -> Tuple[bool, int]:
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    s.centroid = n
    
    new_archive = add_to_niche_fn(
        s,
        n,
        archive
    )
    
    return new_archive
    

def map_elites_selection_fn(
    archive: Dict[str, List[Species]],
    batch_size: int,
) -> Tuple[List[Species], List[Species]]:
    
    keys = list(archive.keys())
    
    # we select all the parents at the same time because randint is slow
    rand1 = np.random.randint(len(keys), size=batch_size)
    rand2 = np.random.randint(len(keys), size=batch_size)

    parents_x = []
    parents_y = []
    
    for n in range(0, batch_size):
        # parent selection
        x = archive[keys[rand1[n]]][0]
        y = archive[keys[rand2[n]]][0]
        parents_x.append(x)
        parents_y.append(y)
    
    return parents_x, parents_y
        

def map_elites_metrics_fn(
    archive,
    config,
    n_evals,
):
    fit_list = np.array([s.fitness for niche in archive.values() for s in niche])
    qd_score = np.sum(fit_list)
    coverage = 100 * len(fit_list) / config.number_of_niches
    
    
    metrics = {
        "evaluations": n_evals,
        "archive_size": len(archive.keys()),
        "max_fit": np.max(fit_list),
        "mean_fit": np.mean(fit_list),
        "median_fit": np.median(fit_list),
        "5th_percentile_fit": np.percentile(fit_list, 5),
        "95th_percentile_fit": np.percentile(fit_list, 95),
        "coverage": coverage,
        "qd_score": qd_score,
    }
    
    return metrics


def evaluate_old(to_evaluate):
    really_relax = True
    z, cellbounds, behavioural_descriptors, n_relaxation_steps, f = to_evaluate
    fit, desc, kill = f(
        z, cellbounds, really_relax, behavioural_descriptors, n_relaxation_steps
    )
    if kill:
        return None
    else:
        return Species(z, desc, fit)


def evaluate(
    z, cellbounds, behavioural_descriptors, n_relaxation_steps, f
) -> Optional[Species]:
    really_relax = True
    z, fit, desc, kill = f(
        z, cellbounds, really_relax, behavioural_descriptors, n_relaxation_steps
    )
    if kill:
        return None
    else:
        return Species(z, desc, fit)


def make_experiment_folder(directory_name: str):
    path = Path(__file__).parent.parent.parent
    new_path = path / directory_name
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
        s = evaluate(z, cellbounds, behavioural_descriptors, n_relaxation_steps, f)
        s_list.append(s)
    return s_list
