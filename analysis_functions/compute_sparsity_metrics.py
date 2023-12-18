import os
import jax
import numpy as np
import pandas as pd

from functools import partial
from jax.flatten_util import ravel_pytree
from qdax.utils.pareto_front import compute_sparsity
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.neuroevolution.networks.networks import MLP

from typing import Callable, List, Tuple

def calculate_sparsity_metrics(
    parent_dirname: str,
    env_names: List[str],
    env_dicts: dict,
    experiment_names: List[str],
    min_rewards: dict,
    max_rewards: dict,
    policy_hidden_layer_sizes: Tuple[int, ...]=(64, 64),
    max_pareto_front_length: int=50,
)-> Tuple[List, List, List, List]:

    """
    Calculate sparsity metrics across all experiment replications, in all envrionments
    """
    print("\n")
    print("---------------------------------------------------------------------------")
    print("  Calculating sparsity metrics for all experiments, in all environments")
    print("---------------------------------------------------------------------------")

    all_global_sparsity_scores = {}
    all_qd_sparsity_scores = {}
    
    random_key = jax.random.PRNGKey(0)

    for env in env_names:
        dirname = os.path.join(parent_dirname, env)

        # Create Reconstruction Function
        policy_layer_sizes = policy_hidden_layer_sizes + (env_dicts[env]["action_size"],)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=np.tanh,
        )

        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=max_pareto_front_length)
        fake_batch = np.zeros(shape=(max_pareto_front_length, env_dicts[env]["observation_size"],))
        fake_params = jax.vmap(policy_network.init)(keys, fake_batch)

        _, reconstruction_fn = ravel_pytree(fake_params)
    
        env_global_sparsities = {}
        env_qd_sparsities = {}
    
        for experiment_name in experiment_names:
            if experiment_name not in env_dicts[env]["exceptions"]:
                print("\n")
                print(f" ENV: {env}      EXP: {experiment_name}")
                
                global_sparsity_scores, qd_sparsity_scores = find_sparsity_metrics(
                    dirname,
                    experiment_name,
                    reconstruction_fn,
                    min_rewards[env],
                    max_rewards[env],
                    
                )
                print("Global Sparsity Median: ", np.median(np.array(global_sparsity_scores)))
                print("Global Sparsity STD: ", np.std(np.array(global_sparsity_scores)))
                print("QD Sparsity Median: ", np.median(np.array(qd_sparsity_scores)))
                print("QD Sparsity STD: ", np.std(np.array(qd_sparsity_scores)))
                
                env_global_sparsities[experiment_name] = global_sparsity_scores
                env_qd_sparsities[experiment_name] = qd_sparsity_scores
        
        env_global_sparsities = pd.DataFrame.from_dict(env_global_sparsities)
        env_qd_sparsities = pd.DataFrame.from_dict(env_qd_sparsities)
        env_global_sparsities.to_csv(f"{parent_dirname}/analysis/final_metrics/{env}_global_sparsity_scores.csv")
        env_qd_sparsities.to_csv(f"{parent_dirname}/analysis/final_metrics/{env}_qd_sparsity_scores.csv")

        all_global_sparsity_scores[env] = env_global_sparsities
        all_qd_sparsity_scores[env] = env_qd_sparsities
    
    return all_global_sparsity_scores, all_qd_sparsity_scores




def find_sparsity_metrics(
    dirname: str,
    experiment_name: str,
    reconstruction_fn: Callable,
    min_fitnesses: np.ndarray,
    max_fitnesses: np.ndarray,
) -> Tuple[List, List]:
    """
    Find QD-Sparsity and Global Sparsity for each experiment replication
    """
    exp_qd_sparsity_scores = []
    exp_global_sparsity_scores = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        repertoire = MOMERepertoire.load(
            reconstruction_fn=reconstruction_fn, 
            path=os.path.join(experiment_replication, "repertoire")
        )
                
        qd_sparsity, global_sparsity = compute_sparsity_metrics(
            repertoire,
            min_fitnesses,
            max_fitnesses,
        )    
        
        exp_global_sparsity_scores.append(global_sparsity)
        exp_qd_sparsity_scores.append(qd_sparsity)                             
    
    return exp_global_sparsity_scores, exp_qd_sparsity_scores


def compute_sparsity_metrics(
    repertoire: MOMERepertoire,
    min_fitnesses: np.ndarray,
    max_fitnesses: np.ndarray,
)-> Tuple[float, float]:
    
    "Calculate Sparsity Scores for given repertoire"
    repertoire_empty = repertoire.fitnesses == -np.inf # num centroids x pareto-front length x num criteria
    repertoire_empty = np.all(repertoire_empty, axis=-1) # num centroids x pareto-front length
    repertoire_not_empty = ~repertoire_empty # num centroids x pareto-front length
    repertoire_not_empty = np.any(repertoire_not_empty, axis=-1) # num centroids

    sparsity_function = partial(compute_sparsity, min_fitnesses=min_fitnesses, max_fitnesses=max_fitnesses)
    sparsities = jax.vmap(sparsity_function)(repertoire.fitnesses)
    average_sparsity = np.mean(repertoire_not_empty * sparsities)
    (
        pareto_front,
        _,
    ) = repertoire.compute_global_pareto_front()

    global_sparsity = sparsity_function(pareto_front)
    
    return average_sparsity, global_sparsity
    
