import json
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_palette("muted")

from typing import Dict, List, Tuple


def get_metrics(
    dirname: str,
    experiment_name: str,
    num_replications) -> pd.DataFrame:
    """
    Read in specific metrics for given experiment
    """

    experiment_metrics_list = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "metrics_history.csv"), nrows=50)
        experiment_metrics_list.append(metrics_df)
        
    if len(experiment_metrics_list) != num_replications:
        print(f"WARNING: {experiment_name} has {len(experiment_metrics_list)} replications, not {num_replications}")

    experiment_metrics_concat = pd.concat(experiment_metrics_list)
    experiment_metrics = experiment_metrics_concat.groupby(experiment_metrics_concat.index)
    return experiment_metrics


def calculate_quartile_metrics(parent_dirname: str,
    env_names: List[str],
    env_dicts: List[Dict],
    experiment_names: List[str],
    num_replications: int,
)-> Tuple[Dict, Dict, Dict, Dict]:

    """
    Calculate quartile metrics across all experiment replications, in all envrionments
    """
    print("\n")
    print("---------------------------------------------------------------------------")
    print("  Calculating quartile metrics for all experiments, in all environments")
    print("---------------------------------------------------------------------------")

    all_metrics = {}
    all_medians = {}
    all_lqs = {}
    all_uqs = {}

    for env in env_names:

        print("\n")
        print(f" ENV: {env}")
        print("---------------")

        dirname = os.path.join(parent_dirname, env)
        _analysis_dir = os.path.join(dirname, "analysis/")
        _plots_dir = os.path.join(_analysis_dir, "plots/")
        _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")
        _median_metrics_dir = os.path.join(_analysis_dir, "median_metrics/")

        os.makedirs(_analysis_dir, exist_ok=True)
        os.makedirs(_plots_dir, exist_ok=True)
        os.makedirs(_emitter_plots_dir, exist_ok=True)
        os.makedirs(_median_metrics_dir, exist_ok=True)

        metrics_dict = {}
        median_metrics_dict = {}
        lq_metrics_dict = {}
        uq_metrics_dict = {}
    
        for experiment_name in experiment_names:

            if experiment_name not in env_dicts[env]["exceptions"]:
                print("\n")
                print(f" EXP: {experiment_name}")
                
                experiment_metrics = get_metrics(dirname, experiment_name, num_replications)
                median_metrics = experiment_metrics.median(numeric_only=True)
                median_metrics.to_csv(f"{_median_metrics_dir}{experiment_name}_median_metrics")
                lq_metrics = experiment_metrics.apply(lambda x: x.quantile(0.25))
                uq_metrics =  experiment_metrics.apply(lambda x: x.quantile(0.75))
                
                metrics_dict[experiment_name] = experiment_metrics
                median_metrics_dict[experiment_name] = median_metrics
                lq_metrics_dict[experiment_name] = lq_metrics
                uq_metrics_dict[experiment_name] = uq_metrics

        all_metrics[env] = metrics_dict
        all_medians[env] = median_metrics_dict
        all_lqs[env] = lq_metrics_dict
        all_uqs[env] = uq_metrics_dict

    return all_metrics, all_medians, all_lqs, all_uqs


def get_final_metrics(dirname: str, 
    experiment_name: str,
    metric: str) -> np.array:
    """
    Load in final score of experiment across all replications for given metric
    """

    experiment_final_scores = []
    experiment_replications = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "metrics_history.csv"), nrows=50)
        final_score = np.array(metrics_df[metric])[-1]
        experiment_final_scores.append(final_score)
        experiment_replications.append(experiment_replication.name)
    
    if experiment_name == "mome_biased" and metric == "moqd_score":
        print("Median replication for mome_biased moqd_score: ", experiment_replications[np.argmax(experiment_final_scores)])

    return np.array(experiment_final_scores)


def get_gold_matching_metrics(dirname: str, 
    experiment_name: str,
    num_replications: str) -> np.array:
    """
    Load in final score of experiment across all replications for given metric
    """

    experiment_gold_matches_scores = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        file_path = os.path.join(experiment_replication, "ind_report_summary.json")
        if os.path.isfile(file_path):
            with open(file_path, 'r') as j:
                matches_dict = json.loads(j.read())
            gold_matches = np.array(matches_dict["number_gold"])
            experiment_gold_matches_scores.append(gold_matches)

    if len(experiment_gold_matches_scores) != num_replications:
        print(f"!!!WARNING!!!: {experiment_name} has {len(experiment_gold_matches_scores)} matches stats, not {num_replications}")

    return np.array(experiment_gold_matches_scores)