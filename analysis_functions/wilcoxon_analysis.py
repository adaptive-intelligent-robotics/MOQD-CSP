import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
from analysis_functions.load_datasets import get_final_metrics, get_gold_matching_metrics

from typing import List, Dict, Tuple


def wilcoxon_analysis(
    parent_dirname: str,
    env_names: List[str],
    env_dicts: Dict,
    experiment_names: List[str],
    metrics_list: List[str],
    num_replications: int
):

    print("--------------------------------------------------")
    print("         Calculating final metric p-values        ")
    print("-------------------------------------------------")

    _analysis_dir = os.path.join(parent_dirname, "analysis/")
    _final_metrics_dir = os.path.join(_analysis_dir, "final_metrics/")
    _wilcoxon_dir = os.path.join(_analysis_dir, "wilcoxon_tests/")

    os.makedirs(_analysis_dir, exist_ok=True)
    os.makedirs(_wilcoxon_dir, exist_ok=True)
    os.makedirs(_final_metrics_dir, exist_ok=True)

    for env in env_names:

        dirname = os.path.join(parent_dirname, env)
        
        for metric in metrics_list:

            print("\n")
            print(f"     ENV: {env}     METRIC: {metric}          ")

            all_final_metrics = {}

            for experiment in experiment_names:
                if experiment not in env_dicts[env]["exceptions"]:
                    experiment_final_scores = get_final_metrics(dirname, experiment, metric)
                    all_final_metrics[experiment] = experiment_final_scores[:num_replications]


            all_final_metrics_df = pd.DataFrame(all_final_metrics)
            all_final_metrics_df.to_csv(f"{_final_metrics_dir}/{env}__{metric}_final_metric.csv")

            pvalue_df, corrected_p_value_df = pairwise_wilcoxon_analysis(all_final_metrics)
            
            # Save final metrics and p-values

            pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_{metric}.csv")
            corrected_p_value_df.to_csv(f"{_wilcoxon_dir}/corrected_{env}_{metric}.csv")





def pairwise_wilcoxon_analysis(all_final_metrics: dict)-> None:
    """
    Calculate p-values for pairwise comparison of experiments for given metric
    """
    experiment_names = all_final_metrics.keys()
    p_value_df = pd.DataFrame(columns=experiment_names, index=experiment_names)
    corrected_p_value_df = pd.DataFrame(columns=experiment_names, index=experiment_names)

    for experiment_1 in experiment_names:
        experiment_1_p_values = []
        for experiment_2 in experiment_names:
            if experiment_1 == experiment_2:
                experiment_1_p_values.append(np.nan)
            else:
                res = wilcoxon(all_final_metrics[experiment_1], all_final_metrics[experiment_2])
                experiment_1_p_values.append(res.pvalue)

        # find holm bonferonni corrected p values
        corrected_p_values = holm_bonf_correction(experiment_1_p_values)
        p_value_df.loc[experiment_1] = experiment_1_p_values
        corrected_p_value_df.loc[experiment_1] = corrected_p_values

    return p_value_df, corrected_p_value_df

def holm_bonf_correction(p_values):

    # remove nans
    nan_indices = np.argwhere(np.isnan(p_values))
    non_nan_p_values = np.array(p_values)[~np.isnan(p_values)]

    # order non-nan p values
    ordered_p_indices = np.argsort(non_nan_p_values)
    n_hypotheses = len(non_nan_p_values) # only compare hypotheses that have non-nan p-values

    corrected_p_values = np.zeros_like(non_nan_p_values)

    for index in ordered_p_indices:
        # correct p value
        corrected_p_value = non_nan_p_values[index]*n_hypotheses

        # check if p value is smaller than previous ones
        if np.any(corrected_p_values > corrected_p_value):
            corrected_p_value = np.max(corrected_p_values)
        
        # update p value
        corrected_p_values[index] = corrected_p_value
        n_hypotheses -= 1
    
    for index in nan_indices:
        corrected_p_values = np.insert(corrected_p_values, index, np.nan)    

    return corrected_p_values



def gold_matches_wilcoxon_analysis(
    parent_dirname: str,
    env_names: List[str],
    env_dicts: Dict,
    experiment_names: List[str],
    num_replications: int
):

    print("--------------------------------------------------")
    print("         Calculating final metric p-values        ")
    print("-------------------------------------------------")

    _analysis_dir = os.path.join(parent_dirname, "analysis/")
    _final_metrics_dir = os.path.join(_analysis_dir, "final_metrics/")
    _wilcoxon_dir = os.path.join(_analysis_dir, "wilcoxon_tests/")

    os.makedirs(_analysis_dir, exist_ok=True)
    os.makedirs(_wilcoxon_dir, exist_ok=True)
    os.makedirs(_final_metrics_dir, exist_ok=True)

    for env in env_names:

        dirname = os.path.join(parent_dirname, env)

        print("\n")
        print(f"     ENV: {env}     METRIC: gold matches          ")

        all_final_metrics = {}

        for experiment in experiment_names:
            if experiment not in env_dicts[env]["exceptions"]:
                experiment_final_scores = get_gold_matching_metrics(dirname, experiment, num_replications)
                all_final_metrics[experiment] = experiment_final_scores[:num_replications]

        pvalue_df, corrected_p_value_df = pairwise_wilcoxon_analysis(all_final_metrics)
        
        # Save final metrics and p-values

        pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_gold_matches.csv")
        corrected_p_value_df.to_csv(f"{_wilcoxon_dir}/corrected_{env}_gold_matches.csv")