import numpy as np
import os
import pandas as pd
from typing import Any, List, Dict, Tuple

# from analysis_functions.compute_sparsity_metrics import calculate_sparsity_metrics
from analysis_functions.load_datasets import calculate_quartile_metrics, get_gold_matching_metrics
# from analysis_functions.plot_final_pfs import plot_pfs
from analysis_functions.plot_grid import plot_experiments_grid
from analysis_functions.print_min_max_rewards import print_env_min_max_rewards
from analysis_functions.wilcoxon_analysis import wilcoxon_analysis, pairwise_wilcoxon_analysis, gold_matches_wilcoxon_analysis


class MOQD_Analysis(
):
    
    def __init__(self,
        parent_dirname: str,
        env_names: List[str],
        env_dicts: Dict,
        experiment_names: List[str],
        experiment_dicts: Dict,
        num_replications: int=5,
        num_iterations: int=4000,
        episode_length: int=1000,
        batch_size: int=256,
    )-> None:
        
        
        self.parent_dirname = parent_dirname
        
        self.env_names = env_names
        self.env_dicts = env_dicts
        
        self.experiment_names = experiment_names
        self.experiment_dicts = experiment_dicts
        
        self.num_replications = num_replications
        self.num_iterations = num_iterations
        self.episode_length = episode_length
        self.batch_size = batch_size
        
        
        # Load in datasets and calculate metrics
        self.all_metrics, self.all_medians, self.all_lqs, self.all_uqs = self.find_quartile_metrics()   
        
        # Find min and max rewards
        # self.min_rewards, self.max_rewards = self.find_min_max_rewards()
                
        return
    
    
    def find_quartile_metrics(self)-> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        
        all_metrics, all_medians, all_lqs, all_uqs = calculate_quartile_metrics(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            self.num_replications,
        )
        
        return all_metrics, all_medians, all_lqs, all_uqs
        
    def find_min_max_rewards(
        self,
    )-> None:
        
        min_rewards, max_rewards = print_env_min_max_rewards(
            self.env_names,
            self.env_dicts,
            self.all_metrics
        )    
        
        return min_rewards, max_rewards
    
    def plot_grid(
        self,
        grid_plot_metrics_list: List[str],
        grid_plot_metrics_labels: Dict,
        num_iterations: int=4000,
        episode_length: int=1000,
        batch_size: int=256,
        x_axis_evaluations: bool=True,
    ) -> None:
                
        plot_experiments_grid(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            self.experiment_dicts,
            grid_plot_metrics_list,
            grid_plot_metrics_labels,
            self.all_medians, 
            self.all_lqs, 
            self.all_uqs,
            self.num_iterations,
            self.episode_length,
            self.batch_size,
            x_axis_evaluations
        )
        
        return


    def calculate_wilcoxon(
        self,
        p_value_metrics_list: List[str],
    )-> None:
        
        wilcoxon_analysis(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            p_value_metrics_list,
            self.num_replications,
        )
        
        gold_matches_wilcoxon_analysis(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            self.num_replications,
        )
        return
    
    def sparsity_analysis(
        self,
    )-> None:
        
        global_sparsity_scores, qd_sparsity_scores = calculate_sparsity_metrics(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            self.min_rewards,
            self.max_rewards,
        )
        
        _wilcoxon_dir = os.path.join(self.parent_dirname, "analysis/wilcoxon_tests/")
        for env in self.env_names:
        
            global_pvalue_df, global_corrected_p_value_df = pairwise_wilcoxon_analysis(global_sparsity_scores[env])
            qd_pvalue_df, qd_corrected_p_value_df = pairwise_wilcoxon_analysis(qd_sparsity_scores[env])            
            
            # Save final metrics and p-values
            global_pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_global_sparsity_scores.csv")
            global_corrected_p_value_df.to_csv(f"{_wilcoxon_dir}/corrected_{env}_global_sparsity_scores.csv")
            qd_pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_qd_sparsity_scores.csv")
            qd_corrected_p_value_df.to_csv(f"{_wilcoxon_dir}/corrected_{env}_qd_sparsity_scores.csv")
        
        return
    
    

    def analyse_gold_matches(self):
        print("--------------------------------------------------")
        print("         Calculating final gold matches p-values        ")
        print("-------------------------------------------------")
                
        _analysis_dir = os.path.join(self.parent_dirname, "analysis/")
        _final_metrics_dir = os.path.join(_analysis_dir, "final_metrics/")
        _wilcoxon_dir = os.path.join(_analysis_dir, "wilcoxon_tests/")

        os.makedirs(_analysis_dir, exist_ok=True)
        os.makedirs(_wilcoxon_dir, exist_ok=True)
        os.makedirs(_final_metrics_dir, exist_ok=True)

        for env in self.env_names:

            dirname = os.path.join(self.parent_dirname, env)
            

            print("\n")
            print(f"     ENV: {env}   ")

            all_gold_matches = {}

            for experiment in self.experiment_names:
                print("Experiment: ", experiment)
                if experiment not in self.env_dicts[env]["exceptions"]:
                    experiment_final_scores = get_gold_matching_metrics(dirname, experiment, self.num_replications)
                    print("Median: ", np.median(experiment_final_scores))
                    print("Mean: ", np.mean(experiment_final_scores))
                    if len(experiment_final_scores) < self.num_replications:
                        experiment_final_scores = np.pad(experiment_final_scores, (0, self.num_replications - len(experiment_final_scores)))
                    all_gold_matches[experiment] = experiment_final_scores[:self.num_replications]

            all_gold_matches = pd.DataFrame(all_gold_matches)
            all_gold_matches.to_csv(f"{_final_metrics_dir}/{env}__gold_matches_final_metric.csv")

            # pvalue_df, corrected_p_value_df = pairwise_wilcoxon_analysis(all_final_metrics)
            
            # Save final metrics and p-values

            # pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_gold_matches.csv")
            # corrected_p_value_df.to_csv(f"{_wilcoxon_dir}/corrected_{env}_gold_matches.csv")
            
            
    def plot_final_pfs(
        self,
    )-> None:
        
        plot_pfs(
            self.parent_dirname,
            self.env_names,
            self.env_dicts,
            self.experiment_names,
            self.experiment_dicts,
            self.num_replications,
        )
        
        return
    
    
    
