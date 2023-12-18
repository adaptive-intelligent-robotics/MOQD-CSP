import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("muted")

from typing import Dict, List


def print_env_min_max_rewards(
    env_names: List[str],
    env_dicts: Dict,
    all_metrics_df: Dict,
)-> None:

    print("\n")
    print("---------------------------------------------------------------------------------")
    print("             Finding min and max of rewards obtained in each environment         ")
    print("---------------------------------------------------------------------------------")


    all_min_rewards = {}
    all_max_rewards = {}

    for env in env_names:

        print("\n")
        print(f"     ENV: {env}             ")
        
        env_reward_labels = env_dicts[env]["reward_labels"]
        num_rewards = len(env_reward_labels)

        min_rewards, max_rewards = print_min_max_rewards(
            all_metrics_df[env],
            num_rewards
        )
        
        all_min_rewards[env] = min_rewards
        all_max_rewards[env] = max_rewards
        
    return all_min_rewards, all_max_rewards


def print_min_max_rewards(
    all_metrics_df: Dict,
    num_rewards: int,
):

    # Create score columns 
    score_columns = []
    for reward_i in range(num_rewards):
        score_columns.append(f"score{reward_i+1}")

    all_min_scores = []
    all_max_scores = []
    
    for exp_metrics in all_metrics_df.values():
        
        # Create df that is one col for each algorithm
        exp_min_metrics = exp_metrics.obj.groupby(level=0).agg(list)["min_scores"].apply(pd.Series)
        exp_max_metrics = exp_metrics.obj.groupby(level=0).agg(list)["max_scores"].apply(pd.Series)
            
        # find min and max of each score for each replication (each column in df)
        for col in exp_min_metrics.columns:
            col_min_scores = []
            col_max_scores = []
            
            if num_rewards == 2:
                exp_min_scores = pd.DataFrame(exp_min_metrics[col].apply(lambda x: np.fromstring(str(x).replace('[','').replace(']',''), sep=' ')).to_list(), columns=score_columns)
                exp_max_scores = pd.DataFrame(exp_max_metrics[col].apply(lambda x: np.fromstring(str(x).replace('[','').replace(']',''), sep=' ')).to_list(), columns=score_columns)

            elif num_rewards == 3:
                exp_min_scores = pd.DataFrame(exp_min_metrics[col].apply(lambda x: np.fromstring(str(x).replace('[','').replace(',', ' ').replace(']',''), sep=' ')).to_list(), columns=score_columns)
                exp_max_scores = pd.DataFrame(exp_max_metrics[col].apply(lambda x: np.fromstring(str(x).replace('[','').replace(',', ' ').replace(']',''), sep=' ')).to_list(), columns=score_columns)
            
            for reward_i in range(num_rewards):
                min_score = exp_min_scores[f"score{reward_i+1}"].min()
                max_score = exp_max_scores[f"score{reward_i+1}"].max()
            
                col_min_scores.append(min_score)
                col_max_scores.append(max_score)
                
            all_min_scores.append(col_min_scores)
            all_max_scores.append(col_max_scores)


    # Find min and max of scores from all replications
    all_min_scores = np.min(np.array(all_min_scores), axis=0)
    all_max_scores = np.max(np.array(all_max_scores), axis=0)

    print(f"Min scores: {all_min_scores}")
    print(f"Max scores: {all_max_scores}")
        
    return all_min_scores, all_max_scores
