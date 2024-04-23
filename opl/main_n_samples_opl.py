from omegaconf import DictConfig, OmegaConf
import hydra
import os

from dataset import SyntheticCombinatorialBanditDataset
from obp.dataset import linear_reward_function
from pandas import DataFrame
import pandas as pd
from sklearn.neural_network import MLPRegressor
from obp.ope import RegressionModel
from sklearn.utils import check_random_state
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from policyleaners import (
    GradientBasedPolicyLearner,
    RegBasedPolicyLearner,
    OPCB,
)
from learner import train_reward_model_via_two_stage
import matplotlib.pyplot as plt
import seaborn as sns

@hydra.main(config_path="../conf",config_name="config_n_samples_opl")
def main(cfg: DictConfig) -> None:

    dataset = SyntheticCombinatorialBanditDataset(
    n_unique_action=cfg.n_unique_action,
    dim_context=cfg.dim_context,
    base_reward_function=linear_reward_function,
    #behavior_policy_function=linear_behavior_policy,
    reward_type=cfg.reward_type,
    #reward_type='binary',
    random_state=cfg.random_state,
    reward_std=cfg.reward_std,
    beta=cfg.beta,
    lambda_=cfg.lambda_,
    )
    
    num_runs = cfg.num_runs
    epoch = cfg.epoch
    num_data_list = cfg.num_data_list
    random_state = cfg.random_state
    random_ = check_random_state(random_state)
    np.random.seed(random_state)
    n_comb_action = 2**(dataset.n_unique_action)

    test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.n_rounds_test_bandit_data, 
                                                            n_users=cfg.n_users, 
                                                            true_element=cfg.true_element,
                                                            )
    
    pi_0_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_bandit_data["expected_reward_matrix"], 
            action_dist=test_bandit_data["all_pscore"][:,:,np.newaxis],
        )

    result_df_list = []
    result_df = DataFrame()
    for num_data in num_data_list:
        test_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            
            test_value_of_learned_policies = dict()
            
            train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data,
                                                                     n_users=cfg.n_users, 
                                                                     true_element=cfg.true_element,
                                                                     )

            ips = GradientBasedPolicyLearner(dim_context = cfg.dim_context, n_action=n_comb_action, epoch = epoch)
            ips.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_ips = ips.predict(test_bandit_data)
            ips_value = (test_bandit_data["expected_reward_matrix"] * pi_ips).sum(1).mean()
            test_value_of_learned_policies["ips"] = ips_value
            
            reg = RegBasedPolicyLearner(dim_context = cfg.dim_context, n_action=n_comb_action, epoch = epoch)
            reg.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_reg = reg.predict(test_bandit_data)
            reg_value = (test_bandit_data["expected_reward_matrix"] * pi_reg).sum(1).mean()
            test_value_of_learned_policies["reg"] = reg_value

            q_hat = train_bandit_data["expected_reward_matrix"] + random_.normal(scale=cfg.q_hat_scale, size=(num_data, n_comb_action))
            dr = GradientBasedPolicyLearner(dim_context = cfg.dim_context, n_action=n_comb_action, epoch = epoch)
            dr.fit(dataset=train_bandit_data, dataset_test=test_bandit_data, q_hat=q_hat)
            pi_dr = dr.predict(test_bandit_data)
            dr_value = (test_bandit_data["expected_reward_matrix"] * pi_dr).sum(1).mean()
            test_value_of_learned_policies["dr"] = dr_value


       
            opcb = OPCB(dim_context = cfg.dim_context, n_action=n_comb_action, epoch = epoch, kth_element=cfg.true_element)
            opcb.fit(dataset=train_bandit_data, dataset_test=test_bandit_data, q_hat=q_hat)
            pi_opcb = opcb.predict(test_bandit_data)
            opcb_value = (test_bandit_data["expected_reward_matrix"] * pi_opcb).sum(1).mean()
            test_value_of_learned_policies["opcb"] = opcb_value
            


            test_policy_value_list.append(test_value_of_learned_policies)
            
        result_df = DataFrame(test_policy_value_list).stack().reset_index(1)\
            .rename(columns={"level_1": "method", 0: "value"})
        result_df["num_data"] = num_data
        result_df["pi_0_value"] = pi_0_value
        result_df["rel_value"] = result_df["value"] / pi_0_value
        result_df_list.append(result_df)

        tqdm.write("=====" * 15)
        
        
    # aggregate all results 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')
    os.chdir('result')
    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv("n_samples_opl.csv")


if __name__ == "__main__":
    main()
