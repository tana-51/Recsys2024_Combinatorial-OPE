from omegaconf import DictConfig, OmegaConf
import hydra
import os

# from logging import getLogger
# from pathlib import Path
# from time import time
# import warnings

import numpy as np

import pandas as pd
from pandas import DataFrame

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from policy import gen_eps_greedy
import math



# import open bandit pipeline (obp)
import obp

from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)

from estimators import (
    OPCB,
) 
from dataset import ExtremeBanditDataset
from ope import OffPolicyEvaluation

@hydra.main(config_path="../conf",config_name="config_n_samples_real")
def main(cfg: DictConfig) -> None:

    num_runs = cfg.num_runs 
    num_data_list = cfg.num_data_list
    n_optimize = cfg.n_optimize

    dataset = ExtremeBanditDataset(
                dataset_name=cfg.dataset_name,
                max_reward_noise=cfg.max_reward_noise,
                random_state=cfg.random_state,

    )

    if dataset.dataset_name == "pen":
        n_unique_action = 10
    else:
        n_unique_action = 5

    np.random.seed(cfg.random_state)
    unique_action_set = np.arange(n_unique_action)
    kth_order_set = np.arange(1,n_unique_action)

    x = np.zeros(n_unique_action-1)
    for i in kth_order_set:
        x[i-1]=(math.comb(n_unique_action, i))
        
    p = x/np.sum(x)


    test_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=cfg.n_rounds_test_bandit_data, 
            n_users=cfg.n_users,
            beta=cfg.beta,
            reward_std=cfg.reward_std,
        )


    ## calculate V(\pi)
    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=test_bandit_data["expected_reward_matrix"], 
        action_dist=gen_eps_greedy(
                    expected_reward=test_bandit_data["expected_reward_matrix"],
                    eps=cfg.eps,
                ),
    )

    result_df_list = []

    for num_data in num_data_list:
        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## generate validation data
            validation_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data,
                n_users=cfg.n_users,
                beta=cfg.beta,
                reward_std=cfg.reward_std,
            )
    
            ## make decisions on vlidation data (pi_e)
            pi_val = gen_eps_greedy(
                    expected_reward=validation_bandit_data["expected_reward_matrix"],
                    eps=cfg.eps,
                )
            

            ## OPE using validation data
            reg_model = RegressionModel(
                n_actions=validation_bandit_data["n_comb_action"], 
                base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=cfg.random_state),
            )
            estimated_rewards = reg_model.fit_predict(
                context=validation_bandit_data["context"], # context; x
                action=validation_bandit_data["action"], # action; a
                reward=validation_bandit_data["reward"], # reward; r
                random_state=cfg.random_state,
            )
            
            #optimize k
            mse_pred1 = math.inf
            mse_pred2 = math.inf
            mse_pred3 = math.inf
            kth_element_optimized1 = 0
            kth_element_optimized2 = 0
            kth_element_optimized3 = 0
            kth_order_for_opt = np.random.choice(kth_order_set, size=n_optimize, replace=True,p=p)
            kth_order_for_opt = np.hstack((kth_order_for_opt, [0,n_unique_action]))
            for num_kth_order in kth_order_for_opt:
                kth_element = np.random.choice(unique_action_set, size=num_kth_order, replace=False)
                
                kth_element = list(kth_element)


                opcb=OPCB(estimator_name="opcb_opt")
                opcb.set_(kth_element = kth_element, bandit_data = validation_bandit_data)

                opcb_round_reward = opcb._estimate_round_rewards(
                                    reward=validation_bandit_data["reward"],
                                    action=validation_bandit_data["action"],
                                    action_context=validation_bandit_data["action_context"],
                                    pscore=validation_bandit_data["pscore"],
                                    all_pscore=validation_bandit_data["all_pscore"],
                                    action_dist=pi_val,
                                )
                opcb_value = opcb_round_reward.mean()


                squared_bias = (policy_value - opcb_value)**2
                sample_variance = np.var(opcb_round_reward)/num_data
                opcb_noisy_mse1 = squared_bias + np.random.normal(loc=0, scale = cfg.bias_noise_std_s) + sample_variance
                opcb_noisy_mse2 = squared_bias + np.random.normal(loc=0, scale = cfg.bias_noise_std_m) + sample_variance
                opcb_noisy_mse3 = squared_bias + np.random.normal(loc=0, scale = cfg.bias_noise_std_l) + sample_variance
                
                if (opcb_noisy_mse1) <  mse_pred1:
                    kth_element_optimized1 = kth_element
                    mse_pred1 = opcb_noisy_mse1
                
                if (opcb_noisy_mse2) <  mse_pred2:
                    kth_element_optimized2 = kth_element
                    mse_pred2 = opcb_noisy_mse2
                
                if (opcb_noisy_mse3) <  mse_pred3:
                    kth_element_optimized3 = kth_element
                    mse_pred3 = opcb_noisy_mse3
                
                    
            
            case1=OPCB(estimator_name="OPCB (sigma=1.0)",use_q_hat=True, estimated_rewards_by_reg_model=estimated_rewards)
            case2=OPCB(estimator_name="OPCB (sigma=3.0)",use_q_hat=True, estimated_rewards_by_reg_model=estimated_rewards)
            case3=OPCB(estimator_name="OPCB (sigma=5.0)",use_q_hat=True, estimated_rewards_by_reg_model=estimated_rewards)
            case1.set_(kth_element = kth_element_optimized1,bandit_data = validation_bandit_data)
            case2.set_(kth_element = kth_element_optimized2,bandit_data = validation_bandit_data)
            case3.set_(kth_element = kth_element_optimized3,bandit_data = validation_bandit_data)

            
            ope = OffPolicyEvaluation(
                bandit_feedback=validation_bandit_data,
                ope_estimators=[
                    IPS(estimator_name="IPS"), 
                    DM(estimator_name="DM"),  
                    DR(estimator_name="DR"),
                    case1,
                    case2,
                    case3,
                ]
            )
            
            estimated_policy_values = ope.estimate_policy_values(
                action_dist=pi_val, # \pi(a|x)
                estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
            )
            estimated_policy_value_list.append(estimated_policy_values)
        ## maximum importance weight in the validation data
        ### a larger value indicates that the logging and evaluation policies are greatly different
        max_iw = (pi_val[
            np.arange(validation_bandit_data["n_rounds"]), 
            validation_bandit_data["action"], 
            0
        ] / validation_bandit_data["pscore"]).max()
        tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")


        
        ## summarize results
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["num_data"] = num_data
        result_df["se"] = (result_df.value - policy_value) ** 2
        result_df["bias"] = 0
        result_df["variance"] = 0

        sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = result_df.loc[result_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            result_df.loc[result_df["est"] == est_, "bias"] = (
                policy_value - mean_estimates
            ) ** 2
            result_df.loc[result_df["est"] == est_, "variance"] = (
                estimates - mean_estimates
            ) ** 2
        result_df_list.append(result_df)
        
        tqdm.write("=====" * 15)
            
    # aggregate all results 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')
    os.chdir('result')
    result_df = pd.concat(result_df_list).reset_index(level=0)

    if dataset.dataset_name == "letter":
        result_df.to_csv(f"n_samples_real_{dataset.dataset_name}.csv")
    else:
        result_df.to_csv(f"n_samples_real_{dataset.dataset_name}_last.csv")

if __name__ == "__main__":
    main()

