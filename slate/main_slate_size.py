from omegaconf import DictConfig, OmegaConf
import hydra
import os
 
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
from learner import train_reward_model_via_two_stage
import itertools
import math
from itertools import combinations

# import open bandit pipeline (obp)
import obp
from obp.dataset import (
    linear_reward_function,
    logistic_reward_function,
    linear_behavior_policy,
    linear_behavior_policy_logit,
)
from obp.policy import IPWLearner
from obp.ope import (
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)

from dataset import SyntheticSlateBanditDataset
from estimators import (
    OPCB,
    PseudoInverse as PI,
    LatentIPS as LIPS, 

)
from ope import OffPolicyEvaluation

@hydra.main(config_path="../conf",config_name="config_slate_size")
def main(cfg: DictConfig) -> None:

    num_runs = cfg.num_runs 
    num_data = cfg.num_data
    slate_size_list = cfg.slate_size_list 


    result_df_list = []

    for slate_size in slate_size_list:
        
        ## 人工データ生成クラス
        dataset = SyntheticSlateBanditDataset(
                n_unique_action=cfg.n_unique_action,
                slate_size=slate_size,
                dim_context=cfg.dim_context,
                base_reward_function=linear_reward_function,
                #behavior_policy_function=linear_behavior_policy,
                reward_type=cfg.reward_type,
                reward_std= cfg.reward_std,
                random_state=cfg.random_state,
                beta=cfg.beta,
                interaction_factor = cfg.interaction_factor,
            )

        n_unique_action = dataset.n_unique_action
        n_slate = n_unique_action**slate_size


        test_rounds = cfg.n_rounds_test_bandit_data 
        test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_rounds, n_users=cfg.n_users,)


        pi_x_a = gen_eps_greedy(
                        expected_reward=test_bandit_data["individual_reward"],
                        eps=cfg.eps,
                    )
        pi_x_s = np.zeros(test_rounds*n_slate).reshape(test_rounds,n_slate)

        for i in range(test_rounds):
            for s in range(n_slate):
                pi_x_s[i,s] = np.prod(pi_x_a[i,test_bandit_data["action_context"][s],np.arange(slate_size)])


        ## calculate V(\pi)
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_bandit_data["expected_reward_matrix"], 
            action_dist=pi_x_s[:,:,np.newaxis],
        )
        
        estimated_policy_value_list = []

        for _ in tqdm(range(num_runs), desc=f"slate_size={slate_size}..."):
            ## generate validation data
            validation_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data,
                n_users=cfg.n_users,
            )

            ## make decisions on vlidation data
            
            pi_x_a = gen_eps_greedy(
                    expected_reward=validation_bandit_data["individual_reward"],
                    eps=cfg.eps,
                )
            
            pi_val = np.zeros(num_data*n_slate).reshape(num_data,n_slate)
            
            for i in range(num_data):
                for s in range(n_slate):
                    pi_val[i,s] = np.prod(pi_x_a[i,validation_bandit_data["action_context"][s],np.arange(slate_size)])

            pi_val = pi_val[:,:,np.newaxis]

            ## OPE using validation data
            reg_model = RegressionModel(
                n_actions=n_slate, 
                base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),
            )
            
            estimated_rewards = reg_model.fit_predict(
                context=validation_bandit_data["context"], # context; x
                action=validation_bandit_data["slate_id"], # action; a
                reward=validation_bandit_data["reward"], # reward; r
                random_state=12345,
            )
            
            #optimize OPCB
            mse_pred = math.inf
            kth_element_optimized = 0
            for i in range(slate_size+1):
                for kth_element_ in combinations(np.arange(slate_size), i):
                    kth_element = list(kth_element_)

                    opcb=OPCB(estimator_name="opcb_for_opt")
                    opcb.set_(kth_element = kth_element, bandit_data = validation_bandit_data)

                    opcb_round_reward = opcb._estimate_round_rewards(
                                    reward=validation_bandit_data["reward"],
                                    action=validation_bandit_data["slate_id"],
                                    action_context=validation_bandit_data["action_context"],
                                    pscore=validation_bandit_data["pscore"],
                                    all_pscore=validation_bandit_data["all_pscore"],
                                    action_dist=pi_val,
                                )

                    opcb_value = opcb_round_reward.mean()

                    squared_bias = (policy_value - opcb_value)**2
                    sample_variance = np.var(opcb_round_reward)/num_data
                    opcb_noisy_mse = squared_bias + np.random.normal(loc=0, scale = cfg.bias_noise_std) + sample_variance

                    
                    if (opcb_noisy_mse) <  mse_pred:
                        kth_element_optimized = kth_element
                        mse_pred = opcb_noisy_mse
                    
            
            

            
            case1=OPCB(estimator_name="OPCB (ours)")
            case2=PI(estimator_name="pi")
            case3=LIPS(estimator_name="lips")
            
            case1.set_(kth_element = kth_element_optimized, bandit_data = validation_bandit_data)
            case2.set_(pi_x_a=pi_x_a, bandit_data = validation_bandit_data)
            case3.set_(n_slate_abstraction=cfg.n_slate_abstraction, bandit_data = validation_bandit_data)
            
            
            
            
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
                #ground_truth_policy_value=value_of_ipw, # V(\pi)
                action_dist=pi_val, # \pi(a|x)
                estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
                #estimated_rewards_by_two_step = f_x_a,
            )
            estimated_policy_value_list.append(estimated_policy_values)
        ## maximum importance weight in the validation data
        ### a larger value indicates that the logging and evaluation policies are greatly different
        max_iw = (pi_val[
            np.arange(validation_bandit_data["n_rounds"]), 
            validation_bandit_data["slate_id"], 
            0
        ] / validation_bandit_data["pscore"]).max()
        tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")


        
        ## summarize results
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["slate_size"] = slate_size
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
    result_df.to_csv("slate_size.csv")


if __name__ == "__main__":
    main()
