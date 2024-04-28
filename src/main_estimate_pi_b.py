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

# import open bandit pipeline (obp)
import obp
from obp.utils import softmax
from obp.dataset import (
    linear_reward_function,
    logistic_reward_function,
    linear_behavior_policy,
    linear_behavior_policy_logit,
)
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)

from dataset import SyntheticCombinatorialBanditDataset
from estimators import OPCB 
from ope import OffPolicyEvaluation
from obp.ope.helper import (
    estimate_high_probability_upper_bound_bias,
    estimate_bias_in_ope,
)


@hydra.main(config_path="../conf",config_name="config_estimate_pi_b")
def main(cfg: DictConfig) -> None:
    ## 実験設定
    num_runs = cfg.num_runs 
    num_data = cfg.num_data
    pi_b_noise_list = cfg.pi_b_noise_list
    n_optimize = cfg.n_optimize


    dataset = SyntheticCombinatorialBanditDataset(
        n_unique_action=cfg.n_unique_action,
        dim_context=cfg.dim_context,
        base_reward_function=linear_reward_function,
        #behavior_policy_function=linear_behavior_policy,
        reward_type=cfg.reward_type,
        reward_std= cfg.reward_std,
        random_state=cfg.random_state,
        beta=cfg.beta,
        lambda_=cfg.lambda_,
    )
    
    

    
    n_comb_action = 2**(dataset.n_unique_action)
    np.random.seed(cfg.random_state)
    unique_action_set = np.arange(dataset.n_unique_action)
    kth_order_set = np.arange(1,dataset.n_unique_action)

    x = np.zeros(len(kth_order_set))
    for i in kth_order_set:
        x[i-1]=(math.comb(dataset.n_unique_action, i))
        
    p = x/np.sum(x)


  
    test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.n_rounds_test_bandit_data, n_users=cfg.n_users,true_element=cfg.true_element)


    ## calculate V(\pi)
    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=test_bandit_data["expected_reward_matrix"], 
        action_dist=gen_eps_greedy(
                    expected_reward=test_bandit_data["expected_reward_matrix"],
                    eps=cfg.eps,
                ),
    )

    result_df_list = []
    for pi_b_noise in pi_b_noise_list:
        estimated_policy_value_list = []
        
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## generate validation data
            validation_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data,
                n_users=cfg.n_users,
                true_element=cfg.true_element,
            )

            ## make decisions on vlidation data
            pi_val = gen_eps_greedy(
                    expected_reward=validation_bandit_data["expected_reward_matrix"],
                    eps=cfg.eps,
                )

            ## OPE using validation data
            reg_model = RegressionModel(
                n_actions=n_comb_action, 
                base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=cfg.random_state),
            )
            estimated_rewards = reg_model.fit_predict(
                context=validation_bandit_data["context"], # context; x
                action=validation_bandit_data["action"], # action; a
                reward=validation_bandit_data["reward"], # reward; r
                random_state=12345,
            )

            pi_b_hat_logits = validation_bandit_data["expected_reward_matrix"]+np.random.normal(loc=0,scale=pi_b_noise,size=(num_data,n_comb_action))
            pi_b_hat=softmax(cfg.beta*pi_b_hat_logits)
            
            estimated_pscore = pi_b_hat[np.arange(num_data), validation_bandit_data["action"]]

            #optimize k
            mse_pred = math.inf
            mse_true = math.inf
            kth_element_optimized = 0
            kth_element_true = 0
            kth_order_for_opt = np.random.choice(kth_order_set, size=n_optimize, replace=True,p=p)
            kth_order_for_opt = np.hstack((kth_order_for_opt, [0,dataset.n_unique_action]))
            for num_kth_order in kth_order_for_opt:
                kth_element = np.random.choice(unique_action_set, size=num_kth_order, replace=False)
                
                kth_element = list(kth_element)


                opcb=OPCB(estimator_name="opcb_opt",use_estimated_pscore=True,estimated_all_pscore=pi_b_hat)
                opcb.set_(kth_element = kth_element, bandit_data = validation_bandit_data)

                opcb_round_reward = opcb._estimate_round_rewards(
                                    reward=validation_bandit_data["reward"],
                                    action=validation_bandit_data["action"],
                                    action_context=validation_bandit_data["action_context"],
                                    pscore=validation_bandit_data["pscore"],
                                    all_pscore=pi_b_hat,
                                    action_dist=pi_val,
                                )
                opcb_value = opcb_round_reward.mean()
                

                squared_bias = (policy_value - opcb_value)**2
                sample_variance = np.var(opcb_round_reward)/num_data
                opcb_noisy_mse = squared_bias + np.random.normal(loc=0, scale = cfg.bias_noise_std) + sample_variance
                opcb_mse_true = (policy_value - opcb_value)**2

                
                if (opcb_noisy_mse) <  mse_pred:
                    kth_element_optimized = kth_element
                    mse_pred = opcb_noisy_mse
                
                if  opcb_mse_true < mse_true:
                    kth_element_true = kth_element
                    mse_true = opcb_mse_true



            case1=OPCB(estimator_name="OPCB (optimized)",use_estimated_pscore=True,estimated_all_pscore=pi_b_hat)
            case2=OPCB(estimator_name="OPCB (true-optimized)",use_estimated_pscore=True,estimated_all_pscore=pi_b_hat)
            case3=OPCB(estimator_name="OPCB (true)",use_estimated_pscore=True,estimated_all_pscore=pi_b_hat)
            case1.set_(kth_element = kth_element_optimized, bandit_data = validation_bandit_data)
            case2.set_(kth_element = kth_element_true, bandit_data = validation_bandit_data)
            case3.set_(kth_element = cfg.true_element, bandit_data = validation_bandit_data)



            ope = OffPolicyEvaluation(
                bandit_feedback=validation_bandit_data,
                ope_estimators=[
                    IPS(estimator_name="IPS (estimated)", use_estimated_pscore=True), 
                    DM(estimator_name="DM"),  
                    DR(estimator_name="DR (estimated)", use_estimated_pscore=True),
                    case1,
                    case2,
                    case3,
                ]
            )

            estimated_policy_values = ope.estimate_policy_values(
                #ground_truth_policy_value=value_of_ipw, # V(\pi)
                action_dist=pi_val, # \pi(a|x)
                estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
                estimated_pscore=estimated_pscore,
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
        result_df["pi_b_noise"] = pi_b_noise
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
    result_df.to_csv("estimate_pi_b_val_noise.csv")



if __name__ == "__main__":
    main()