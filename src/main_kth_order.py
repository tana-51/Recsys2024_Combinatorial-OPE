from omegaconf import DictConfig, OmegaConf
import hydra
import os

#kth_order
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

# import open bandit pipeline (obp)
import obp
from obp.dataset import (
    #SyntheticBanditDataset,
    #SyntheticSlateBanditDataset,
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
from estimators import WOPCB 
from ope import OffPolicyEvaluation


@hydra.main(config_path="../conf",config_name="config_kth_order")
def main(cfg: DictConfig) -> None:
    ## 実験設定
    num_runs = cfg.num_runs #できるだけ大きい方が良い
    num_data = cfg.num_data
    num_unique_action = cfg.num_unique_action
    kth_order_list = cfg.kth_order_list
    unique_action_set = [i for i in range(num_unique_action)]

    result_df_list = []
    
    np.random.seed(12345)

    ## 人工データ生成クラス
    dataset = SyntheticCombinatorialBanditDataset(
        n_unique_action=num_unique_action,
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
    
    ### 評価対象の意思決定方策の真の性能(value)を近似するためのデータ
    test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.n_rounds_test_bandit_data,n_users=cfg.n_users,true_element=cfg.true_element)


    ## 評価対象の意思決定方策の真の性能(value)をテストデータ上で計算
    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=test_bandit_data["expected_reward_matrix"], 
        action_dist=gen_eps_greedy(
                    expected_reward=test_bandit_data["expected_reward_matrix"],
                    eps=cfg.eps,
                ),
    )

    

    
    for kth_order in kth_order_list:
        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"kth_order={kth_order}..."):

            kth_element = np.random.choice(unique_action_set, size=kth_order, replace=False)
            kth_element = list(kth_element)

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
            
            
            case1=OPCB(estimator_name="opcb")
            case1.set_(kth_element = kth_element, bandit_data = validation_bandit_data)


            ope = OffPolicyEvaluation(
                bandit_feedback=validation_bandit_data,
                ope_estimators=[
                    IPS(estimator_name="IPS"), 
                    DM(estimator_name="DM"),  
                    DR(estimator_name="DR"),
                    case1,
                ]
            )


            estimated_policy_values = ope.estimate_policy_values(
                #ground_truth_policy_value=value_of_ipw, # V(\pi)
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
        result_df["kth_order"] = kth_order
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
    result_df.to_csv("kth_order.csv")

    
if __name__ == "__main__":
    main()
