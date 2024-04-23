# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns
from sklearn.utils import check_scalar

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import check_confidence_interval_arguments
from obp.ope.estimators import BaseOffPolicyEstimator
from obp.ope.estimators import DirectMethod as DM
from obp.ope.estimators import DoublyRobust as DR


logger = getLogger(__name__)


@dataclass
class OffPolicyEvaluation:
  

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in ["action", "position", "reward", "context"]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
        self.ope_estimators_ = dict()
        self.is_model_dependent = False
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator
            if isinstance(estimator, DM) or isinstance(estimator, DR):
                self.is_model_dependent = True

    def _create_estimator_inputs(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value using subclasses of `BaseOffPolicyEstimator`"""
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if estimated_rewards_by_reg_model is None:
            pass
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                check_array(
                    array=value,
                    name=f"estimated_rewards_by_reg_model[{estimator_name}]",
                    expected_dim=3,
                )
                if value.shape != action_dist.shape:
                    raise ValueError(
                        f"Expected `estimated_rewards_by_reg_model[{estimator_name}].shape == action_dist.shape`, but found it False."
                    )
        else:
            check_array(
                array=estimated_rewards_by_reg_model,
                name="estimated_rewards_by_reg_model",
                expected_dim=3,
            )
            if estimated_rewards_by_reg_model.shape != action_dist.shape:
                raise ValueError(
                    "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
                )
        for var_name, value_or_dict in {
            "estimated_pscore": estimated_pscore,
            "estimated_importance_weights": estimated_importance_weights,
            "action_embed": action_embed,
            "pi_b": pi_b,
            "p_e_a": p_e_a,
        }.items():
            if value_or_dict is None:
                pass
            elif isinstance(value_or_dict, dict):
                for estimator_name, value in value_or_dict.items():
                    expected_dim = 1
                    if var_name in ["p_e_a", "pi_b"]:
                        expected_dim = 3
                    elif var_name in ["action_embed"]:
                        expected_dim = 2
                    check_array(
                        array=value,
                        name=f"{var_name}[{estimator_name}]",
                        expected_dim=expected_dim,
                    )
                    if var_name != "p_e_a":
                        if value.shape[0] != action_dist.shape[0]:
                            raise ValueError(
                                f"Expected `{var_name}[{estimator_name}].shape[0] == action_dist.shape[0]`, but found it False"
                            )
                    else:
                        if value.shape[0] != action_dist.shape[1]:
                            raise ValueError(
                                f"Expected `{var_name}[{estimator_name}].shape[0] == action_dist.shape[1]`, but found it False"
                            )
            else:
                expected_dim = 1
                if var_name in ["p_e_a", "pi_b"]:
                    expected_dim = 3
                elif var_name in ["action_embed"]:
                    expected_dim = 2
                check_array(
                    array=value_or_dict, name=var_name, expected_dim=expected_dim
                )
                if var_name != "p_e_a":
                    if value_or_dict.shape[0] != action_dist.shape[0]:
                        raise ValueError(
                            f"Expected `{var_name}.shape[0] == action_dist.shape[0]`, but found it False"
                        )
                else:
                    if value.shape[0] != action_dist.shape[1]:
                        raise ValueError(
                            f"Expected `{var_name}[{estimator_name}].shape[0] == action_dist.shape[1]`, but found it False"
                        )

        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["action", "position", "reward", "context"]
            }
            for estimator_name in self.ope_estimators_
        }

        for estimator_name in self.ope_estimators_:
            if "pscore" in self.bandit_feedback:
                estimator_inputs[estimator_name]["pscore"] = self.bandit_feedback[
                    "pscore"
                ]
            else:
                estimator_inputs[estimator_name]["pscore"] = None
            estimator_inputs[estimator_name]["action_dist"] = action_dist
            estimator_inputs = self._preprocess_model_based_input(
                estimator_inputs=estimator_inputs,
                estimator_name=estimator_name,
                model_based_input={
                    "estimated_rewards_by_reg_model": estimated_rewards_by_reg_model,
                    "estimated_pscore": estimated_pscore,
                    "estimated_importance_weights": estimated_importance_weights,
                    "action_embed": action_embed,
                    "pi_b": pi_b,
                    "p_e_a": p_e_a,
                },
            )
        return estimator_inputs

    def _preprocess_model_based_input(
        self,
        estimator_inputs: Dict[str, Optional[np.ndarray]],
        estimator_name: str,
        model_based_input: Dict[
            str, Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        ],
    ) -> Dict[str, Optional[np.ndarray]]:
        for var_name, value_or_dict in model_based_input.items():
            if isinstance(value_or_dict, dict):
                if estimator_name in value_or_dict:
                    estimator_inputs[estimator_name][var_name] = value_or_dict[
                        estimator_name
                    ]
                else:
                    estimator_inputs[estimator_name][var_name] = None
            else:
                estimator_inputs[estimator_name][var_name] = value_or_dict
        return estimator_inputs

    def estimate_policy_values(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, float]:
       
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        policy_value_dict = dict()
#         estimator_inputs = self._create_estimator_inputs(
#             action_dist=action_dist,
#             estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
#             estimated_pscore=estimated_pscore,
#             estimated_importance_weights=estimated_importance_weights,
#             action_embed=action_embed,
#             pi_b=pi_b,
#             p_e_a=p_e_a,
#         )
            
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                reward=self.bandit_feedback["reward"],
                action=self.bandit_feedback["slate_id"],
                action_context=self.bandit_feedback["action_context"],
                pscore=self.bandit_feedback["pscore"],
                all_pscore=self.bandit_feedback["all_pscore"],
                #kth_element=self.bandit_feedback["kth_element"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                #estimated_rewards_by_two_step=estimated_rewards_by_two_step,
                position= None,
            )
            
#         for estimator_name, estimator in self.ope_estimators_.items():
#             policy_value_dict[estimator_name] = estimator.estimate_policy_value(
#                 **estimator_inputs[estimator_name]
#             )
            
        return policy_value_dict

    def estimate_intervals(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
       
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
            action_embed=action_embed,
            pi_b=pi_b,
            p_e_a=p_e_a,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs[estimator_name],
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
       
        policy_value_df = DataFrame(
            self.estimate_policy_values(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_of_behavior_policy = self.bandit_feedback["reward"].mean()
        policy_value_df = policy_value_df.T
        if policy_value_of_behavior_policy <= 0:
            logger.warning(
                f"Policy value of the behavior policy is {policy_value_of_behavior_policy} (<=0); relative estimated policy value is set to np.nan"
            )
            policy_value_df["relative_estimated_policy_value"] = np.nan
        else:
            policy_value_df["relative_estimated_policy_value"] = (
                policy_value_df.estimated_policy_value / policy_value_of_behavior_policy
            )
        return policy_value_df, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
       
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "`fig_dir` must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "`fig_dir` must be a string"

        estimated_round_rewards_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
            action_embed=action_embed,
            pi_b=pi_b,
            p_e_a=p_e_a,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_round_rewards_dict[
                estimator_name
            ] = estimator._estimate_round_rewards(**estimator_inputs[estimator_name])
        estimated_round_rewards_df = DataFrame(estimated_round_rewards_dict)
        estimated_round_rewards_df.rename(
            columns={key: key.upper() for key in estimated_round_rewards_dict.keys()},
            inplace=True,
        )
        if is_relative:
            estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_round_rewards_df,
            ax=ax,
            ci=100 * (1 - alpha),
            n_boot=n_bootstrap_samples,
            seed=random_state,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int32(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        #estimated_rewards_by_two_step: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        metric: str = "se",
    ) -> Dict[str, float]:
       
        check_scalar(
            ground_truth_policy_value,
            "ground_truth_policy_value",
            float,
        )
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"`metric` must be either 'relative-ee' or 'se', but {metric} is given"
            )
        if metric == "relative-ee" and ground_truth_policy_value == 0.0:
            raise ValueError(
                "`ground_truth_policy_value` must be non-zero when metric is relative-ee"
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
            action_embed=action_embed,
            pi_b=pi_b,
            p_e_a=p_e_a,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(
                reward=self.bandit_feedback["reward"],
                action=self.bandit_feedback["action"],
                action_context=self.bandit_feedback["action_context"],
                pscore=self.bandit_feedback["pscore"],
                all_pscore=self.bandit_feedback["all_pscore"],
                #kth_element=self.bandit_feedback["kth_element"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                #estimated_rewards_by_two_step=estimated_rewards_by_two_step,
                position= None,
            )
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        metric: str = "se",
    ) -> DataFrame:
       
        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T

    def visualize_off_policy_estimates_of_multiple_policies(
        self,
        policy_name_list: List[str],
        action_dist_list: List[np.ndarray],
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
       
        if len(policy_name_list) != len(action_dist_list):
            raise ValueError(
                "the length of `policy_name_list` must be the same as `action_dist_list`"
            )
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "`fig_dir` must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "`fig_dir` must be a string"

        estimated_round_rewards_dict = {
            estimator_name: {} for estimator_name in self.ope_estimators_
        }

        for policy_name, action_dist in zip(policy_name_list, action_dist_list):
            estimator_inputs = self._create_estimator_inputs(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
            )
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_round_rewards_dict[estimator_name][
                    policy_name
                ] = estimator._estimate_round_rewards(
                    **estimator_inputs[estimator_name]
                )

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(8, 6.2 * len(self.ope_estimators_)))

        for i, estimator_name in enumerate(self.ope_estimators_):
            estimated_round_rewards_df = DataFrame(
                estimated_round_rewards_dict[estimator_name]
            )
            if is_relative:
                estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

            ax = fig.add_subplot(len(action_dist_list), 1, i + 1)
            sns.barplot(
                data=estimated_round_rewards_df,
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            ax.set_title(estimator_name.upper(), fontsize=20)
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int32(100*(1 - alpha))}% CI)",
                fontsize=20,
            )
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=25 - 2 * len(policy_name_list))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))
 
