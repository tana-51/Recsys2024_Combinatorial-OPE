"""Off-Policy Estimators."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from obp.ope import (
    BaseOffPolicyEstimator,
    OffPolicyEvaluation, 
    RegressionModel,
)

from obp.ope import (
    InverseProbabilityWeighting as IPS
)

import obp
import numpy as np
from sklearn.utils import check_scalar

from obp.utils import check_array
from obp.utils import check_ope_inputs
from obp.utils import estimate_confidence_interval_by_bootstrap
from obp.ope.helper import estimate_bias_in_ope
from obp.ope.helper import estimate_high_probability_upper_bound_bias
from itertools import combinations
from learner import train_reward_model_via_two_stage
import optuna
import logging


    
def objective(trial,estimated_value,estimate_reward_list,opcb_variance_list):
    alpha = np.zeros(len(estimate_reward_list))
    for i in range(len(estimate_reward_list)):
        alpha[i] = trial.suggest_float('x{}'.format(i), 0, 1)
  
    sum_ = np.sum(alpha)
    alpha = alpha/sum_

    wopcb = np.average(estimate_reward_list, weights=alpha)

    wopcb_variance = (np.array(opcb_variance_list)*(alpha**2)).sum()
    wopcb_squared_bias = (estimated_value-wopcb)**2

    wopcb_mse = wopcb_squared_bias + wopcb_variance
    return wopcb_mse

def objective_true(trial,true_value,estimate_reward_list,):
    alpha = np.zeros(len(estimate_reward_list))
    for i in range(len(estimate_reward_list)):
        alpha[i] = trial.suggest_float('x{}'.format(i), 0, 1)
  
    sum_ = np.sum(alpha)
    alpha = alpha/sum_

    wopcb = np.average(estimate_reward_list, weights=alpha)

    wopcb_mse = (true_value - wopcb)**2

    return wopcb_mse
  

@dataclass
class OPCB(BaseOffPolicyEstimator):
   
    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "opcb"
    use_q_hat: bool = False
    estimated_rewards_by_reg_model: float = 0
    

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        
        
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )
    
    def set_(self, kth_element,bandit_data,**kwargs):
        self.kth_element = kth_element
        self.bandit_data=bandit_data
    
    def obtain_importance_weight_for_opcb(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        pscore: np.ndarray,
        all_pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        
        kth_action_context = action_context[:,self.kth_element]
        
        id_list = []
        for i in action:
            id_ = np.where(np.all(kth_action_context==kth_action_context[i,:],axis=1))
            id_list.append(id_[0].tolist())
        
        pscore_OPCB = np.zeros(n)
        action_dist_OPCB = np.zeros(n)

        pos_ = 0
        for i in id_list:
            pscore_OPCB[pos_]=all_pscore[pos_,i].sum()
            action_dist_OPCB[pos_] = action_dist[pos_,i].sum()
            pos_ += 1
        
        iw_opcb = action_dist_OPCB / pscore_OPCB
        
        return iw_opcb

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        pscore: np.ndarray,
        all_pscore: np.ndarray,
        #kth_element: list,
        action_dist: np.ndarray,
        #estimated_rewards_by_reg_model: np.ndarray,
        #estimated_rewards_by_two_step:np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        
        kth_action_context = action_context[:,self.kth_element]
        
        id_list = []
        for i in action:
            id_ = np.where(np.all(kth_action_context==kth_action_context[i,:],axis=1))
            id_list.append(id_[0].tolist())
        
        pscore_OPCB = np.zeros(n)
        action_dist_OPCB = np.zeros(n)

        pos_ = 0
        for i in id_list:
            pscore_OPCB[pos_]=all_pscore[pos_,i].sum()
            action_dist_OPCB[pos_] = action_dist[pos_,i].sum()
            pos_ += 1
        
        iw = action_dist_OPCB / pscore_OPCB
        
        action_dist_=action_dist.reshape(n,action_context.shape[0])
        
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        
        if self.use_q_hat:
            f_hat = self.estimated_rewards_by_reg_model[:,:,0]
        else:
            f_hat = train_reward_model_via_two_stage(
                    bandit_data = self.bandit_data,
                    kth_element = self.kth_element,
        )
        

        q_hat_at_position = f_hat[np.arange(n), :]
        q_hat_factual = f_hat[np.arange(n), action]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = iw * (reward - q_hat_factual)
        estimated_rewards += np.average(
            q_hat_at_position,
            weights=action_dist_,
            axis=1,
        )

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        action_dist: np.ndarray,
        #estimated_rewards_by_reg_model: np.ndarray,
        #estimated_rewards_by_two_step: np.ndarray,
        all_pscore : np.ndarray,
        #kth_element : list,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
       
        
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            #estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            action_context=action_context,
            position=position,
            pscore=pscore_,
            all_pscore=all_pscore,
            #kth_element=kth_element,
            action_dist=action_dist,
            #estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            #estimated_rewards_by_two_step=estimated_rewards_by_two_step,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
      
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
    ) -> float:
       
        n = reward.shape[0]
        
        #calculate OPCB_action_dist
        
        # estimate the sample variance of DR with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                action_context=action_context,
                pscore=pscore,
                all_pscore=all_pscore,
                action_dist=action_dist,
                #estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )

        sample_variance /= n

        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score
    
    
##-----------------------------------------------------------------------------------------------------------------------------##

@dataclass
class WOPCB(BaseOffPolicyEstimator):
   
    
    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "wopcb"
    use_true_value: bool = False
    true_value: float = 0
    use_q_hat: bool = False
    estimated_rewards_by_reg_model: float = 0
    

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        
        
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )
    
    def set_(self, kth_order,bandit_data,**kwargs):
        self.kth_order = kth_order
        self.bandit_data = bandit_data
      


    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        pscore: np.ndarray,
        all_pscore: np.ndarray,
        kth_element: list,
        action_dist: np.ndarray,
        #estimated_rewards_by_reg_model: np.ndarray,
        #estimated_rewards_by_two_step:np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
       
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        
        kth_action_context = action_context[:,kth_element]
        
        id_list = []
        for i in action:
            id_ = np.where(np.all(kth_action_context==kth_action_context[i,:],axis=1))
            id_list.append(id_[0].tolist())
        
        pscore_OPCB = np.zeros(n)
        action_dist_OPCB = np.zeros(n)

        pos_ = 0
        for i in id_list:
            pscore_OPCB[pos_]=all_pscore[pos_,i].sum()
            action_dist_OPCB[pos_] = action_dist[pos_,i].sum()
            pos_ += 1
        
        iw = action_dist_OPCB / pscore_OPCB
        
        action_dist_=action_dist.reshape(n,action_context.shape[0])
        
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        
        if self.use_q_hat:
            f_hat = self.estimated_rewards_by_reg_model[:,:,0]
        else:
            f_hat = train_reward_model_via_two_stage(
                    bandit_data = self.bandit_data,
                    kth_element = kth_element,
                )
        
        q_hat_at_position = f_hat[np.arange(n), :] 
        q_hat_factual = f_hat[np.arange(n), action]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = iw * (reward - q_hat_factual)
        estimated_rewards += np.average(
            q_hat_at_position,
            weights=action_dist_,
            axis=1,
        )

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        action_dist: np.ndarray,
        #estimated_rewards_by_reg_model: np.ndarray,
        #estimated_rewards_by_two_step: np.ndarray,
        all_pscore : np.ndarray,
        #kth_element : list,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
       
        
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            #estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        
        action_set = np.arange(action_context.shape[1])
        estimate_reward_list = []
        opcb_variance_list = []
        comb_list = list(combinations(action_set,self.kth_order))
        
        
        for i in comb_list:
            round_reward =  self._estimate_round_rewards(
                reward=reward,
                action=action,
                action_context=action_context,
                position=position,
                pscore=pscore_,
                all_pscore=all_pscore,
                kth_element=list(i),
                action_dist=action_dist,
                #estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            
            estimate_reward_list.append(round_reward.mean())
            opcb_variance_list.append(np.var(round_reward)/len(reward))
        
        alpha = np.zeros(len(estimate_reward_list))
        
        if self.use_true_value == True:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study()
            study.optimize(lambda trial: objective_true(trial,self.true_value,estimate_reward_list), n_trials=50)
            
            for i in range(len(estimate_reward_list)):
                alpha[i] = study.best_params["x{}".format(i)]
                
            alpha = alpha/np.sum(alpha)
                
        else:
            ips=IPS(estimator_name="IPS_opt")
            ips = ips.estimate_policy_value(
                                reward=reward,
                                action=action,
                                action_dist=action_dist,
                                pscore=pscore,
                            )

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study()
            study.optimize(lambda trial: objective(trial,ips,estimate_reward_list,opcb_variance_list), n_trials=50)

            for i in range(len(estimate_reward_list)):
                alpha[i] = study.best_params["x{}".format(i)]

            alpha = alpha/np.sum(alpha)
        
        

                
        return np.average(estimate_reward_list, weights=alpha)

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
    ) -> float:
        
        n = reward.shape[0]
        

        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n

        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)


#------------------------------------------------------------------------------------------------------------------------#

@dataclass
class SelfNormalizedOPCB(OPCB):
    
    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
        pscore: np.ndarray,
        all_pscore: np.ndarray,
        #kth_element: list,
        action_dist: np.ndarray,
        #estimated_rewards_by_reg_model: np.ndarray,
        #estimated_rewards_by_two_step:np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        
        kth_action_context = action_context[:,self.kth_element]
        
        id_list = []
        for i in action:
            id_ = np.where(np.all(kth_action_context==kth_action_context[i,:],axis=1))
            id_list.append(id_[0].tolist())
        
        pscore_OPCB = np.zeros(n)
        action_dist_OPCB = np.zeros(n)

        pos_ = 0
        for i in id_list:
            pscore_OPCB[pos_]=all_pscore[pos_,i].sum()
            action_dist_OPCB[pos_] = action_dist[pos_,i].sum()
            pos_ += 1
        
        iw = action_dist_OPCB / pscore_OPCB
        
        action_dist_=action_dist.reshape(n,action_context.shape[0])
        
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        
        if self.use_q_hat:
            f_hat = self.estimated_rewards_by_reg_model[:,:,0]
        else:
            f_hat = train_reward_model_via_two_stage(
                    bandit_data = self.bandit_data,
                    kth_element = self.kth_element,
        )
        

        q_hat_at_position = f_hat[np.arange(n), :]
        q_hat_factual = f_hat[np.arange(n), action]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = iw * (reward - q_hat_factual)/ iw.mean()
        estimated_rewards += np.average(
            q_hat_at_position,
            weights=action_dist_,
            axis=1,
        )

        return estimated_rewards

    
    
    
    
    
    
    
    
    
    
    
    
    