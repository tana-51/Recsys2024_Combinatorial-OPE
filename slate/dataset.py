
import obp
from dataclasses import dataclass
from itertools import permutations
from itertools import product
import itertools
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.special import logit
from scipy.special import perm
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from tqdm import tqdm

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid
from obp.utils import softmax
from obp.dataset.base import BaseBanditDataset


def generate_combinations(current_combination, n): 
    if len(current_combination) == n:
        return [current_combination]

    combinations = []
    combinations.extend(generate_combinations(current_combination + [0], n))
    combinations.extend(generate_combinations(current_combination + [1], n))

    return combinations

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):

    n_unique_action: int
    dim_context: int = 5
    reward_type: str = "binary"
    base_reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    is_factorizable: bool = False
    random_state: int = 12345
    dataset_name: str = "synthetic_slate_bandit_dataset"
    interaction_factor: float = 2.0
    reward_std: float = 1.0
    beta: float = -0.5
    slate_size : int = 4

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_unique_action, "n_unique_action", int, min_val=1)
        if self.is_factorizable:
            max_len_list = None
        else:
            max_len_list = self.n_unique_action

        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(self.interaction_factor, "interaction_factor", float, min_val=0.0)
        check_scalar(self.reward_std, "reward_std", float, min_val=0.0)
        check_scalar(self.beta, "beta", float)
        self.random_ = check_random_state(self.random_state)
        
        if self.reward_type not in [
            "binary",
            "continuous",
        ]:
            raise ValueError(
                f"`reward_type` must be either 'binary' or 'continuous', but {self.reward_type} is given."
            )
        
        
        if self.base_reward_function is not None:
            self.reward_function = action_interaction_reward_function
            
        if self.behavior_policy_function is None:
            self.uniform_behavior_policy = (
                np.ones(self.n_unique_action) / self.n_unique_action
            )
        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10

        
        
        #self.action_context 
        elements = np.arange(self.n_unique_action)
        self.action_context = np.array(list(product(elements, repeat=self.slate_size)))
        
        # n_slate
        self.n_slate = self.action_context.shape[0]
    

   

    def sample_action_and_obtain_pscore(
        self,
        context,
        behavior_policy_logit_: np.ndarray,
        n_rounds: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        
        action = np.zeros(n_rounds*self.slate_size , dtype=int).reshape(n_rounds,self.slate_size)
        slate_id = np.zeros(n_rounds , dtype=int)
        pscore = np.zeros(n_rounds)
        pscore_position_wise = np.zeros(n_rounds*self.slate_size).reshape(n_rounds,self.slate_size)
        all_pscore = np.zeros(n_rounds*self.n_slate).reshape(n_rounds,self.n_slate)
        
        for i in np.arange(n_rounds):    
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(self.beta *behavior_policy_logit_)
            pscore_i = 1.0
            
            for l in range(self.slate_size):
                sampled_action = self.random_.choice(
                    unique_action_set, p=score_[i,:,l], replace=False
                )
                action[i,l] = sampled_action
                pscore_position_wise[i,l] = score_[i,sampled_action,l]
                
            slate_id[i] = np.where(np.all(self.action_context == action[i,:],axis=1))[0][0]
            pscore[i] = np.prod(pscore_position_wise[i])
            
            # calculate pscore_OPCB
            for s in range(self.n_slate):
                all_pscore[i,s] = np.prod(score_[i,self.action_context[s],np.arange(self.slate_size)])
            
        return action, slate_id, pscore, pscore_position_wise, all_pscore


    def sample_reward_given_expected_reward(
        self, expected_reward_factual: np.ndarray
    ) -> np.ndarray:
        
        if self.reward_type == "binary":
            sampled_reward_list = list()
            sampled_rewards = np.zeros(expected_reward_factual.shape[1])
            for i in np.arange(expected_reward_factual.shape[1]):
                sampled_rewards = self.random_.binomial(
                    n=1, p=expected_reward_factual[0,i]
                )
                sampled_reward_list.append(sampled_rewards)
            reward = np.array(sampled_reward_list)

        elif self.reward_type == "continuous":
            reward = np.zeros(expected_reward_factual.shape)
            
            mean = expected_reward_factual

            reward[0,:] = self.random_.normal(
                loc=mean,
                scale=self.reward_std,
                
            )
            
            reward = np.array(reward[0,:])
        else:
            raise NotImplementedError
        # return: array-like, shape (n_rounds, len_list)
        return reward

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        n_users: int = 200,
    ) -> BanditFeedback:
        
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        check_scalar(n_users, "n_users", int, min_val=1)
        
        fixed_user_contexts = self.random_.normal(size=(n_users, self.dim_context))
        user_idx = self.random_.choice(n_users, size=n_rounds)
        context = fixed_user_contexts[user_idx]
        
        # sample expected reward factual matrix q(x,m)
        if self.base_reward_function is None:
            self.expected_reward_factual_matrix = self.sample_contextfree_expected_reward(
                n_rounds=n_rounds,
                reward_type=self.reward_type,
                random_state=self.random_state,
            )
          
        else:
            self.expected_reward_factual_matrix, self.individual_reward = self.reward_function(
                context=context,
                action_context=self.action_context,
                n_slate=self.n_slate,
                slate_size=self.slate_size,
                #action=action,
                n_unique_action=self.n_unique_action,
                interaction_factor=self.interaction_factor,
                base_reward_function=self.base_reward_function,
                reward_type=self.reward_type,
                n_users=n_users,
                user_idx=user_idx,
                random_state=self.random_state,
            )
            
            
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            if self.reward_type == "binary":
                behavior_policy_logit_ = self.individual_reward #sigmoid(self.individual_reward)
            else:
                behavior_policy_logit_ = self.individual_reward
        else:
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # check the shape of behavior_policy_logit_
        if not (
            isinstance(behavior_policy_logit_, np.ndarray)
            and behavior_policy_logit_.shape == (n_rounds, self.n_unique_action, self.slate_size)
        ):
            raise ValueError("`behavior_policy_logit_` has an invalid shape")
            
        # sample actions and calculate the three variants of the propensity scores
        (
            action,
            slate_id,
            pscore,
            pscore_position_wise,
            all_pscore,
        ) = self.sample_action_and_obtain_pscore(
            context=context,
            behavior_policy_logit_=behavior_policy_logit_,
            n_rounds=n_rounds,
        )
        
        
        all_expected_reward = np.zeros(n_rounds)
        for i in range(n_rounds):
            all_expected_reward[i] = self.expected_reward_factual_matrix[i,slate_id[i]]


        expected_reward_factual = np.zeros_like(slate_id, dtype="float16")


        if self.reward_type == "binary":
            expected_reward_factual = all_expected_reward
        else:
            expected_reward_factual = all_expected_reward

        expected_reward_factual = np.array([expected_reward_factual])

        assert expected_reward_factual.shape == (
            1,n_rounds
        ), f"response shape must be (n_rounds (* enumerated_slate_actions), len_list), but {expected_reward_factual.shape}"
        
        # check the shape of expected_reward_factual
        if not (
            isinstance(expected_reward_factual, np.ndarray)
            and expected_reward_factual.shape == (1,n_rounds)
        ):
            raise ValueError("`expected_reward_factual` has an invalid shape")
        
        # sample reward
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )
        
        reward_matrix = np.zeros((n_users, self.n_slate))
        obs_action_matrix = np.zeros((n_users, self.n_slate))
        for u, s, r in zip(user_idx, slate_id, reward):
            reward_matrix[u, s] = r
            obs_action_matrix[u,s] = 1
       
        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            n_slate=self.n_slate,
            slate_size=self.slate_size,
            n_users=n_users,
            interaction_factor=self.interaction_factor,
            user_idx=user_idx,
            fixed_user_contexts=fixed_user_contexts,
            context=context,
            action_context=self.action_context,
            action=action,
            slate_id=slate_id,
            position=None,
            reward=reward.reshape(action.shape[0]),
            reward_matrix=reward_matrix,
            obs_action_matrix=obs_action_matrix,
            expected_reward_factual=expected_reward_factual.reshape(action.shape[0]),
            expected_reward_matrix=self.expected_reward_factual_matrix, #q(x,s)
            individual_reward=self.individual_reward, #q(x,a)
            pscore=pscore,
            pscore_position_wise=pscore_position_wise,
            all_pscore=all_pscore,
        )

       

    def calc_on_policy_policy_value(
        self, reward: np.ndarray, slate_id: np.ndarray
    ) -> float:
        
        check_array(array=slate_id, name="slate_id", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        if reward.shape[0] != slate_id.shape[0]:
            raise ValueError(
                "Expected `reward.shape[0] == slate_id.shape[0]`, but found it False"
            )

        return reward.sum() / np.unique(slate_id).shape[0]
    

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )
        
        return np.average(expected_reward, weights=action_dist[:, :, 0],axis=1).mean()


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    n_slate : int,
    slate_size : int, 
    #action: np.ndarray,
    n_unique_action:int,
    interaction_factor:float,
    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reward_type: str,
    n_users: int,
    user_idx: np.ndarray,
    is_enumerated: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
  
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)
    #check_array(array=action, name="action", expected_dim=1)
   
    if reward_type not in [
        "binary",
        "continuous",
    ]:
        raise ValueError(
            f"`reward_type` must be either 'binary' or 'continuous', but {reward_type} is given."
        )
    
    n_rounds = context.shape[0]
    random_ = check_random_state(random_state)
    
    n_use_item = np.zeros(n_slate)
    n_use_item[0] = 1
    for i in range(n_slate-1):
        n_use_item[i+1] = len(action_context[i+1,:][np.nonzero(action_context[i+1,:])])
    
    # q_x_a 
    q_x_a = np.zeros(n_rounds*(n_unique_action)*slate_size).reshape(n_rounds,n_unique_action,slate_size)
    
    for l in range(slate_size):
        #a = random_.normal(loc=0,scale=3.0,size=(n_unique_action+1, 5))
        a = np.eye(n_unique_action)
        q_x_a[:,:,l] = base_reward_function(
            context=context, action_context=a, random_state=12345
        )
    
    #q_x_s_linear
    q_x_s_linear = np.zeros(n_rounds*n_slate).reshape(n_rounds,n_slate)
    
    for i in range(n_rounds):
        q_x_s_linear[i,:] = q_x_a[i, action_context, np.arange(slate_size)].sum(axis=1)
    
    q_x_s_linear = q_x_s_linear / n_use_item
    

    # q_x_s
    q_x_s = q_x_s_linear 
    
    if reward_type == "binary":
        expected_reward_factual_matrix = sigmoid(q_x_s)
    else:
        expected_reward_factual_matrix = q_x_s
        
    return expected_reward_factual_matrix, q_x_a

