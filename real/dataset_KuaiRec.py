from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from obp.dataset import BaseRealBanditDataset
from obp.utils import check_array
from obp.utils import sample_action_fast
from obp.utils import softmax
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from scipy.stats import gmean


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def generate_combinations(current_combination, n): #000もあり
    if len(current_combination) == n:
        return [current_combination]

    combinations = []
    combinations.extend(generate_combinations(current_combination + [0], n))
    combinations.extend(generate_combinations(current_combination + [1], n))

    return combinations



@dataclass
class ExtremeBanditDataset():
    n_components: int = 10
    reward_std: float = 1.0
    n_unique_action: int = 13
    n_train: int = 1500
    random_state: int = 12345

    def __post_init__(self):
        self.data_path = Path(r"/home/kouichi2/CA_Combinatorial-OPE_new/real/data/kuairec")
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        
        self.random_ = check_random_state(self.random_state)
        self.n_comb_action = 2**self.n_unique_action
        self.action_context = np.array(generate_combinations([], self.n_unique_action))
        self.big_matrix, self.train_data, self.small_matrix, self.contexts = self.pre_process(self.data_path)

        # train a classifier to define a logging policy
        self.train_pi_b()

    def pre_process(
        self, file_path: Path, 
    ) -> Tuple[int, int, int, coo_matrix, coo_matrix]:
        """Preprocess raw dataset."""
        df_big_matrix = pd.read_csv(self.data_path / "big_matrix.csv")
        df_small_matrix = pd.read_csv(self.data_path / "small_matrix.csv")
        df_user_feature = pd.read_csv(self.data_path / "user_features.csv")
        
        # small_matrix
        small_user_id = df_small_matrix["user_id"]
        user_idx = list(set(small_user_id))
        user_idx = sorted(user_idx)
        
        small_video_id = df_small_matrix["video_id"]
        video_idx = list(set(small_video_id))
        video_idx = sorted(video_idx)
        
        small_watch_ratio = df_small_matrix["watch_ratio"]

        small_matrix = np.zeros(7176*10728).reshape(7176,10728)

        small_matrix[small_user_id,small_video_id] = small_watch_ratio
        
        use_video_id = sorted(self.random_.choice(video_idx, self.n_unique_action, replace=False))
        small_matrix = small_matrix[user_idx,:]
        small_matrix = small_matrix[:,use_video_id]
        

        # big_matrix
        big_video_id = df_big_matrix["video_id"]
        big_user_id = df_big_matrix["user_id"]
        big_watch_ratio = df_big_matrix["watch_ratio"]

        big_matrix = np.zeros(7176*10728).reshape(7176,10728)

        big_matrix[big_user_id,big_video_id] = big_watch_ratio
        
        big_matrix = big_matrix[:,use_video_id]
        big_matrix = np.delete(big_matrix, user_idx, axis=0)
        
        train_data, big_matrix = np.split(big_matrix, [self.n_train])
        
        # contexts
        delete_column = df_user_feature.columns.values[df_user_feature.dtypes == object]
        df_user_feature = df_user_feature.drop(delete_column, axis=1)
        df_user_feature = df_user_feature.dropna()
        contexts = np.array(df_user_feature.iloc[:,1:])
        
        return big_matrix, train_data, small_matrix, contexts
        
        

    def train_pi_b(
        self,
    ) -> None:
        idx = self.random_.choice(self.contexts.shape[0], size=self.n_train, replace=False)
        contexts = self.contexts[idx]
        contexts = self.sc.fit_transform(self.pca.fit_transform(contexts))
        idx = self.random_.choice(self.n_train, size=self.n_train, replace=False)
        expected_rewards = self.train_data[idx]
        #expected_rewards = sigmoid(expected_rewards)
        
        # generate combination rewards and sample action
        rewards = np.zeros(self.n_train)
        actions = np.zeros(self.n_train , dtype=int)
        
        expected_rewards_  = expected_rewards.copy()
        expected_rewards_[expected_rewards_!=0] = 1

        
        for i in range(expected_rewards.shape[0]):
            x = expected_rewards[i][np.nonzero(expected_rewards[i])]
            if len(x)==0:
                x = np.array([0])
            rewards[i] = gmean(x) 
            sampled_action = np.where(np.all(self.action_context == expected_rewards_[i],axis=1))
            actions[i] = sampled_action[0][0]
        
        q_x_m = np.zeros(self.n_train*self.n_comb_action).reshape(self.n_train,self.n_comb_action)
        q_x_m[np.arange(self.n_train),actions] = rewards
        
        noise = self.random_.uniform(0.0, 1.0, size=(self.n_train,self.n_comb_action))
       
        q_x_m += noise

        self.regressor = MultiOutputRegressor(Ridge(max_iter=500, random_state=12345))
        self.regressor.fit(contexts, q_x_m)


    def compute_pi_b(
        self,
        contexts: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        r_hat = self.regressor.predict(contexts)
        pi_b = softmax(r_hat * beta)
        return pi_b

    def obtain_batch_bandit_feedback(
        self, n_rounds: Optional[int] = None, n_users: int = 300, beta: float = 10, reward_std :float = 3.0,
    ) -> dict:
        """Obtain batch logged bandit data."""
        
        idx = self.random_.choice(self.contexts.shape[0], size=n_users, replace=False)
        fixed_user_contexts = self.contexts[idx]
        fixed_user_contexts = self.sc.fit_transform(
            self.pca.fit_transform(fixed_user_contexts)
        )
        
        user_idx = self.random_.choice(n_users, size=n_rounds)
        contexts = fixed_user_contexts[user_idx]
        
        
        fixed_q_x_m = self.obtain_expected_reward_matrix()
        q_x_m = fixed_q_x_m[user_idx,:]
        
        
        #sample actions
        actions = np.zeros(n_rounds , dtype=int)
        pi_b = self.compute_pi_b(contexts, beta=beta)
        
        for i in np.arange(n_rounds):    
            unique_action_set = np.arange(self.n_comb_action)
            score_ = pi_b[i]
            
            sampled_action = self.random_.choice(
                unique_action_set, p=score_, replace=False
            )
            actions[i] = sampled_action
        
        
        q_x_m_factual = q_x_m[np.arange(n_rounds), actions]

        rewards = self.random_.normal(
                loc=q_x_m_factual,
                scale=reward_std,
            )
        

        reward_matrix = np.zeros((n_users, self.n_comb_action))
        obs_action_matrix = np.zeros((n_users, self.n_comb_action))
        for u, a, r in zip(user_idx, actions, rewards):
            reward_matrix[u, a] = r
            obs_action_matrix[u,a] = 1

        return dict(
            n_rounds=n_rounds,
            n_users=n_users,
            user_idx=user_idx,
            n_comb_action=self.n_comb_action,
            action_context=self.action_context,
            context=contexts,
            fixed_user_contexts=fixed_user_contexts,
            action=actions,
            position=None,
            reward=rewards,
            reward_matrix=reward_matrix,
            obs_action_matrix=obs_action_matrix,
            #expected_reward_matrix=q_x_m,
            #expected_reward_factual=q_x_m_factual,
            all_pscore=pi_b,
            pscore=pi_b[np.arange(n_rounds), actions],
        )
    
    def obtain_expected_reward_matrix(self,):
        
        expected_reward_matrix = np.zeros(self.small_matrix.shape[0]*self.n_comb_action).reshape(self.small_matrix.shape[0],self.n_comb_action)
        for i in range(self.small_matrix.shape[0]):
            x = self.small_matrix[i,:]*self.action_context
            prod = np.prod(np.where(x[1:,:] == 0, 1, x[1:,:]), axis=1)
            sum_ = np.sum(np.where(x[1:,:] == 0, 1, x[1:,:]), axis=1)
            idx = np.count_nonzero(self.action_context[1:,:], axis=1)
            expected_reward_matrix[i, 1:] = prod** (1 / idx)
            
        return expected_reward_matrix
    
    @staticmethod
    def calc_ground_truth_policy_value(
        expected_reward: np.ndarray, action_dist: np.ndarray
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

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()