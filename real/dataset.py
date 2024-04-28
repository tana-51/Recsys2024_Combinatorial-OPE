from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.stats import gmean
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
from itertools import product

from obp.dataset import linear_reward_function


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def generate_combinations(current_combination, n): 
    if len(current_combination) == n:
        return [current_combination]

    combinations = []
    combinations.extend(generate_combinations(current_combination + [0], n))
    combinations.extend(generate_combinations(current_combination + [1], n))

    return combinations


@dataclass
class ExtremeBanditDataset(BaseRealBanditDataset):
    n_components: int = 10
    reward_std: float = 1.0
    max_reward_noise: float = 0.2
    dataset_name: str = "letter"  # letter or pen
    random_state: int = 12345

    def __post_init__(self):
        #self.data_path = Path().cwd().parents[1] / "data" / self.dataset_name
        self.data_path = Path(r"/home/kouichi2/CA_Combinatorial-OPE_new/real/data/"+ self.dataset_name)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        self.random_ = check_random_state(self.random_state)
        
        
        if self.dataset_name == "letter":
            self.process_for_letter()
            
                
        elif self.dataset_name == "pen":
            self.process_for_pen()
        
        self.load_raw_data()

        # train a classifier to define a logging policy
        self.train_pi_b()
    
    def process_for_letter(self) -> None:
        self.num_feat = 16
        self.num_label = 26
        self.num_data = 20000  #self.data.shape[0]
        unique_action_set = list(range(0, self.num_label))
        #self.random_.shuffle(unique_action_set)

        #grouping
        group1 = unique_action_set[:4]
        group2 = unique_action_set[4:9]
        group3 = unique_action_set[9:17]
        group4 = unique_action_set[17:22]
        group5 = unique_action_set[22:]

        self.n_comb_action = len(group1)*len(group2)*len(group3)*len(group4)*len(group5)

        self.alphabet_mapping = {}
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.alphabet_mapping[char] = i + 1
            
        self.action_context = np.zeros(self.n_comb_action * self.num_label).reshape(-1,self.num_label)
        pos_ = 0
        for a,b,c,d,e in product(group1, group2,group3,group4,group5):
            action = [a,b,c,d,e]
            self.action_context[pos_,action] = 1
            pos_ +=1
    
    def process_for_pen(self) -> None:
        self.num_feat = 16
        self.num_label = 10
        self.num_data =  7494 #self.data.shape[0]
        self.num_test_data = 3498
        self.n_comb_action = 2**(self.num_label)
        self.action_context = np.array(generate_combinations([], self.num_label))
        
        
    def load_raw_data(self) -> None:
        """Load raw dataset."""
        if self.dataset_name == "letter":
            self.data, self.label = self.pre_process(
                self.data_path / "data.data"
            )
            
                
        elif self.dataset_name == "pen":
            self.data, self.label = self.pre_process(
                self.data_path / "pendigits.tra"
            )
            
            self.test_data, self.test_label = self.pre_process_test(
                self.data_path / "pendigits.tes"
            )
        

    def pre_process(
        self, file_path: Path
    ) -> Tuple[int, int, int, coo_matrix, coo_matrix]:
        """Preprocess raw dataset."""
        data_file = open(file_path, "r")
     
        data, label = [], []
        
        if self.dataset_name == "letter":
            for i in range(self.num_data):
                raw_data_i = data_file.readline().split(",")
                label_pos_i = [int(self.alphabet_mapping[raw_data_i[0]])]
                data_pos_i = [int(x) for x in raw_data_i[1:]]
                label.append(
                    sparse.csr_matrix(
                        ([1.0] * len(label_pos_i), label_pos_i, [0, len(label_pos_i)]),
                        shape=(1, self.num_label),
                    )
                )
                data.append(
                    sparse.csr_matrix(
                        (data_pos_i, [i for i in range(self.num_feat)], [0, len(data_pos_i)]), shape=(1, self.num_feat)
                    )
                )
            data_file.close()
        
        elif self.dataset_name == "pen":
            for i in range(self.num_data):
                raw_data_i = data_file.readline().split(",")
                label_pos_i = [int(raw_data_i[-1])]
                data_pos_i = [int(x) for x in raw_data_i[:-1]]
                label.append(
                    sparse.csr_matrix(
                        ([1.0] * len(label_pos_i), label_pos_i, [0, len(label_pos_i)]),
                        shape=(1, self.num_label),
                    )
                )
                data.append(
                    sparse.csr_matrix(
                        (data_pos_i, [i for i in range(self.num_feat)], [0, len(data_pos_i)]), shape=(1, self.num_feat)
                    )
                )
            data_file.close()
        
        return sparse.vstack(data).toarray(), sparse.vstack(label).toarray()
    
    
    def pre_process_test(
        self, file_path: Path
    ) -> Tuple[int, int, int, coo_matrix, coo_matrix]:
        """Preprocess raw dataset."""
        data_file = open(file_path, "r")
     
        data, label = [], []
        
        if self.dataset_name == "pen":
            for i in range(self.num_test_data):
                raw_data_i = data_file.readline().split(",")
                label_pos_i = [int(raw_data_i[-1])]
                data_pos_i = [int(x) for x in raw_data_i[:-1]]
                label.append(
                    sparse.csr_matrix(
                        ([1.0] * len(label_pos_i), label_pos_i, [0, len(label_pos_i)]),
                        shape=(1, self.num_label),
                    )
                )
                data.append(
                    sparse.csr_matrix(
                        (data_pos_i, [i for i in range(self.num_feat)], [0, len(data_pos_i)]), shape=(1, self.num_feat)
                    )
                )
            data_file.close()
        
        return sparse.vstack(data).toarray(), sparse.vstack(label).toarray()

    def train_pi_b(
        self,
        n_rounds: int = 2000,
    ) -> None:
        idx = self.random_.choice(self.num_test_data, size=n_rounds, replace=False)
        contexts = self.test_data[idx]
        contexts = self.sc.fit_transform(self.pca.fit_transform(contexts))
        
        eta = self.random_.uniform(
            0, 0.5, size=(n_rounds, self.n_comb_action)
        )
        
        q_x_a = self.test_label[idx]
        
        q_x_m = np.zeros(n_rounds*self.n_comb_action).reshape(n_rounds,self.n_comb_action)
        
        for i in range(q_x_a.shape[0]):
            x = q_x_a[i,:]*self.action_context
            q_x_m[i, :] = np.count_nonzero(x, axis=1)
        

        q_x_m_ = q_x_m
        q_x_m = q_x_m_*(1-eta)
        q_x_m += (1 - q_x_m_) * eta

        n_use_action = self.action_context.sum(axis=1)

        q_x_m[:,1:] = q_x_m[:,1:] / n_use_action[1:]
        q_x_m[:,0] = 0
        
        self.regressor = MultiOutputRegressor(Ridge(max_iter=500, random_state=12345))
        self.regressor.fit(contexts, q_x_m)

    def compute_pi_b(
        self,
        train_contexts: np.ndarray,
        #train_expected_rewards: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        r_hat = self.regressor.predict(train_contexts)
        pi_b = softmax(r_hat * beta)
        return pi_b

    def obtain_batch_bandit_feedback(
        self, n_rounds: Optional[int] = None, n_users: int = 300, beta: float = 3.0, reward_std: float = 3.0,
    ) -> dict:
        """Obtain batch logged bandit data."""
        
        idx = self.random_.choice(self.num_data, size=n_users, replace=False)
        fixed_user_contexts = self.data[idx]
        fixed_user_contexts = self.sc.fit_transform(
            self.pca.fit_transform(fixed_user_contexts)
        )
        
        user_idx = self.random_.choice(n_users, size=n_rounds)
        contexts = fixed_user_contexts[user_idx]
        
        
        #fixed_q_x_a (n_users,n_label)
        eta = self.random_.uniform(
            0, 0.5, size=(n_users, self.n_comb_action)
        )
        
        fixed_q_x_a = self.label[idx]
        
        
        fixed_q_x_m = np.zeros(n_users*self.n_comb_action).reshape(n_users,self.n_comb_action)

        for i in range(fixed_q_x_a.shape[0]):
            x = fixed_q_x_a[i,:]*self.action_context
            fixed_q_x_m[i, :] = np.count_nonzero(x, axis=1)

        fixed_q_x_m_ = fixed_q_x_m
        fixed_q_x_m = fixed_q_x_m_*(1-eta)
        fixed_q_x_m += (1 - fixed_q_x_m_) * eta

        n_use_action = self.action_context.sum(axis=1)

        fixed_q_x_m[:,1:] = fixed_q_x_m[:,1:] / n_use_action[1:]
        fixed_q_x_m[:,0] = 0
        
        q_x_m = fixed_q_x_m[user_idx]
        
        
        pi_b = self.compute_pi_b(contexts,beta=beta)
        
        
        #sample actions
        actions = np.zeros(n_rounds , dtype=int)
        
        for i in np.arange(n_rounds):    
            unique_action_set = np.arange(self.n_comb_action)
            score_ = pi_b[i]
            
            sampled_action = self.random_.choice(
                unique_action_set, p=score_, replace=False
            )
            actions[i] = sampled_action
        
        
        q_x_m_factual = q_x_m[np.arange(n_rounds), actions]
        #rewards = self.random_.binomial(n=1, p=q_x_m_factual)
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
            expected_reward_matrix=q_x_m,
            expected_reward_factual=q_x_m_factual,
            all_pscore=pi_b,
            pscore=pi_b[np.arange(n_rounds), actions],
        )
 

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