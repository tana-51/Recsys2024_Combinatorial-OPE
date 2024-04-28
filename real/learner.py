from dataclasses import dataclass
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from obp.ope import RegressionModel
from obp.utils import check_array
from obp.utils import check_ope_inputs
from scipy import stats
from scipy.stats import rankdata
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.utils import check_random_state
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


def generate_combinations(current_combination, n): 
    if len(current_combination) == n:
        return [current_combination]

    combinations = []
    combinations.extend(generate_combinations(current_combination + [0], n))
    combinations.extend(generate_combinations(current_combination + [1], n))

    return combinations

@dataclass
class PairWiseDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action1: np.ndarray
    action2: np.ndarray
    reward1: np.ndarray
    reward2: np.ndarray

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action1[index],
            self.action2[index],
            self.reward1[index],
            self.reward2[index],
        )

    def __len__(self):
        return self.context.shape[0]
    
    
    

class PairWiseRegression(nn.Module):
    def __init__(
        self,
        n_actions: int,
        x_dim: int,
        hidden_dim: int = 50,
    ):
        super(PairWiseRegression, self).__init__()
        # init
        self.n_actions = n_actions
        self.x_dim = x_dim

        # relative reward network
        self.fc1 = nn.Linear(self.x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.n_actions)

    def rel_reward_pred(self, x):
        h_hat = F.elu(self.fc1(x))
        h_hat = F.elu(self.fc2(h_hat))
        h_hat = F.elu(self.fc3(h_hat))
        h_hat = self.fc4(h_hat)
        return h_hat
    
   
    def forward(self, x, a1, a2, r1, r2):
        h_hat = self.rel_reward_pred(x)
        h_hat1, h_hat2 = h_hat[:, a1], h_hat[:, a2]
        loss = ((r1 - r2) - (h_hat1 - h_hat2)) ** 2

        return loss.mean()
 
    
    
    
    
    
def make_pairwise_data(bandit_data: dict,kth_element,cluster_contents_mat,obs_cluster_matrix):
    n_users = bandit_data["n_users"]
    fixed_user_contexts = bandit_data["fixed_user_contexts"]
    obs_action_matrix = bandit_data["obs_action_matrix"]
    reward_matrix = bandit_data["reward_matrix"]
    
  
    contexts_ = [fixed_user_contexts[0]]
    actions1_ = [0]
    actions2_ = [0]
    rewards1_ = [0]
    rewards2_ = [0]
    
    for u in np.arange(n_users):
        cluster = np.where(obs_cluster_matrix[u,:]==1)[0].tolist()
        for c in np.array(cluster):
            actions_in_c = np.where(cluster_contents_mat[:,c]==1)[0].tolist()
            for (a1, a2) in permutations(actions_in_c, 2):
                if obs_action_matrix[u,a1]==1 and obs_action_matrix[u,a2]==1:
                    r1, r2 = reward_matrix[u, (a1, a2)]
                    contexts_.append(fixed_user_contexts[u])
                    actions1_.append(a1), actions2_.append(a2)
                    rewards1_.append(r1), rewards2_.append(r2)    
                
    return PairWiseDataset(
        torch.from_numpy(np.array(contexts_)).float(),
        torch.from_numpy(np.array(actions1_)).long(),
        torch.from_numpy(np.array(actions2_)).long(),
        torch.from_numpy(np.array(rewards1_)).float(),
        torch.from_numpy(np.array(rewards2_)).float(),
    )


def train_pairwise_model(
    bandit_data: dict,
    kth_element,
    cluster_contents_mat,
    obs_cluster_matrix,
    lr: float = 1e-2,
    batch_size: int = 128,
    num_epochs: int = 50,
    gamma: float = 0.95,
    weight_decay: float = 1e-4,
    random_state: int = 12345,
    verbose: bool = False,
) -> None:
    pairwise_dataset = make_pairwise_data(bandit_data,kth_element,cluster_contents_mat,obs_cluster_matrix)
    data_loader = torch.utils.data.DataLoader(
        pairwise_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = PairWiseRegression(
        n_actions=bandit_data["n_comb_action"],
        x_dim=bandit_data["context"].shape[1],
    )
    
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    model.train()
    loss_list = []
    for _ in range(num_epochs):
        losses = []
        for x, a1, a2, r1, r2 in data_loader:
            loss = model(x, a1, a2, r1, r2)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        if verbose:
            print(_, np.sqrt(np.average(losses)))
        loss_list.append(np.average(losses))
        scheduler.step()
    x = torch.from_numpy(bandit_data["context"]).float()
    h_hat_mat = model.rel_reward_pred(x).detach().numpy()

    return h_hat_mat



def train_reward_model_via_two_stage(
    bandit_data: dict,
    kth_element,
    random_state: int = 12345,
) -> np.ndarray:
    
    n_users = bandit_data["n_users"]
    user_idx = bandit_data["user_idx"]
    n_comb_action = bandit_data["n_comb_action"]
    action = bandit_data["action"]
    action_context = bandit_data["action_context"]
    
    
    
    cluster_context = np.array(generate_combinations([], len(kth_element)))
    n_cluster = cluster_context.shape[0]
    
    #cluster_mat (n_comb_action, 2**len(kth_element) ) 
    cluster_contents_mat = np.zeros(n_comb_action*(2**len(kth_element))).reshape(n_comb_action,-1)

    for i in range(n_comb_action):
        for j in range(n_cluster):
            if np.all(action_context[i,kth_element] == cluster_context[j,:]):
                 cluster_contents_mat[i,j] = 1
    
    observed_cluster = []
    for a in action:
        observed_cluster.append(np.where(cluster_contents_mat[a,:]==1)[0][0])
    
    obs_cluster_matrix = np.zeros((n_users, n_cluster))
    for u, c in zip(user_idx,observed_cluster ):
        obs_cluster_matrix[u,c] = 1
        
        
        

    h_hat_mat = train_pairwise_model(bandit_data,kth_element,cluster_contents_mat,obs_cluster_matrix)
    reward_main = bandit_data["reward"].astype(float)
    reward_main -= h_hat_mat[
        np.arange(bandit_data["context"].shape[0]), bandit_data["action"]
    ]
    
    if n_cluster == 1:
        g_hat_mat = reward_main.reshape(-1,1)
    else:
        reg_model = RegressionModel(
            n_actions=n_cluster,
            action_context=np.eye(n_cluster),
            base_model=MLP(hidden_layer_sizes=(50, 50, 50), random_state=random_state, max_iter=3000,early_stopping=True),
        )

        g_hat_mat = reg_model.fit_predict(
            context=bandit_data["context"],
            action=np.array(observed_cluster),
            reward=reward_main,
            random_state=12345,
        )[:, :, 0]

    f_hat_mat = h_hat_mat #f_hat(x,m)
    
    x = np.where(cluster_contents_mat[np.arange(n_comb_action),:]==1)[1].tolist()
    
    for i in np.arange(h_hat_mat.shape[0]):
        f_hat_mat[i] += g_hat_mat[i][x]
    
    return f_hat_mat



    

