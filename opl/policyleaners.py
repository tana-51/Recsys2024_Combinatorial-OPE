from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataclasses import dataclass
from utils import (
    GradientBasedPolicyDataset,
    RegBasedPolicyDataset,
)
from typing import List
from obp.utils import softmax

seed = 12345
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

@dataclass
class RegBasedPolicyLearner:
    dim_context : int
    n_action : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    
    def __post_init__(self, ):
        seed = 12345
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, self.n_action)
        )
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test):
        
            
        mydataset = RegBasedPolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["expected_reward_matrix"], dataset_test["expected_reward_matrix"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            
            for x_,a_,r_ in train_dataloader:
                optimizer.zero_grad()
                q_hat = self.nn_model(x_)
                idx = torch.arange(a_.shape[0], dtype=torch.long)
                loss = ((r_ - q_hat[idx, a_]) ** 2).mean()
                loss.backward()
                optimizer.step()
            
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            #self.train_loss.append(loss_epoch)

    def predict(self, dataset_test: np.ndarray, beta: float = 10):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        q_hat = self.nn_model(x).detach().numpy()

        return softmax(beta * q_hat)
    
    def predict_q_hat(self, dataset_test):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()

        return self.nn_model(x).detach().numpy()
    
    
    
@dataclass
class GradientBasedPolicyLearner:
    dim_context : int
    n_action : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    
    def __post_init__(self, ):
        seed = 12345
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, self.n_action),
            nn.Softmax(dim=1)
        )
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat: np.ndarray = None):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_action))
            
        mydataset = GradientBasedPolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pscore"], 
            q_hat = q_hat, 
            pi_0 = dataset["all_pscore"], 
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["expected_reward_matrix"], dataset_test["expected_reward_matrix"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            
            for x_,a_,r_,p,q_hat_,pi_0_ in train_dataloader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            #self.train_loss.append(loss_epoch)
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, a] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        return self.nn_model(x).detach().numpy()
    

@dataclass
class OPCB:
    dim_context : int
    n_action : int
    kth_element : List[List[int]]
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    
    def __post_init__(self, ):
        seed = 12345
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, self.n_action),
            nn.Softmax(dim=1)
        )
        

        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat: np.ndarray = None):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_action))
            
        
        self.action_context = dataset["action_context"]
            
        mydataset = GradientBasedPolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pscore"], 
            q_hat = q_hat, 
            pi_0 = dataset["all_pscore"], 
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["expected_reward_matrix"], dataset_test["expected_reward_matrix"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            
            for x_,a_,r_,p,q_hat_,pi_0_ in train_dataloader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            #self.train_loss.append(loss_epoch)
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        


        #iw for OPCB
        n = a.shape[0]
        
        kth_action_context = self.action_context[:,self.kth_element]
        
        id_list = []
        for i in a:
            id_ = np.where(np.all(kth_action_context==kth_action_context[i,:],axis=1))
            id_list.append(id_[0].tolist())
        

        pscore_OPCB = torch.zeros(n)
        action_dist_OPCB = torch.zeros(n)
        pi_OPCB = torch.zeros(n)

        pos_ = 0
        for i in id_list:
            pscore_OPCB[pos_]=torch.sum(pi_0[pos_,i])
            action_dist_OPCB[pos_] = torch.sum(current_pi[pos_,i])
            pi_OPCB[pos_] = torch.sum(pi[pos_,i])
            pos_ += 1
            
        
        log_prob_OPCB = torch.log(pi_OPCB + self.log_eps)
        iw_opcb = action_dist_OPCB / pscore_OPCB
        #estimated_policy_grad_arr = iw_opcb * (r - q_hat_factual) * log_prob[idx, a]
        estimated_policy_grad_arr = iw_opcb * (r - q_hat_factual) * log_prob_OPCB
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        return self.nn_model(x).detach().numpy()