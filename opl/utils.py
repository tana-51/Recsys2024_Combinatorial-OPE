from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataclasses import dataclass

seed = 12345
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

class GradientBasedPolicyDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()
        self.pscore = torch.from_numpy(pscore).float()
        self.q_hat = torch.from_numpy(q_hat).float()
        self.pi_0 = torch.from_numpy(pi_0).float()
    
    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )
    
    
class RegBasedPolicyDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()

    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )
        
    