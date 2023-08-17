import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class Linear_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super().__init__()
        self.linear1 = nn.Linear(*input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        actions = self.linear2(x)
        return actions
    
    def save(self, file_name='model_presentation.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    

class Agent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, output_size,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(output_size)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        file_name_model = "./model/cart_tutorial_score.pth"

        
        self.Q_eval = Linear_DQN(input_size=input_size, output_size = output_size, lr = self.lr, hidden_size=256)
        self.Q_eval.load_state_dict(torch.load(file_name_model))

        

    def choose_action(self, observation):
        state = torch.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = torch.argmax(actions).item()
    
        return action
    
   