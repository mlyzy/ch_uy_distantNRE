import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class Gateunit(nn.Module):


    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输入门i_t
        self.W_i = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # 遗忘门f_t
        self.W_f = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))

        self.W_g = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))

    def forward(self, x, init_states=None):


    
        i_t = torch.sigmoid(x @ self.W_i + self.b_i)
        f_t = torch.sigmoid(x @ self.W_f + self.b_f)
        g_t = torch.tanh(x @ self.W_g  + self.b_g)
        h_t = f_t * x + i_t * g_t
        
        return h_t

