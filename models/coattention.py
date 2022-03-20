import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking
 
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, Q_lengths):

        C = torch.matmul(Q, torch.matmul(self.W_b, V))

        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))

        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
 

        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)

        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
 
        # (batch_size, hidden_dim)
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_dim)
        q = torch.squeeze(torch.matmul(a_q, Q.permute(0, 2, 1)))
 
        return  v, q
