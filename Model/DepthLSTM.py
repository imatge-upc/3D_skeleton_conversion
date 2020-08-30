# Here we define an LSTM that regresses the depth
import torch
import torch.nn as nn


class DepthLSTM(nn.Module):

  def __init__(self, hidden_size, num_layers, seq_len, batch_size, num_joints):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_joints = num_joints

    # Define the LSTM layer
    self.lstm = nn.LSTM(2*num_joints, hidden_size, num_layers, bias=True, batch_first = True)

    # Define a Linear Layer to obtain the depth coordinate
    self.Linear = nn.Linear(hidden_size, num_joints, bias=True)

  def forward(self, seq, state=None):
    h, state = self.lstm(seq, state) # h.shape = [batch_size, seq_len, hidden_size]
    h = h.contiguous().view(-1, self.hidden_size) # h.shape = [batch_size*seq_len, hidden_size]
    y = self.Linear(h) # y.shape = [batch_size*seq_len, num_joints]
    y = y.contiguous().view(self.batch_size, self.seq_len, self.num_joints) #y.shape = [batch_size, seq_len, num_joints]
    return y, state