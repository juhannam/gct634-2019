import torch
import torch.nn as nn
from constants import *


class Baseline(nn.Module):
  def __init__(self, hidden_size, rnn_cell='lstm', n_layers=1):
    super(Baseline, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.encoder = nn.Embedding(N_DICT+1, N_DICT+1)  # because of start token, we add +1 on N_DICT

    # TODO: Fill in below
    # Hint: nn.LSTM / nn.GRU.
    if rnn_cell == 'lstm':
      self.rnn = None
    elif rnn_cell == 'gru':
      self.rnn = None

    # TODO: Fill in below
    # output of decoder should be number of classes
    self.decoder = nn.Linear(in_features=None, out_features=None)
    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, x, hidden, temperature=1.0):
    encoded = self.encoder(x)  # shape of (Batch, N_DICT)
    # To match the RNN input form(step, Batch, Feature), add new axis on first dimension
    encoded = encoded.unsqueeze(0)

    # TODO: Fill in below
    # hint: use self.rnn you made
    output, hidden = None
    output = output.squeeze(0)

    # TODO: Fill in below
    # hint: use self.decoder
    output = None

    # Optional: apply temperature
    pred = self.log_softmax(output)

    return pred, hidden

  def init_hidden(self, batch_size, random_init=False):
    if random_init:
      return torch.randn(self.n_layers, batch_size, self.hidden_size), \
             torch.randn(self.n_layers, batch_size, self.hidden_size)
    else:
      return torch.zeros(self.n_layers, batch_size, self.hidden_size),\
             torch.zeros(self.n_layers, batch_size, self.hidden_size)
