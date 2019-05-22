import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import summary, cycle
import rnn, dataloader
import torch.nn as nn

config=dict(
  logdir = 'runs/improvise_RNN_' + datetime.now().strftime('%y%m%d-%H%M%S'),
  model_name = 'Baseline',

  iterations = 1024,
  checkpoint_interval = 32,

  batch_size = 60,
  temperature = 0.5,

  hidden_size = 100,
  n_layers = 2,
  rnn_cell = 'lstm',

  learning_rate = 1e-2,
  learning_rate_decay_steps = 500,
  learning_rate_decay_rate = 0.98
)


def train(logdir, model_name, iterations, checkpoint_interval, batch_size, temperature, hidden_size, n_layers, rnn_cell,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    os.makedirs(logdir, exist_ok=True)

    dataset = dataloader.JazzDataset()
    loader = DataLoader(dataset, batch_size, shuffle=True)

    model_class = getattr(rnn, model_name)
    model = model_class(hidden_size, rnn_cell, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    criterion = nn.NLLLoss()
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    model = model.to(device)
    summary(model)

    loop = tqdm(range(0, iterations + 1), desc='Training', unit='Steps')
    for i, batch in zip(loop, cycle(loader)):
        scheduler.step()
        optimizer.zero_grad()

        batch = batch.to(device)  # shape of (batch_size, sequence_length)

        c_0, h_0 = model.init_hidden(batch.shape[0])
        c_0 = c_0.to(device)
        h_0 = h_0.to(device)

        # TODO: Fill in below
        init_hidden = None

        hidden = init_hidden
        loss = 0.0

        for step in range(batch.shape[1] - 1):  # sequence_length - 1
          # TODO: Fill in below
          # run a step of training model.
          # x = semgent of batch, corresponds to current step. shape of (batch_size, 1)
          pred, hidden = model(x=None, hidden=hidden)

          # TODO: Fill in below
          # calcuate loss between prediction and the values of next step.
          # Hint: use criterion. See torch.nn.NLLLoss() function
          loss += None

        loss.backward()
        optimizer.step()

        # print loss
        loop.set_postfix_str("loss: {:.3f}".format(loss))

        # save model
        if i % checkpoint_interval == 0:
          torch.save({'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'model_name': model_name,
                      'hparams': dict(hidden_size=hidden_size,
                                      n_layers=n_layers,
                                      rnn_cell=rnn_cell)
                      },
                     os.path.join(logdir, 'model-{:d}.pt'.format(i)))

if __name__ == '__main__':
  train(**config)