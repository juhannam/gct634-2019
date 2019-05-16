import numpy as np
from constants import *
from improvise_rnn import *
import torch
from torch.utils.data import Dataset

class JazzDataset(Dataset):
  def __init__(self):
    X, _, n_values, indices_values = load_music_utils()
    self.data = X

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    data = self.data[index, :, :]
    features = []
    for n in range(data.shape[0]):
      try:
        feature_index = np.where(data[n, :])[0][0]
      except:
        # first step is filled with only False,
        # thus 'np.where(data[n, :])[0][0]' will return None
        feature_index = N_DICT
      features.append(feature_index)
    return torch.LongTensor(features)


if __name__ == '__main__':
  train_dataset = JazzDataset()
  data_generator = iter(train_dataset)

  data_example = next(data_generator)
  print(data_example)

  # If you want to see the dictionaries, uncomment below

  X, _, n_values, indices_values = load_music_utils()
  for key in indices_values:
    print('{}: {}'.format(key, indices_values[key]))


