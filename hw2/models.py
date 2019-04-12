'''
model_archive.py

A file that contains neural network models.
You can also implement your own model here.
'''
import torch.nn as nn

class Baseline(nn.Module):
	def __init__(self, hparams):
		super(Baseline, self).__init__()

		self.conv0 = nn.Sequential(
			nn.Conv1d(hparams.num_mels, 32, kernel_size=8, stride=1, padding=0),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(8, stride=8)
		)

		self.conv1 = nn.Sequential(
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(8, stride=8)
		)

		self.conv2 = nn.Sequential(
			nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.MaxPool1d(4, stride=4)
		)

		self.linear = nn.Linear(192, len(hparams.genres))

	def forward(self, x):
		x = x.transpose(1, 2)
		x = self.conv0(x)
		x = self.conv1(x)
		x = self.conv2(x)

		x = x.view(x.size(0), x.size(1)*x.size(2))
		x = self.linear(x)

		return x
