import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from model import Model


def build_data_loader(is_train, batch_size):
    return DataLoader(MNIST('/files/', train=is_train,
        transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)


class Graph(object):
  def __init__(self, n, train_batch_size, eval_batch_size):
    self.n = n
    self.models = [Model() for model in range(n)]
    # data loading
    self.train_loader = build_data_loader(True, train_batch_size)
    self.eval_loader = build_data_loader(True, eval_batch_size)
    # global param tracking
    self.global_params = {'mean_lr': []}

  def get_normed_fitness(self, x_batch, y_batch):
    fitnesses = np.array([model.eval(x_batch, y_batch) for model in self.models])
    return (fitnesses / self.n)

  def train_models(self, x_batch, y_batch):
    for model in self.models:
      model.step(x_batch, y_batch)

  def update_models(self, x_eval_batch, y_eval_batch):
    fitnesses = self.get_normed_fitness(x_eval_batch, y_eval_batch)
    # TODO: some selection stuff for updating models prob uses an adj mat
    for model in self.models:
      # select nums
      # model.update_hyperparams() ...
      model.log_hyperparams()

  def log_global_params(self):
    lr_buffer = []
    for model in self.models:
      lr_buffer.append(model.param_logs['lr'])
    self.global_params['mean_lr'].append(np.mean(lr_buffer))

  def train(steps):
    for step in trange(steps, desc='Training'):
      x_train_batch, y_train_batch = next(iter(self.train_loader))
      self.train_models(x_train_batch, y_train_batch)
      x_eval_batch, y_eval_batch = next(iter(self.eval_loader))
      self.update_models(x_eval_batch, y_eval_batch)
      self.log_global_params()
