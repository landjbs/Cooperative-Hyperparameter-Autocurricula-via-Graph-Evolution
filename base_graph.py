import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


from model import Model
from evolution import generate_graph

def build_data_loader(is_train, batch_size):
    download = not os.path.exists('data/MNIST')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                  ])
    dataset = MNIST('data', train=is_train,
                    transform=transform, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Graph(object):
  def __init__(self, n, type, flag, train_batch_size=64, eval_batch_size=64):
    # ad
    self.n = n
    self.models = [Model(id) for id in range(n)]
    # graph
    (self.adjMat,
     self.childrenList) = generate_graph(n, type=type, flag=flag)
    # data loading
    self.train_loader = build_data_loader(True, train_batch_size)
    self.eval_loader = build_data_loader(True, eval_batch_size)
    # global param tracking
    self.global_params = {'fitness': [], 'mean_lr': []}

  def get_normed_fitness(self, x_batch, y_batch, track=False):
    fitnesses = np.array([1.0 / model.eval(x_batch, y_batch)
                          for model in self.models])
    if track:
        self.global_params['fitness'].append(np.mean(fitnesses))
    return (fitnesses / self.n)

  def step_models(self, x_batch, y_batch):
    for model in self.models:
      model.step(x_batch, y_batch)
    return True

  def select_parents(self, fitnesses, n_parents):
      inv_fitnesses = [fit for fit in fitnesses]
      fitSum = sum(inv_fitnesses)
      p = [fit/fitSum for fit in inv_fitnesses]
      return np.random.choice(range(self.n),size = n_parents, p=p)

  def select_child(self,parent):
    p = self.adjMat[parent]
    if sum(p) > 0:
      child = np.random.choice(range(self.n), p=p)
    else:
      child = None
    return child

  def update_models(self, x_eval_batch, y_eval_batch):
    fitnesses = self.get_normed_fitness(x_eval_batch, y_eval_batch, track=True)
        self.adjMat, self.childrenList =
    # TODO: some selection stuff for updating models prob uses an adj mat
    for model in self.models:
      # select nums
      # model.update_hyperparams(delta_lr=?) ...
      model.log_hyperparams()
    return True

  def log_global_params(self):
    lr_buffer = []
    for model in self.models:
      lr_buffer.append(model.param_logs['lr'])
    self.global_params['mean_lr'].append(np.mean(lr_buffer))
    return True

  def train(self, steps):
    for step in trange(steps, desc='Training'):
      x_train_batch, y_train_batch = next(iter(self.train_loader))
      self.step_models(x_train_batch, y_train_batch)
      x_eval_batch, y_eval_batch = next(iter(self.eval_loader))
      self.update_models(x_eval_batch, y_eval_batch)
      self.log_global_params()
    return True

  def vis_global_params(self, exclude=[]):
      ''' Plots all globally-tracked params '''
      for name, buffer in self.global_params.items():
          if name in exclude:
              continue
          plt.plot(buffer)
          plt.title(f'Global - {name}')
          plt.xlabel('Iteration')
          plt.ylabel(name)
          plt.show()

  def vis_individual_params(self):
      ''' Plots graphs of each param across nets '''
      for name, buffer in self.models[0].param_logs.items():
          for model in self.models:
              plt.plot(model.param_logs[name], label=str(model))
          plt.xlabel('Iteration')
          plt.ylabel(name)
          plt.title(f'{name} Across Networks')
          plt.legend()
          plt.show()

  def vis_all_single_net(self, id, exclude=[]):
      ''' Plots all params of net with id=id on single graph '''
      model = self.models[id]
      for name, buffer in model.param_logs.items():
          if name in exclude:
              continue
          plt.plot(buffer, label=name)
      plt.title(f'Params for {model}')
      plt.xlabel('Iteration')
      plt.show()
