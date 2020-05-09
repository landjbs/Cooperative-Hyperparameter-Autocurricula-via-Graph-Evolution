import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class Model(nn.Module):
  def __init__(self, id):
    super(Model, self).__init__()
    self.id = id
    # initialize hyper params
    self.init_lr = np.random.uniform(0, 1)
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
    self.optim = optim.SGD(self.parameters(), lr=self.init_lr)
    self.param_logs = {'loss': [], 'lr': [self.init_lr]}

  def __repr__(self):
      return f'Model {self.id}'

  def fetch_lr(self):
      return self.optim.param_groups[0]['lr']

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)

  def step(self, batch_x, batch_y):
    self.train()
    self.optim.zero_grad()
    y_hat = self(batch_x)
    loss = F.nll_loss(y_hat, batch_y)
    loss.backward()
    self.optim.step()
    self.param_logs['loss'].append(loss.item())

  def update_hyperparams(self, new_lr):
    for g in self.optim.param_groups:
      g['lr'] = max(0, new_lr)

  def log_hyperparams(self):
    self.param_logs['lr'].append(self.fetch_lr())

  def eval(self, batch_x, batch_y):
    with torch.no_grad():
      y_hat = self(batch_x)
      loss = F.nll_loss(y_hat, batch_y)
    return loss.item()
