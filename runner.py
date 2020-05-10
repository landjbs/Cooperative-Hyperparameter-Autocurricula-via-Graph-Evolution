from base_graph import Graph
from model import Model

from tqdm import trange
import matplotlib.pyplot as plt

g = Graph(16, type='Funnel', flag=5)

sweep_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.4, 0.5, 1, 5, 10]
final_losses = []

g.train(1000)
g.vis_global_params(root='shorter')
g.vis_individual_params(root='shorter')
g.vis_all_single_net(id=0, root='shorter')

converged_loss = g.global_params['mean_lr'][-1]

for lr in sweep_vals:
    m = Model(id=0, lr=lr)
    for _ in trange(500):
        x, y = next(iter(g.train_loader))
        m.step(x, y)
        x, y = next(iter(g.eval_loader))
    final_losses.append(m.eval(x, y))


plt.plot(sweep_vals, final_losses)
plt.vlines([0.4], 0, 2)
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Validation Loss After 500 Iterations')
plt.savefig('lr_sweep')
plt.show()
