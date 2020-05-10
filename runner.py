from base_graph import Graph
from model import Model

from tqdm import trange
import matplotlib.pyplot as plt

g = Graph(31, type='Superfan', flag=5)

sweep_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.4, 0.5, 1, 5, 10]
final_losses = []

g.train(schedule=[(10,31,5),(10,21,5),(100,13,4),(100,8,3),(500,8,7)])
g.vis_global_params(root='schedule')
g.vis_individual_params(root='schedule')
g.vis_all_single_net(id=0, root='schedule')

converged_loss = g.global_params['mean_lr'][-1]

for lr in sweep_vals:
    m = Model(id=0, lr=lr)
    for _ in trange(1000):
        x, y = next(iter(g.train_loader))
        m.step(x, y)
        x, y = next(iter(g.eval_loader))
    final_losses.append(m.eval(x, y))


plt.plot(sweep_vals, final_losses)
plt.vlines([converged_loss], 0, 2)
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Validation Loss After 500 Iterations')
plt.savefig('lr_sweep')
plt.show()
