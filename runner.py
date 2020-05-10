from base_graph import Graph
from model import Model

from tqdm import trange
import matplotlib.pyplot as plt

g = Graph(31, type='Superfan', flag=5)

sweep_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2,
              0.5, 1, 5, 10]
final_losses = []

# g.train(schedule=[(200,31,5),(200,21,5),(200,13,4),(200,8,3),(200,8,7)])
g.train(1000)
winner = g.models[0].id
g.vis_global_params(root='schedule')
g.vis_individual_params(root='schedule')
g.vis_all_single_net(id=winner, root='schedule')

converged_loss = g.global_params['mean_lr'][-1]
print(f'converged at: {converged_loss}')
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
plt.ylabel('Validation Loss After 1000 Iterations')
plt.savefig('lr_sweep')
plt.show()
