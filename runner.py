from base_graph import Graph
from model import Model

from tqdm import trange
import matplotlib.pyplot as plt

g = Graph(36, type='Funnel', flag=5)
m = Model(id=0, lr=0.01)

sweep_vals = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
final_losses = []

for lr in sweep_vals:
    for _ in trange(100):
        x, y = next(iter(g.train_loader))
        m.step(x, y)
        x, y = next(iter(g.eval_loader))
        final_losses.append(m.eval(x, y))


plt.plot(sweep_vals, final_losses)
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Validation Loss After 1000 Iterations')
plt.savefig('lr_sweep')
plt.show()

# g.train(1000)
# g.vis_global_params(root='test')
# g.vis_individual_params(root='test')
# g.vis_all_single_net(id=0, root='test')
