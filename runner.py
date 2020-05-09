from base_graph import Graph
from model import Model


g = Graph(36, type='Funnel', flag=5)
m = Model(id=0, lr=0.01)

for _ in range(1000):
    x, y = next(iter(g.train_loader))
    m.step(x, y)
    x, y = next(iter(g.eval_loader))
    m.eval(x, y)

plt.plot(m.param_logs['loss'])
plt.show()

# g.train(1000)
# g.vis_global_params(root='test')
# g.vis_individual_params(root='test')
# g.vis_all_single_net(id=0, root='test')
