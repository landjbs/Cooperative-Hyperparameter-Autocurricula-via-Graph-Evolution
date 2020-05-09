from base_graph import Graph


x = Graph(156, type='funel', flag=5)
x.train(1000)
x.vis_global_params()
