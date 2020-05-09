from base_graph import Graph


x = Graph(36, type='Funnel', flag=5)
x.train(1000)
x.vis_global_params(root='test')
x.vis_individual_params(root='test')
x.vis_all_single_net(id=0, root='test')
