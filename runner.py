from base_graph import Graph


x = Graph(156, type='Funnel', flag=5)
x.train(3)
x.vis_global_params(root='test')
x.vis_individual_params(root='test')
x.vis_all_single_net(id=0, root='test')
