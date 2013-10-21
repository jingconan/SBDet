"""  generate topology

"""
import numpy as np
import networkx as nx
from Util import save_graph
from Util import zdump


#Assuming that the graph g has nodes and edges entered
# g = nx.random_powerlaw_tree(20)
# g = nx.fast_gnp_random_graph(200, 0.01)
g = nx.scale_free_graph(200)
g = g.to_undirected()
cg_list = nx.connected_component_subgraphs(g)
save_graph(cg_list[0],"my_graph.pdf")
zdump(np.asarray(nx.to_numpy_matrix(cg_list[0])), 'overlay_adj.pkz')
