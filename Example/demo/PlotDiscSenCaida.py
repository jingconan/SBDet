from SBDet import *
import pylab as P
import sys
# tr = zload('./w1_influe_res.pkz')

# w1_set = tr['w1_set']
# stat = tr['stat']
# tp, fn, tn, fp, fpr, tpr = zip(*stat)
# P.plot(w1_set, fp)
# P.plot(w1_set, tp)
# P.plot(fp, tp)
# for f, t, l in zip(fp, tp, w1_set):
#     P.text(f, t, str(l))
# P.show()

def w2():
    tr = zload('./w2_influe_res-2.pkz')

    w2_set = tr['w2_set']
    stat = tr['stat']
    tp, fn, tn, fp, fpr, tpr = zip(*stat)
    P.plot(w2_set, fp)
    P.plot(w2_set, tp)
    P.plot(w2_set, np.array(tp) - np.array(fp))
    # P.plot(fp, tp)
    # for f, t, l in zip(fp, tp, w1_set):
        # P.text(f, t, str(l))
    P.show()

# w2()

tr = zload('./viz-com-test.pkz')
A = tr['A']
solution = tr['solution']
botnet_nodes = tr['botnet_nodes']
mix_nodes = tr['mix_nodes']
nc = ['blue', 'red', 'green', 'black', 'm']
ns = 'os><x'
pos = zload('graph_pos.pkz')
# import ipdb;ipdb.set_trace()
shrink_map = tr['shrink_map']
# pos = [pos[v] for k,v in shrink_map.iteritems()]
# P.plot(tr['inta'])
# P.show()
# import ipdb;ipdb.set_trace()


import igraph
ig = igraph.Graph(directed=False)
ig.add_vertices(range(len(shrink_map)))
I, J = A.nonzero()
edges = zip(I, J)
ig.add_edges(edges)
# res = ig.community_leading_eigenvector(clusters=2)
# res = ig.community_leading_eigenvector(clusters=-1)
res = ig.community_leading_eigenvector(clusters=-1)
solution_leading_eigen = res.membership
# import ipdb;ipdb.set_trace()

# ic = ig.components()
# icm = ic.membership
# comp1 = (np.array(icm) == 0).nonzero()[0]
# sg = ig.subgraph(comp1)
# res = sg.community_leading_eigenvector(clusters=2)
# sol_1c = [3] * len(shrink_map)
# for a, b in zip(comp1, res.membership):
#     sol_1c[a] = b
# solution_leading_eigen_one_component = sol_1c
# tr['graph_components'] = icm
# tr['solution_leading_eigen_one_component'] = solution_leading_eigen_one_component



# import ipdb;ipdb.set_trace()
# res = ig.community_infomap(trials=1)
res = ig.community_infomap(trials=50)
solution_info_map = res.membership
# import ipdb;ipdb.set_trace()

solution_walk_trap = ig.community_walktrap().as_clustering(3).membership

tr['solution_walk_trap'] = solution_walk_trap
tr['solution_leading_eigen'] = solution_leading_eigen
# tr['solution_info_map'] = solution_info_map
# zdump(tr, 'com-det-compare-res.pkz')


# P.figure((800,600))
P.figure()
# P.show()
# import sys; sys.exit(0)
P.subplot(221)
        # pos='graphviz',
        # pos='spring',
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in solution],
        node_shape=[ns[int(i > 0)] for i in solution],
        edge_color='grey',
        # pic_name='./det_res_sdp.pdf'
        )
P.title('SBDet')
        # pos=pos,


P.subplot(222)
# P.figure()
pos = draw_graph(A,
        None,
        pic_show=False,
        # pos=pos,
        pos='graphviz',
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i)] for i in solution_leading_eigen],
        node_shape=[ns[int(i)] for i in solution_leading_eigen],
        edge_color='grey',
        )
P.title('LeadingEigen')

# pos = draw_graph(A,
#         None,
#         pic_show=False,
#         pos=pos,
#         with_labels=False,
#         node_size=100,
#         node_color=[nc[int(i)] for i in solution_walk_trap],
#         node_shape=[ns[int(i)] for i in solution_walk_trap],
#         edge_color='grey',
        # )



P.subplot(223)
# P.figure()
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in tr['ref_sol']],
        node_shape=[ns[int(i > 0)] for i in tr['ref_sol']],
        # edge_color='grey',
        # pic_name='./com_ground_truth.pdf'
        )
P.title('GroundTruth')
# zdump(pos, 'graph_pos.pkz')
# P.figure()
# P.figure()

P.subplot(224)
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in solution_info_map],
        node_shape=[ns[int(i > 0)] for i in solution_info_map],
        edge_color='grey',
        # pic_name='./det_res_info_map.pdf'
        )
P.title('InfoMap')

# P.figure()

# P.subplot(224)
# ignore_nodes = (np.array(solution_leading_eigen_one_component) == 3).nonzero()[0]
# ignore_nodes = ignore_nodes.tolist()
# pos = draw_graph(A,
#         ignore_nodes=ignore_nodes,
#         pic_show=False,
#         pos=pos,
#         with_labels=False,
#         node_size=100,
#         node_color=[nc[int(i)] for i in solution_leading_eigen_one_component],
#         node_shape=[ns[int(i)] for i in solution_leading_eigen_one_component],
#         edge_color='grey',
        # pic_name='./det_res_info_map.pdf'
        # )



# P.figure()
# sol_inta = tr['inta'] > 0.001
# pos = draw_graph(A,
#         None,
#         pic_show=False,
#         pos=pos,
#         with_labels=False,
#         node_size=100,
#         node_color=[nc[int(i > 0)] for i in sol_inta],
#         node_shape=[ns[int(i > 0)] for i in sol_inta],
#         edge_color='grey',
        # pic_name='./det_res_info_map.pdf'
        # )



P.show()
        # pic_name='det.pdf')
        # pos='graphviz',

# P.figure()

# N = len(mix_nodes)
# nm = dict(zip(mix_nodes, range(N)))
# botnet_idx = [nm[ip] for ip in botnet_nodes]
# ref_sol = np.zeros((N,))
# ref_sol[botnet_idx] = 1

# draw_graph(A,
#         None,
#         pic_show=True,
#         pos=pos,
#         with_labels=False,
#         node_size=100,
#         node_color=[nc[int(i)] for i in ref_sol],
#         edge_color='grey',
#         pic_name='det.pdf')



