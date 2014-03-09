#!/usr/bin/env python
############
from __future__ import print_function, division
from SBDet import *
import pylab as P
import networkx as nx
from subprocess import check_call
from Util import load, zload

def get_ips(data, format_=None):
    nnx = NetworkXGraph(data=data)
    if format_ is not None:
        return [format_(ip) for ip in nnx.get_vertices()]
    return nnx.get_vertices()

def ab_traf_bot_det(adj_mats, w1, w2, lamb):
#### Identify the Pivot Nodes ######
    # tr = load('./Result/GCM_tr.pk')
    # weights = tr['solution']
    node_num = adj_mats[0].shape[0]
    weights = np.ones((node_num, )) / node_num
    p_nodes = ident_pivot_nodes(adj_mats, weights, 0.8)

#### Calculate interactions of nodes with pivot nodes ####
    inta = cal_inta_pnodes(adj_mats, weights, p_nodes)

#### Calculate the correlation graph ####
    A, npcor = cal_cor_graph(adj_mats, p_nodes, 0.2)
    # import ipdb;ipdb.set_trace()

    P0, q0, W = com_det_reg(A, inta, w1=w1, w2=w2, lamb=0, out='./prob.sdpb')
    # W = com_det_reg2(A, inta, w1=w1, w2=w2, lamb=0, out='./prob.sdpb')
    check_call('./csdp6.1.0linuxp4/bin/csdp ./prob.sdpb ./botnet.sol',
               shell=True)

    node_num = len(inta)
    Z, X = parse_CSDP_sol('./botnet.sol', node_num+1)
    solution = randomization(X, P0, q0)
    inta_diff = np.dot(inta, solution)
    print('inta_diff', inta_diff)

    botnet, = np.nonzero(solution > 0)
    print('[%i] ips out of [%i] ips are detected as bots' %
          (len(botnet), node_num))
    return botnet


def ana1(w1, w2, lamb):
    data = load('./Result/small_prob.pk')
    # bot_ids = ab_traf_bot_det(data['adj_mats'], w1=10, w2=0.001, lamb=0)
    bot_ids = ab_traf_bot_det(data['adj_mats'], w1=w1, w2=w2, lamb=lamb)
    bot_ips = [data['ips'][i] for i in bot_ids]
# print('bot_ips', bot_ips)

    ddos_ips = data['ddos_ips']
    stat = get_quantitative(ddos_ips, bot_ips,
                            data['ips'], show=True)
    return stat, bot_ips
    # return stat


##########################
# stat_vec = []
# w1_set = P.linspace(0, 2, 10)
# for w1 in w1_set:
#     stat_vec.append(ana1(w1, w2=0.001, lamb=0))

# tr = dict(w1=w1_set, stat_vec=stat_vec)
# zdump(tr, './Result/w1_stat_vec.pkz')

# fpr, tpr = roc(zip(*stat_vec))
# P.plot(fpr, tpr, '+-')
##########################
# P.plot(w1_set, fpr, '+-')
# P.plot(w1_set, tpr, 'x--')
# P.xlabel('w1')
# P.legend(['fpr', 'tpr'])
# P.title('Influence of w1 on fpr and tpr')
# P.savefig('./w1_influ.pdf')
# P.show()
###########################

# stat_vec = []
# w2_set = P.linspace(0, 1, 10)
# for w2 in w2_set:
#     stat_vec.append(ana1(w1=4, w2=w2, lamb=0))

# tr = dict(w2=w2_set, stat_vec=stat_vec)
# zdump(tr, './Result/w1_stat_vec.pkz')

# fpr, tpr = roc(zip(*stat_vec))
# P.plot(fpr, tpr, '+-')

###########################
# P.subplot(211)
# P.plot(w2_set, fpr, '+-')
# P.plot(w2_set, tpr, 'x--')
# P.legend(['fpr', 'tpr'])
# P.title('Influence of w2 on fpr and tpr')
# P.subplot(212)
# P.plot(w2_set, P.array(tpr)-P.array(fpr), '*-.')
# P.legend(['tpr-fpr'])
# P.xlabel('w2')
# P.savefig('./w2_influ.pdf')
# P.show()
###########################


# stat_vec = []
# lamb_set = P.linspace(0, 30, 10)
# for lamb in lamb_set:
#     stat_vec.append(ana1(w1=4, w2=0.1, lamb=lamb))

# fpr, tpr = roc(zip(*stat_vec))
# P.plot(fpr, tpr, '+-')
##########################
# P.subplot(211)
# P.plot(lamb_set, fpr, '+-')
# P.plot(lamb_set, tpr, 'x--')
# P.legend(['fpr', 'tpr'])
# P.title('Influence of lambda on fpr and tpr')
# P.subplot(212)
# P.plot(lamb_set, P.array(tpr)-P.array(fpr), '*-.')
# P.legend(['tpr-fpr'])
# P.xlabel('lambda')
# P.savefig('./lamb_influ.pdf')
# P.show()

###########################

stat_vec = []
w2_set = P.linspace(0, 1, 10)
dv = []
for w2 in w2_set:
    stat, detected_ips = ana1(w1=4, w2=w2, lamb=0)
    dv.append(len(detected_ips))

###########
P.plot(w2_set, dv, '+-')
P.xlabel('w2')
P.ylabel('no. of reported bots')
P.title('Influence of regularization weight')
P.savefig('./w2_reported_bots.pdf')
P.show()
##########

