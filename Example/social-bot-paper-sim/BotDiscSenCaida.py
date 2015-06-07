"""  Sensisity Analysis Botnet Discovery for Caida dataset

Dependency
---------------
loc6-20070501-2055-40.sigs
ddostrace.20070804_134936-10.sigs
det_caida_divs.pkz

"""
from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
from subprocess import check_call

bg_sigs, bg_nodes = parseToCoo('loc6-20070501-2055-40.sigs',
        undirected=True)
# bg_sigs = bg_sigs[0:300]
bg_sigs = bg_sigs[0:360]


def symmetrize(a):
        # return a + a.T - P.diag(a.diagonal())
        return a + a.T

def convert(sigs):
    return [symmetrize(g.tocsr()) for g in sigs]



botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
                                       undirected=True)
mix_sigs, mix_nodes = mix((bg_sigs, bg_nodes),
        (botnet_sigs, botnet_nodes), 200)

# div_data = zload('./det_caida_divs.pkz')
div_data = zload('./det_caida_divs-360.pkz')
divs = div_data['divs']
# P.plot(div_data['divs'])
# P.show()
# sys.exit(0)

THRE = 0.19

det_idx = [i for i, div in enumerate(divs) if div > THRE]

# def convert (g):
#     ms = g.todense()
#     return ms + ms.T
bot_adjs = [mix_sigs[idx] for idx in det_idx]


# convert bot_adjs to symmetric and to csr format
bot_adjs = convert(bot_adjs)

# def ident_pivot_nodes(bot_adjs, weights, thres):
# pivot_th = 0.2
pivot_th = 0.1
cor_th = 0.3
# w1 = 0
# w2 = 0
w1 = 2
# w2 = 0.013
w2 = 0.01
# w2 = 0.013
lamb = 10

# tr = dict()
# w1_set = range(10)
# tr['w1_set'] = w1_set
# tr['stat'] = []
# for w1 in range(10):
#     botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
#     det_ips = set([mix_nodes[i] for i in botnet])
#     trace = get_quantitative(botnet_nodes, det_ips, mix_nodes)
#     tr['stat'].append(trace)

# zdump(tr, 'w1_influe_res.pkz')

# w2_set = P.linspace(0, 0.1, 10)
# tr['w2_set'] = w2_set
# tr['stat'] = []
# for w2 in w2_set:
#     botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
#     det_ips = set([mix_nodes[i] for i in botnet])
#     trace = get_quantitative(botnet_nodes, det_ips, mix_nodes)
#     tr['stat'].append(trace)

# zdump(tr, 'w2_influe_res-2.pkz')
