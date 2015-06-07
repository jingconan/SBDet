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
# import ipdb;ipdb.set_trace()
# For Test of Dropbox
# bg_sigs, bg_nodes = parseToCoo('ddostrace.20070804_141436-10.sigs',
# bg_sigs, bg_nodes = parseToCoo('ddostrace.20070804_141936-10.sigs',
# import ipdb;ipdb.set_trace()

# bg_sigs = bg_sigs[0:200]

    # mix_sigs, mix_nodes = mix((normal_sigs, normal_nodes),
                              # (botnet_sigs, botnet_nodes), 20)

deg_samples = mg_sample(len(bg_nodes), bg_sigs, 100, 50)

# model, para, debug_ret = select_model(deg_samples, ['ER', 'PA'], True)

iso_n = np.sum(deg_samples.ravel() == 0)
PA_para, PA_lk = mle(deg_samples, 'PA')
print('PA_para', PA_para)
print('PA_lk', PA_lk - 2.439260 * iso_n)
ER_para, ER_lk = mle(deg_samples, 'ER')
print('ER_lk', ER_lk)
print('ER_para', ER_para)

# bg_divs = monitor_deg_dis(bg_sigs, 'ER', (ER_para, 1e-10), minlength=None)

# THRE = 0.05
# THRE = 1
THRE = 0.18
# normal_sigs = [bg_sigs[i] for i, div in enumerate(bg_divs) if div < THRE]
normal_sigs = bg_sigs


# botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_141936-10.sigs',
botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
                                       undirected=True)


# mix_sigs, mix_nodes = mix((normal_sigs, bg_nodes),
# normal_sigs = [sig.tolil() for sig in normal_sigs]
# botnet_sigs = [sig.tolil() for sig in botnet_sigs]
mix_sigs, mix_nodes = mix_append((normal_sigs, bg_nodes),
        (botnet_sigs, botnet_nodes), 200)
# import ipdb;ipdb.set_trace()
# mix_sigs, mix_nodes = mix_append((normal_sigs, bg_nodes),

divs = monitor_deg_dis(mix_sigs, 'ER', (ER_para, 1e-10), minlength=None)
# zdump(dict(divs=divs, ab_rg=[200, 230]), './det_caida_divs-360.pkz')


det_idx = [i for i, div in enumerate(divs) if div > THRE]

P.plot(divs)
P.show()
import ipdb;ipdb.set_trace()

# def convert (g):
#     ms = g.todense()
#     return ms + ms.T
bot_adjs = [mix_sigs[idx] for idx in det_idx]

# def ident_pivot_nodes(bot_adjs, weights, thres):
# pivot_th = 0.2
pivot_th = 0.15
cor_th = 0.2
w1 = 1
w2 = 1
lamb = 1

# botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
node_num = bot_adjs[0].shape[0]
weights = np.ones((node_num, )) / node_num  # equal weights
sigs = bot_adjs
p_nodes, total_inta_mat = ident_pivot_nodes(bot_adjs, weights, pivot_th)
print('p_nodes', p_nodes)

inta = cal_inta_pnodes(sigs, weights, p_nodes)
print('inta', inta)
P.plot(inta)
P.show()
import ipdb;ipdb.set_trace()
