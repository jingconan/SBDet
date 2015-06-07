"""  Estimate the Biase parameter of PA model
"""
from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
import numpy as np


N = 300
ba_sigs = gen_sigs('BA', 100, N, 1)
ba_nodes = ['pl-%i' %(i) for i in xrange(N)]
er_sigs = gen_sigs('ER', 100, N, 0.001)
er_nodes = ['er-%i' %(i) for i in xrange(N)]
# bg_nodes = botnet_nodes + range(extra_n)

cb_sigs, cb_nodes = mix_append((ba_sigs, ba_nodes),
                        (er_sigs, er_nodes), 0)

## Model Selection ##
tr = dict()
tr['n_num'] = []
n_num_set = range(10, 250, 10)
for n_num in n_num_set:
    deg_samples = mg_sample(2*N, cb_sigs, 80, n_num)
    iso_n = np.sum(deg_samples.ravel() == 0)

    PA_para, PA_lk = mle(deg_samples, 'PA')
# print('PA_para', PA_para)
# print('PA_lk', PA_lk)
    ER_para, ER_lk = mle(deg_samples, 'ER')
    tr['n_num'].append((n_num, iso_n, PA_para, PA_lk, ER_para, ER_lk))
    print(n_num, iso_n, PA_para, PA_lk, ER_para, ER_lk)

# zdump(tr, './EstBiase-n_num.pk')
zdump(tr, './EstBiase-n_num-revised.pk')
# print('ER_lk', ER_lk)
# print('ER_para', ER_para)

# md, para = select_model(deg_samples, ['PA','ER'])
# print('md', md)
# print('para', para)

# numel = lambda x: len(x.nonzero()[0])
# nv = [numel(sig) for sig in cb_sigs]
# P.plot(nv)
# P.show()

# import ipdb;ipdb.set_trace()
