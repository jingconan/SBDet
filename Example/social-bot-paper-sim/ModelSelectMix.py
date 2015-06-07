"""  Compare the Loglikelihood under different ratio of ER and PA models.
"""
from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
import numpy as np


N = 600

tr = dict()
tr['ratio'] = []
for ratio in P.linspace(0.01, 0.99, 10):
    # print('ratio', ratio)
    ba_sigs = gen_sigs('BA', 100, int(N * ratio), 1)
    ba_nodes = ['pl-%i' %(i) for i in xrange(N)]
    er_sigs = gen_sigs('ER', 100, int(N * (1-ratio)), 0.001)
    er_nodes = ['er-%i' %(i) for i in xrange(N)]

    cb_sigs, cb_nodes = mix((ba_sigs, ba_nodes),
                            (er_sigs, er_nodes), 0)

    deg_samples = mg_sample(N, cb_sigs, 80, 20)
    iso_n = np.sum(deg_samples.ravel() == 0)

    PA_para, PA_lk = mle(deg_samples, 'PA')
    ER_para, ER_lk = mle(deg_samples, 'ER')
    tr['ratio'].append((ratio, iso_n, PA_para, PA_lk, ER_para, ER_lk))
    print(ratio, iso_n, PA_para, PA_lk, ER_para, ER_lk)

zdump(tr, './ModelSelectionMix-ratio.pk')
