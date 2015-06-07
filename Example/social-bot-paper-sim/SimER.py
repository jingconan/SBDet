from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P


botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
                                       undirected=True)
extra_n = 300
# bg_sigs = gen_sigs('ER', 100, len(botnet_sigs) + extra_n, 0.001)
bg_sigs = gen_sigs('BA', 100, len(botnet_sigs) + extra_n, 1)
bg_nodes = botnet_nodes + range(extra_n)

## Model Selection ##
deg_samples = mg_sample(len(botnet_sigs) + extra_n, bg_sigs, 80, 100)
iso_n = np.sum(deg_samples.ravel() == 0)


PA_para, PA_lk = mle(deg_samples, 'PA')
print('PA_para', PA_para)
print('PA_lk', PA_lk - 2.439260 * iso_n)
ER_para, ER_lk = mle(deg_samples, 'ER')
print('ER_lk', ER_lk)
print('ER_para', ER_para)

# md, para = select_model(deg_samples, ['PA','ER'])
# print('md', md)
# print('para', para)




cb_sigs, cb_nodes = mix((bg_sigs, bg_nodes),
                        (botnet_sigs, botnet_nodes), 50)

numel = lambda x: len(x.nonzero()[0])
nv = [numel(sig) for sig in cb_sigs]
# P.plot(nv)
# P.show()

import ipdb;ipdb.set_trace()
