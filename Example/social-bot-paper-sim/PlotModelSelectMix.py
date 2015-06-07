from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
import numpy as np

tr = zload('./ModelSelectionMix-ratio.pk')
ratio, iso_n, PA_para, PA_lk, ER_para, ER_lk= zip(*tr['ratio'])
x = ratio
y = np.array(PA_lk).reshape(-1) - np.array(ER_lk) - 2.439260 * np.array(iso_n)
print('iso_n', iso_n)
P.plot(x, y, 'b+-', ms=15)
P.plot([0, max(x)], [0, 0], 'r--')

# import ipdb;ipdb.set_trace()
P.xlim([0, 1])
P.show()

