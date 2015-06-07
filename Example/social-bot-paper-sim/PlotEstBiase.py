from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
import numpy as np

# tr = zload('./EstBiase-n_num.pk')
tr = zload('./EstBiase-n_num-revised.pk')
n_num, iso_n, PA_para, PA_lk, ER_para, ER_lk = zip(*tr['n_num'])
x = iso_n
y = np.array(PA_lk).reshape(-1) - np.array(ER_lk)
P.plot(x, y, 'r.')
# import ipdb;ipdb.set_trace()
P.axis([0, max(x), 0, max(y)])

# solve least square problem
n = len(x)
# X = np.hstack([np.ones((n, 1)), np.array(x).reshape(-1, 1)])
X = np.array(x).reshape(-1, 1)
beta = P.dot(P.dot(P.inv(P.dot(X.T, X)), X.T), np.array(y).reshape(-1, 1))
xp = np.linspace(0, max(x), 100)
yp = xp * beta[0, 0]
P.plot(xp, yp, 'b--')
P.xlabel('Number of isolated nodes')
P.ylabel('Difference between log liklihood of PA and ER')
xm = max(x) * 0.5
P.text(xm, xm * beta - 500, 'y=%f x' %(beta))
P.show()

