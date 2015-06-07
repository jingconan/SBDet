from SBDet import *
import pylab as P
data = zload('./cor-graph-360.pkz')
npcor = data['npcor']
nd = P.diag(npcor)
valid_idx = np.nonzero(nd == 1)[0]
npcor = npcor[valid_idx, :][:, valid_idx]
P.pcolor(npcor, cmap='Greys', vmin=0, vmax=1)
plt.colorbar()
n = npcor.shape[0]
P.axis([0, n, 0, n])
P.savefig('cor-graph-pcolor.pdf')
P.show()
# import ipdb;ipdb.set_trace()
