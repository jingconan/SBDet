import os
import pylab as pl
from Util import zload

DATA_FOLDER = '/home/wangjing/LocalResearch/CyberSecurity/CommunityDetectionNormalData/scale_free_n_200_edge_642/'
OVERLAY_FILE = './overlay_adj.pkz'
def get_file_size(n):
    return os.path.getsize(DATA_FOLDER + 'n%i_flow.txt' % (n))

file_size = []
for i in xrange(200):
    fs = get_file_size(i)
    file_size.append(fs)

# sfs = sorted(file_size, reverse=True)
sfs = pl.asarray(file_size) / 1e6
pl.subplot(211)
pl.plot(sfs)
pl.title('size of data collected in each node')
pl.ylabel('data size (MB)')
pl.subplot(212)
adj = zload(OVERLAY_FILE)
deg = pl.sum(adj, axis=1)
# sdeg = sorted(deg, reverse=True)
sdeg = deg
pl.plot(sdeg)
pl.title('degrees no. for each node')
pl.xlabel('node sequence')
pl.ylabel('degree no.')
pl.savefig('data_deg.pdf')
pl.show()

