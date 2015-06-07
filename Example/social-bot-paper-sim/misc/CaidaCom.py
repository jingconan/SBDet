from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P

# f_name = '/home/wangjing/LocalResearch/CyberData/caida-data/ddos-20070804/ddostrace.20070804_134936-10-test.sigs'
# f_name = '/home/wangjing/LocalResearch/CyberData/caida-data/ddos-20070804/ddostrace.20070804_134936-1-test.sigs'
f_name = '/home/wangjing/LocalResearch/CyberData/caida-data/ddos-20070804/ddostrace.20070804_134936-10-tcp.sigs'
# botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
botnet_sigs, botnet_nodes = parseToCoo(f_name,
                                       undirected=True)

edge_num = [len(sig.nonzero()[0]) for sig in botnet_sigs]
print('node_num', len(botnet_nodes))
print('edge_num', edge_num)

import ipdb;ipdb.set_trace()


