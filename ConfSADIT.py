#!/usr/bin/env python
# !/home/wangjing/Apps/python/bin/python
from __future__ import print_function, division

SADIT_PATH = '/home/wangjing/Dropbox/Research/CyberSecurity/CommunityDetection/sadit_with_traffic_graph/sadit'
import sys
sys.path.insert(0, SADIT_PATH.rstrip('sadit'))
sys.path.insert(0, SADIT_PATH)

# from sadit.Experiment.GUITopoSim import get_inet_adj_mat, fix_fs_addr_prefix_bug
from sadit.Configure import gen_dot

# zeros = lambda s:[[0 for i in xrange(s[1])] for j in xrange(s[0])]
import numpy as np
import networkx as nx
# np_ver = [int(v) for v in np.__version__.split('rc')[0].split('.')]
# if np_ver[1] < 7:
#     raise Exception('must install numpy with version > 1.7')

# from numpy import zeros

from CGraph import NetworkXGraph
from FlowExporter import pcap2flow

# from random import randrange


def gen_rand_ips(num):
    not_valid = [10,127,169,172,192]
    ips = np.random.randint(1, 256, size=(num, 4))
    for val in not_valid:
        ips[ips[:, 0] == val, 0] += 1
    return ips

    # first = randrange(1,256)
    # while first in not_valid:
    #     first = randrange(1,256)

    # ip = first, randrange(1,256), randrange(1,256),randrange(1,256)

# Get topology
# adj = get_inet_adj_mat('../inet-3.0/inet_out.txt')
# adj = np.array(adj)
graph = nx.erdos_renyi_graph(100, 0.08)
# adj = np.ones((100, 100))
adj = np.asarray(nx.to_numpy_matrix(graph))
# import ipdb;ipdb.set_trace()

# generate flow data
# pcap_data_file = '/home/wangjing/LocalResearch/ddos-20070804/ddostrace.20070804_134936.pcap'
# pcap2flow(pcap_data_file, './Result/flows.txt', 1)

# get ip address
tg = NetworkXGraph('./Result/flows.txt')
ips = tg.get_vertices()
N = ips.shape[0]  # no. of know ip addresses

src_nodes, dst_nodes = adj.nonzero()
assert(len(src_nodes) * 2 >= N)
nodes = np.concatenate((src_nodes, dst_nodes))

# generate additional address
rand_addr_num = len(nodes) - N
rand_ips = gen_rand_ips(rand_addr_num)
all_ips = np.concatenate((ips, rand_ips), 0)
# assigned_nodes = np.random.choice(nodes, N)
perm = np.random.permutation(len(nodes))
all_ips = all_ips[perm, :].tolist()
M = len(nodes)
src_ips = all_ips[:(M // 2)]
ip_to_str = lambda ip: '.'.join(str(d) for d in ip)
src_ips_str = [ip_to_str(ip) for ip in src_ips]
dst_ips = all_ips[(M // 2):]
dst_ips_str = [ip_to_str(ip) for ip in dst_ips]


# Create Network Descriptor
net_desc = dict()
net_desc['ipv4_net_addr_base'] = '10.7.0.1/24'
net_desc['link_attr_default'] = ['2ms','5mbps']
net_desc['link_attr'] = {}
net_desc['link_to_ip_map'] = dict(zip(zip(src_nodes, dst_nodes),
                                      zip(src_ips_str,dst_ips_str)))
net_desc['topo'] = adj
net_desc['node_type'] = 'NNode'


g_size = adj.shape[0]
# norm_desc = dict()
# norm_desc['src_nodes'] = range(nodes_num)
# norm_desc['dst_nodes'] = range(nodes_num)
# norm_desc['TYPE'] = 'stationary'
# norm_desc['node_para'] = 'stationary'


#################################
##   Parameter For Normal Case ##
#################################
sim_t = 3600  # simulation time
start = 0  # start time
DEFAULT_PROFILE = ((sim_t,),(1,))

gen_desc1 = {
    'TYPE':'harpoon',  # type of flow generated, defined in fs
    'flow_size_mean':'4e3',  # flow size is normal distribution. Mean
    'flow_size_var':'100',  # variance
    'flow_arrival_rate':'0.1'  # flow arrival is poisson distribution. Arrival rate
}

norm_desc = dict(
    TYPE='stationary',
    start='0',
    sim_t=sim_t,
    node_para={
        'states': [gen_desc1],
    },
    profile=DEFAULT_PROFILE,
    src_nodes=range(g_size),
    dst_nodes=range(g_size),
)

gen_dot([], net_desc, norm_desc, './test.dot')

# print network settings
# import pprint
# def write_settings(ns, f_name):
#     with open(f_name, 'w') as ns_f:
#         for k, v in ns.iteritems():
#             ns_f.write(k + ' = ')
#             pprint.pprint(v, stream=ns_f)
#             ns_f.write('\n')

# write_settings(ns, './net_settings.py')
# import ipdb;ipdb.set_trace()
