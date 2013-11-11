from __future__ import print_function, division, absolute_import
"""  generate topology

"""
import numpy as np
import networkx as nx
# from subprocess import check_call
from .Util import save_graph, zdump
# from Util import zdump, zload, log
from .Data import HDF_Merge


def get_overlay_topology(N, f_name=None):
    g = nx.scale_free_graph(N)
    g = g.to_undirected()
    cg_list = nx.connected_component_subgraphs(g)
    if f_name:
        save_graph(cg_list[0], f_name)
    # zdump(np.asarray(nx.to_numpy_matrix(cg_list[0])), 'overlay_adj.pkz')
    return nx.to_scipy_sparse_matrix(cg_list[0])


def makeup_address(M, ips):
    def gen_rand_ips(num):
        not_valid = [10, 127, 169, 172, 192]
        ips = np.random.randint(1, 256, size=(num, 4))
        for val in not_valid:
            ips[ips[:, 0] == val, 0] += 1
        return ips

    rand_addr_num = M - ips.shape[0]
    rand_ips = gen_rand_ips(rand_addr_num)
    all_ips = np.concatenate((ips, rand_ips), 0)
    perm = np.random.permutation(M)
    all_ips = all_ips[perm, :].tolist()
    src_ips = all_ips[:(M // 2)]
    ip_to_str = lambda ip: '.'.join(str(d) for d in ip)
    src_ips_str = [ip_to_str(ip) for ip in src_ips]
    dst_ips = all_ips[(M // 2):]
    dst_ips_str = [ip_to_str(ip) for ip in dst_ips]
    return src_ips_str, dst_ips_str


def create_net_desc(adj, ips):
    src_nodes, dst_nodes = adj.nonzero()
    # print('src_nodes', src_nodes.shape)
    # import ipdb;ipdb.set_trace()
    # assert(len(src_nodes) * 2 >= ips.shape[0])
    nodes = np.concatenate((src_nodes, dst_nodes))
    # import ipdb;ipdb.set_trace()
    src_ips_str, dst_ips_str = makeup_address(len(nodes), ips)

    net_desc = {
        'ipv4_net_addr_base': '10.7.0.1/24',
        'link_attr_default': ['2ms', '5mbps'],
        'link_attr': {},
        'link_to_ip_map': dict(zip(zip(src_nodes, dst_nodes),
                               zip(src_ips_str, dst_ips_str))),
        'topo': adj,
        'node_type': 'NNode',
    }
    return net_desc


def create_normal_desc(sim_t, g_size):
    gen_desc1 = {
        'TYPE': 'harpoon',  # type of flow generated, defined in fs
        'flow_size_mean': '4e3',  # flow size is normal distribution. Mean
        'flow_size_var': '100',  # variance
        'flow_arrival_rate': '0.02',  # flow arrival is poisson distribution.
    }

    norm_desc = dict(
        TYPE='stationary',
        start='0',
        sim_t=sim_t,
        node_para={
            'states': [gen_desc1],
        },
        profile=((sim_t,), (1,)),
        src_nodes=range(g_size),
        dst_nodes=range(g_size),
    )
    return norm_desc


def sample_traffic(data, ratio, out_name):
    step = int(1.0 / ratio)
    sampled_data = data.table[::step]
    zdump(sampled_data, out_name)


def mix_traffic(*args):
    assert(len(args) % 2 == 0)
    traf_vec = []
    for i in xrange(0, len(args), 2):
        traf = args[i]
        traf.shift_time(-1 * traf.t[0] + args[i+1])
        traf_vec.append(traf)
    return HDF_Merge(traf_vec)



# src_ips_str, dst_ips_str = makeup_address(len(nodes), ips)
# net_desc = create_net_desc(src_nodes, src_ips_str, dst_nodes, dst_ips_str, adj)
# norm_desc = create_normal_desc(3000, adj.shape[0])

# log('generate %s ' % (DOT_FILE))
# gen_dot([], net_desc, norm_desc, DOT_FILE)

# log('run simulator')
# os.environ['SADIT_ROOT'] = SADIT_PATH
# check_call(['python', SADIT_PATH + '/Simulator/fs.py', '-t', str(sim_t),
#             DOT_FILE])
