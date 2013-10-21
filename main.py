#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Util import zdump, zload
from FlowExporter import pcap2flow
from CGraph import Igraph, NetworkXGraph


def visualize_caida():
    # data_file = '/home/wangjing/LocalResearch/ddos-20070804/ddostrace.20070804_134936.pcap'
    data_file = '/home/wangjing/LocalResearch/CyberData/CaidaData/ddos-20070804/ddostrace.20070804_134936.pcap'
    # data_file = '../imalse_ddos_ping_data/trace-8-1.pcap'
    # data_file = '../imalse_ddos_ping_data/trace-1-1.pcap'
    pcap2flow(data_file, './Result/flows.txt', 1)
    node_info = {}
    execfile('./node_info.py', node_info)

    interval = 10
    dur = 10
    layout = None

    # ani_folder = './imalse_ddos_n_0_3/'
    ani_folder = './caida2/'
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    for i in xrange(50):
        print('i=', i)
        # tg = NetworkXGraph('./test.txt', node_info)
        tg = Igraph('./Result/flows.txt', node_info)
        ips = tg.get_vertices()
        tg.add_vertices(ips)
        # records = tg.filter(prot='UDP', rg=[i * interval, i * interval + dur], rg_type='time')
        records = tg.filter(prot=None, rg=[i * interval, i * interval + dur], rg_type='time')
        edges = tg.get_edges(records)
        tg.add_edges(edges)
        layout = tg.gen_layout() if layout is None else layout
        tg.plot(ani_folder + "%04d.png" % (i), layout=layout)


def cal_corrcoef():

    # data_file = '/home/wangjing/LocalResearch/ddos-20070804/ddostrace.20070804_134936.pcap'
    # pcap2flow(data_file, './Result/flows.txt', 1)
    interval = 10
    dur = 10

    adj_mats = []
    for i in xrange(28):
        print('i=', i)
        tg = NetworkXGraph('./Result/flows.txt')
        ips = tg.get_vertices()
        tg.add_vertices(ips)
        # import ipdb;ipdb.set_trace()

        records = tg.filter(prot=None, rg=[i * interval, i * interval + dur], rg_type='time')
        edges = tg.get_edges(records)
        tg.add_edges(edges)
        am = np.array(nx.adj_matrix(tg.graph), copy=True)
        adj_mats.append(am[:, 93])

    adj_mats = np.asarray(adj_mats)
    npcor = np.corrcoef(adj_mats, rowvar=0)
    np_cor_no_nan = np.nan_to_num(npcor)
    plt.pcolor(np_cor_no_nan)
    plt.colorbar()
    plt.show()
    zdump(npcor, './Result/npcor.pkz')
    import ipdb;ipdb.set_trace()

    # import ipdb;ipdb.set_trace()


def ana_corrcoef():
    tg0 = Igraph('./Result/flows.txt', None)
    ips = tg0.get_vertices()
    tg0.add_vertices(ips)
    records = tg0.filter(prot=None, rg=[0, 10], rg_type='time')
    edges = tg0.get_edges(records)
    tg0.add_edges(edges)
    layout = tg0.gen_layout()

    ani_folder = './caida_corref_network/'
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    npcor = zload('./Result/npcor.pkz')
    cor = np.nan_to_num(npcor)

    i = 0
    for cor_th in np.linspace(0, 1, 20):
        # get edges
        adj = (cor > cor_th)
        edges_list = zip(*adj.nonzero())
        edges = dict(zip(edges_list, [0] * len(edges_list)))
        # import ipdb;ipdb.set_trace()

        tg = Igraph('./Result/flows.txt', None)
        ips = tg.get_vertices()
        tg.add_vertices(ips)
        # records = tg.filter(prot='UDP', rg=[i * interval, i * interval + dur], rg_type='time')
        # edges = tg.get_edges(records)
        tg.add_edges(edges)
        tg.plot(ani_folder + "cor-%04d-%f.png" % (i, cor_th), layout=layout)
        i += 1


def degree_distribution():
    data_file = '/home/wangjing/LocalResearch/ddos-20070804/ddostrace.20070804_134936.pcap'
    pcap2flow(data_file, './Result/flows.txt', 1)
    interval = 10
    dur = 10

    dd_list = []
    for i in xrange(28):
        print('i=', i)
        tg = Igraph('./Result/flows.txt')
        ips = tg.get_vertices()
        tg.add_vertices(ips)

        records = tg.filter(prot=None, rg=[i * interval, i * interval + dur], rg_type='time')
        edges = tg.get_edges(records)
        tg.add_edges(edges)
        dd = list(tg.graph.degree_distribution().bins())
        dd_list.append(dd)

    zdump(dd_list, './Result/dd.pkz')


def plot_dd():
    dd_list = zload('./Result/dd.pkz')
    ani_folder = './Result/degree_dist/'
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    i = -1
    for dd in dd_list:
        i += 1
        left, right, height = zip(*dd)
        width = right[0] - left[0]
        ind = (np.array(left) + np.array(right)) / 2.0
        plt.bar(ind, height, width)
        plt.xlim([0, 130])
        plt.savefig(ani_folder + "dd-%04d.png" % (i))


if __name__ == "__main__":
    cal_corrcoef()
    # visualize_caida()
    # cal_corrcoef()
    # ana_corrcoef()
    # degree_distribution()
    # plot_dd()
