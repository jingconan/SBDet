#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
from Util import load, zload, dump
import sys
from SBDet import *
import matplotlib.pyplot as plt
import numpy as np
import pylab as P
import networkx as nx

# def transform_and_dump():
#     folder = '/home/wangjing/LocalResearch/CyberData/CaidaData/normal_part_1_sigs/'
#     out_folder = '/home/wangjing/LocalResearch/CyberData/CaidaData/normal_part_1_sigs_2/'
# N = 1000
    # N = sys.maxint
    # try:
    #     dump(load('%s%i.pk' % (folder, 0))['nodes'], '%snodes.pk' % (out_folder))
    #     for i in xrange(N):
    #         sys.stdout.write("\r%d" % i)
    #         sys.stdout.flush()
    #         dat = load('%s%i.pk' % (folder, i))
    #         l = len(dat['nodes'])
    #         dump({'edges': dat['edges']}, '%s%i.pk' % (out_folder, i))
    # except:
    #     pass


    # import ipdb;ipdb.set_trace()

# transform_and_dump_compact()

def loadSIGS():
    sigs = load('/home/wangjing/LocalResearch/CyberData/CaidaData/normal_part_1_sigs_3/normal_part_1_sigs.pk')
    import ipdb;ipdb.set_trace()

# loadSIGS()


def to_sigs(data, out_folder, dur):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cal_SIG_low_mem(data,
                    interval=dur,
                    dur=dur,
                    folder=out_folder)
    pack_sigs(out_folder, out_folder+'sigs.pk')

# from subprocess import check_call
# def cvt():
#     folder = '/home/wangjing/LocalResearch/CyberData/CaidaData/bot-data/ddostrace.20070804_141436/'
#     for i in xrange(29):
#         fn = folder + 'ddostrace.20070804_141436' + str(i) + '.pk'
#         ftn = folder + str(i) + '.pk'
#         check_call(['mv', fn, ftn])

def plot_normal_no_edges(sig_file, pic_name):
    sig_dat = load(sig_file)
    en = [len(el) for el in sig_dat['sig_edges']]
    plt.plot(10 * np.arange(len(en)) / 3600, en)
    plt.xlabel('time (h)')
    plt.ylabel('edge number')
    plt.title('edge number in the normal SIGs')
    plt.savefig(pic_name)
    plt.close()

def sparsity(sig_file):
    sig_dat = load(sig_file)
    node_num = len(sig_dat['nodes'])
    print('node_num', node_num)
    en = [len(el) for el in sig_dat['sig_edges']]
    sparsity = np.array(en) / (node_num ** 2)
    mean_sparsity = np.mean(sparsity)
    return mean_sparsity

def percent_iso_nodes(sig_file):
    sig_dat = load(sig_file)
    node_num = len(sig_dat['nodes'])
    isonv = []
    for el in sig_dat['sig_edges']:
        if len(el) == 0:
            isonv.append(node_num)
            continue
        edges = zip(*el)[0]
        g = nx.Graph()
        g.add_nodes_from(range(node_num))
        g.add_edges_from(edges)
        ison = np.sum(np.array(g.degree().values()) == 0)
        isonv.append(ison)
    return np.mean(isonv) / node_num

def plot_deg(sig_file):
    sig_dat = load(sig_file)
    node_num = len(sig_dat['nodes'])
    isonv = []
    # ddc = Counter()
    dd = np.zeros((node_num,))
    for el in sig_dat['sig_edges']:
        if len(el) == 0:
            isonv.append(node_num)
            continue
        edges = zip(*el)[0]
        g = nx.Graph()
        g.add_nodes_from(range(node_num))
        g.add_edges_from(edges)
        # ddc += degree_distribution(g, 'networkx')
        dd += g.degree().values()
    return dd

if __name__ == "__main__":
    ROOT = '/home/wangjing/LocalResearch/CyberData/CaidaData/'
    # to_sigs(ROOT + 't_shark_prot_first/ddostrace.20070804_134936_tshark.txt',
    #         ROOT + 'bot-data2/ddostrace.20070804_134936/')

    # data = HDF_tshark(ROOT + 'loc6-20070501-2055_tshark_part1.txt')
    T = 66095.977196
    msv = []
    # dur_set = np.arange(10, 3000, 300)
    dur_set = np.linspace(10, T*0.9, 50)
    for dur in dur_set:
        # to_sigs(data, ROOT + 'sigs1/loc6-%i/' % (dur), dur)
        # plot_normal_no_edges(ROOT+'sigs1/loc6-%i/sigs.pk' % (dur),
        #                      ROOT+'edge-num/dur-%i.png' % (dur))
        # ms = sparsity(ROOT+'sigs1/loc6-%i/sigs.pk' % (dur))
        # ms = percent_iso_nodes(ROOT+'sigs1/loc6-%i/sigs.pk' % (dur))
        ddc = plot_deg(ROOT+'sigs1/loc6-%f/sigs.pk' % (dur))
        import ipdb;ipdb.set_trace()
        # msv.append(ms)

    # dump({'x':dur_set/T, 'y':msv}, './percent_nodes.pk')
    # dump({'x':dur_set/T, 'y':msv}, './sparsity_vs_window.pk')

    # P.plot(dur_set/T, msv, '+-', ms=18, lw=3)
    # P.xlabel('window size / total time')
    # P.ylabel('sparsity')
    # P.title('sparsity vs. window size')
    # P.savefig('sparsity.pdf')
    # P.show()
    import ipdb;ipdb.set_trace()


    # to_sigs(ROOT + 't_shark_prot_first/ddostrace.20070804_141436_tshark.txt',
    #         ROOT + 'bot-data2/ddostrace.20070804_141436/')
    # compact_dump(ROOT + 'bot-data/ddostrace.20070804_141436/',
    #              ROOT + 'bot-data/ddostrace.20070804_141436/compact/')

    # cvt()

    # to_sigs(ROOT + 't_shark/ddostrace.20070804_134936_tshark.txt',
    #         ROOT + 'bot-data/ddostrace.20070804_134936/')
    # compact_dump(ROOT + 'bot-data/ddostrace.20070804_134936/',
    #              ROOT + 'bot-data/ddostrace.20070804_134936/compact/')

    # plot_normal_no_edges(ROOT + 'bot-data/normal_part_1_compact.pk')

