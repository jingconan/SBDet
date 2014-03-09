#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import parseToCoo, select_model
from SBDet import monitor_deg_dis, detect_botnet
from SBDet import mix, load, dump, gen_sigs
import numpy as np
import scipy as sp
import pylab as P


def old_pk_to_coo(data, undirected):
    nodes = data['nodes']
    sigs = data['sig_edges']
    sparse_sigs = []
    g_size = len(nodes)
    for g in sigs:
        N = len(g)
        I, J = zip(*zip(*g)[0])
        if undirected:
            IJMat = np.array([I, J], copy=True)
            I = np.min(IJMat, axis=0)
            J = np.max(IJMat, axis=0)
        mat = sp.sparse.coo_matrix((sp.ones(N,), (I, J)),
                shape=(g_size, g_size))
        sparse_sigs.append(mat)
    return sparse_sigs, nodes

def old_caida():
    ROOT = '/home/wangjing/LocalResearch/CyberData/caida-data/'
    T = 4.33
    dur_set = np.linspace(0.1, T*0.9, 20)

    lk_list = []
    for dur in dur_set:
        dat = load(ROOT+'passive-2013-sigs-%f/sigs.pk' % (dur))
        sigs, nodes = old_pk_to_coo(dat, True)
        print('len sigs', len(sigs))
        # model, para, debug_ret = select_model(len(nodes), sigs, min([40, len(sigs)]), 200, True)
        model, para, debug_ret = select_model(len(nodes), sigs, len(sigs), 200, True)
        lk_list.append(debug_ret['ER'][2])
        print('log likelihood value', debug_ret['ER'][2])
        print('para', para)
        print('model', model)

    dump({'x':dur_set, 'y':lk_list, 'x_name':'dur_set', 'y_name':'lk_list'},
            './caida-backbone-er-different-window-size-large-sample.pk')

    P.plot(dur_set, lk_list)
    # P.plot(sdd)
    P.show()
    import ipdb;ipdb.set_trace()


def validate_select_model_with_BA(m):
    N_set = [50, 150, 250, 350]
    dat = dict()
    dat['N'] = N_set
    dat['ret'] = []
    for N in N_set:
        normal_sigs = gen_sigs('BA', 100, N, m)
        normal_nodes = range(N)
        model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)

        dat['ret'].append(debug_ret)

    dump(dat, 'validate_select_model_with_BA-m-%i.pk' % (m))


def validate_select_model_with_power_law(p):
    N_set = [50, 150, 250, 350]
    dat = dict()
    dat['N'] = N_set
    dat['ret'] = []
    for N in N_set:
        normal_sigs = gen_sigs('powerlaw_cluster_graph', 100, N, 2, p)
        normal_nodes = range(N)
        model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)

        dat['ret'].append(debug_ret)
    # import ipdb;ipdb.set_trace()

    dump(dat, 'validate_select_model_with_power_law-p-%s.pk' % (p))

def validate_select_model_with_ER(N):
    p_set = np.linspace(0.0001, 0.05, 10)
    dat = dict()
    dat['p_set'] = p_set
    dat['ret'] = []
    for p in p_set:
        normal_sigs = gen_sigs('ER', 100, N, p)
        normal_nodes = range(N)
        model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)
        dat['ret'].append(debug_ret)

    dump(dat, 'validate_select_model_with_ER-N-%s.pk' % (N))



def release(dat, models, pos):
    res = []
    for model in models:
        res.append([])
        for ss in dat['ret']:
            res[-1].append(ss[model][pos])
    return res


def plotv(name):
    dat = load(name)
    lk1, lk2, lk3 = release(dat, ['CHJ', 'BA', 'ER'], 2)
    P.subplot(211)
    P.plot(lk1, 'o--', lw=2, ms=15)
    P.plot(lk2, '>g', lw=2, ms=15)
    P.plot(lk3, 'xr-', lw=2, ms=15)
    P.legend(['CHJ', 'BA', 'ER'], loc=4)
    # P.title('log likelihood vs. n for BA(n, %i)' % (m))

    P.subplot(212)
    p1, p2, p3 = release(dat, ['CHJ', 'BA', 'ER'], 1)
    P.plot(p1, 'o--', lw=2, ms=15)
    P.plot(p2, '>g', lw=2, ms=15)
    P.plot(p3, 'xr-', lw=2, ms=15)
    P.legend(['CHJ', 'BA', 'ER'], loc=4)
    # P.title('parameters vs. n for BA(n, %i)' % (m))
    P.show()


def loc6():
    normal_sigs, normal_nodes = parseToCoo('../loc6-20070501-100.sigs',
            undirected=True)
    model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)
    import ipdb;ipdb.set_trace()

def caida():
    # normal_sigs, normal_nodes = parseToCoo('../loc6-20070501-100.sigs',
    #         undirected=True)
    # model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)
    # import ipdb;ipdb.set_trace()

    # ROOT = '/home/wangjing/LocalResearch/CyberData/caida-data/passive-2013/'
    ROOT = '../'
    normal_sigs, normal_nodes = parseToCoo(ROOT + \
            'equinix-sanjose.dirA.20130117-125912.UTC.anon-10.sigs',
            undirected=True, first_k=None)

    botnet_sigs, botnet_nodes = parseToCoo('../ddostrace.20070804_134936-10.sigs', undirected=True)

    model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 4, 1000, True)
    data_set = normal_sigs * 3  + botnet_sigs + normal_sigs * 3
    divs1 = monitor_deg_dis(data_set, 'CHJ', [debug_ret['CHJ'][1], 1e-10])
    divs3 = monitor_deg_dis(data_set, 'BA', [debug_ret['BA'][1], 1e-10])
    divs2 = monitor_deg_dis(data_set, 'ER', [debug_ret['ER'][1], 1e-10])
    dump_data = {'d1': divs1, 'd2': divs2, 'd3': divs3}
    dump(dump_data, './CHJ_overperform_BA.pk')
    P.subplot(311)
    P.plot(divs1)
    P.subplot(312)
    P.plot(divs2)
    P.subplot(313)
    P.plot(divs3)
    P.show()


    import ipdb;ipdb.set_trace()





if __name__ == "__main__":
    # loc6()
    caida()
    # validate_select_model_with_power_law(0.5)
    # plotv('validate_select_model_with_power_law-p-0.5.pk')

    # validate_select_model_with_ER(500)
    # plotv('./validate_select_model_with_ER-N-500.pk')
    # plot(v)
    # validate_select_model_with_BA(1)
    # plot_validate_select_model_with_BA(1)
    # old_caida()

    # ROOT = '/home/wangjing/LocalResearch/CyberData/caida-data/passive-2013/'
    # normal_sigs, normal_nodes = parseToCoo('../loc6-20070501-2055-500000.sigs',
    # normal_sigs, normal_nodes = parseToCoo(ROOT + 'equinix-sanjose.dirA.20130117-125912.UTC.anon-10.sigs',
            # undirected=True, first_k=None)
    # normal_sigs = gen_sigs('BA', 100, 100, 1)
    # normal_nodes = range(100)

    # normal_sigs *= 10
    # botnet_sigs, botnet_nodes = parseToCoo('../ddostrace.20070804_134936-10.sigs', undirected=True)
    # import ipdb;ipdb.set_trace()
    # mix_sigs, mix_nodes = mix((normal_sigs, normal_nodes),
                              # (botnet_sigs, botnet_nodes), 20)
    # import ipdb;ipdb.set_trace()
    # dump(dict(sigs=mix_sigs, nodes=mix_nodes), './mix_sigs.pk')
    # import ipdb;ipdb.set_trace()
    # data = load('./mix_sigs.pk')
    # import ipdb;ipdb.set_trace()
    # model, para, debug_ret = select_model(len(normal_nodes), normal_sigs, 50, 50, True)
    # import ipdb;ipdb.set_trace()
    # divs = monitor_deg_dis(data['sigs'][0:100], model, [para, 1e-10])
    # import ipdb;ipdb.set_trace()
    # print('select model [', model, '] para [', para, ']')
    # divs = monitor_deg_dis(data['sigs'][0:100], model, [para, 1e-10])
    # print('divs', divs)
    # divs = monitor_deg_dis(data['sigs'][0:100], 'BA', [0.1, 1e-10])
    # import matplotlib.pyplot as plt
    # plt.plot(divs)
    # plt.show()
    # divs1 = monitor_deg_dis(normal_sigs * 3 + botnet_sigs + normal_sigs * 3, 'CHJ', [0.6993162, 1e-10])
    # divs2 = monitor_deg_dis(normal_sigs * 3 + botnet_sigs + normal_sigs * 3, 'ER', [0.18, 1e-10])
    # divs2 = monitor_deg_dis(normal_sigs, 'BA', [1.32575282, 1e-10])
    # plt.subplot(211)
    # plt.plot(divs1)
    # plt.subplot(212)
    # plt.plot(divs2)
    # plt.show()
    # import ipdb;ipdb.set_trace()

    pass
