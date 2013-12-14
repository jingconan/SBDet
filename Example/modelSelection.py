#!/usr/bin/env python
from __future__ import print_function, division
from SBDet.Util import load, dump
from SBDet import mg_sample, mle
import matplotlib.pyplot as plt
import numpy as np


def plot_diff_sample_graph(sigs):
    beta_hat_v = []
    lk_er_v = []

    n_set = xrange(10, 500, 20)
    for n in n_set:
        s_v = mg_sample(n=n, k=100, **sigs)
        beta_hat, lk_er = mle(s_v, 'ER')
        beta_hat_v.append(beta_hat)
        lk_er_v.append(lk_er)

    plt.subplot(211)
    plt.plot(n_set, beta_hat_v)
    plt.subplot(212)
    plt.plot(n_set, lk_er_v)
    plt.show()
    res = {
        'beta_hat_v': beta_hat_v,
        'lk_er_v': lk_er_v,
        'n_set': n_set,
    }
    dump(res, './ER_model_selection_with_num_sampled_graph.pk')


def plot_diff_one_graph(sigs):
    beta_hat_v = []
    lk_er_v = []

    k_set = xrange(10, 500, 20)
    for k in k_set:
        s_v = mg_sample(n=100, k=k, **sigs)
        beta_hat, lk_er = mle(s_v, "ER")
        beta_hat_v.append(beta_hat)
        lk_er_v.append(lk_er)

    res = {
        'beta_hat_v': beta_hat_v,
        'lk_er_v': lk_er_v,
        'k_set': k_set,
    }
    dump(res, './ER_model_selection_with_num_sampled_points_2.pk')

    plt.subplot(211)
    plt.plot(k_set, beta_hat_v)
    plt.subplot(212)
    plt.plot(k_set, lk_er_v)
    plt.show()


def select_caida_backbone():
    ROOT = '/home/wangjing/LocalResearch/CyberData/CaidaData/'
    T = 4.33
    dur_set = np.linspace(0.1, T*0.9, 20)
    tr = dict(alphav=[], lkav=[], betav=[], lkbv=[], dur=[])
    for dur in dur_set:
        print('dur', dur)
        f_name = ROOT + 'passive-2013-sigs-%f/sigs.pk' % (dur)
        sigs = load(f_name)
        s_v = mg_sample(n=min([4, len(sigs['sig_edges'])]), k=200, **sigs)
        alpha, lka = mle(s_v, 'BA')
        beta, lkb = mle(s_v, 'ER')
        tr['dur'].append(dur)
        tr['alphav'].append(alpha)
        tr['betav'].append(beta)
        tr['lkav'].append(lka)
        tr['lkbv'].append(lkb)
    dump(tr, './model-select-caida-backbone.pk')


def select_simple_pkt():
    ROOT = '/home/wangjing/LocalResearch/CyberData/CaidaData/'
    T = 66095.977196
    # msv = []
    dur_set = np.linspace(10, T*0.9, 50)
    tr = dict(alphav=[], lkav=[], betav=[], lkbv=[], dur=[])
    for dur in dur_set:
        print('dur', dur)
        f_name = ROOT+'sigs1/loc6-%i/sigs.pk' % (dur)
        sigs = load(f_name)
        s_v = mg_sample(n=min([4, len(sigs['sig_edges'])]), k=400, **sigs)
        alpha, lka = mle(s_v, 'BA')
        beta, lkb = mle(s_v, 'ER')
        tr['dur'].append(dur)
        tr['alphav'].append(alpha)
        tr['betav'].append(beta)
        tr['lkav'].append(lka)
        tr['lkbv'].append(lkb)
    dump(tr, './model-select-simple-pkt.pk')


if __name__ == "__main__":
    # select_caida_backbone()
    select_simple_pkt()

    # plot_diff_sample_graph(sigs)
    # plot_diff_one_graph(sigs)

    # print('s_v', np.max(s_v))
    # beta_hat, lk_er = ER_MLE(s_v)
    # print('lk_er', lk_er)
    # print('beta_hat', beta_hat)
    # import ipdb;ipdb.set_trace()

    # BA_MLE(s_v)

    # xc = np.arange(2.0, 100.0, 0.1)
    # res = phi(xc)
    # plt.plot(xc, res)
    # plt.show()
    # print('res', res)
    # import ipdb;ipdb.set_trace()


##############
