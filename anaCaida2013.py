#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
from Util import load, zload, dump
import sys
from SBDet import *
# import matplotlib.pyplot as plt
import numpy as np
import pylab as P
import networkx as nx
from loadSIGs import *


if __name__ == "__main__":
    ROOT = '/home/wangjing/LocalResearch/CyberData/CaidaData/'
    # dur = 1
    T = 4.33
    dur_set = np.linspace(0.1, T*0.9, 20)

    # data = HDF_tshark(ROOT + 'passive-2013/equinix-sanjose.dirA.20130117-125912.UTC.anon_tshark.txt')
    msv = []
    for dur in dur_set:
        ms = percent_iso_nodes(ROOT+'passive-2013-sigs-%f/sigs.pk' % (dur))
        msv.append(ms)
    # P.plot(msv)
    # P.show()
    dump({'x':dur_set/T, 'y':msv}, './caida_backbone_iso_nodes_vs_ws.pk')

        # to_sigs(data, ROOT + 'passive-2013-sigs-%f/' % (dur), dur)
    # dd = plot_deg(ROOT + 'passive-2013-sigs-%f/sigs.pk' % (dur))
    # sdd = sorted(dd, reverse=True)
    # P.plot(sdd)
    # P.show()
    import ipdb;ipdb.set_trace()



