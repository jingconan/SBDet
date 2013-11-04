#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import *
import pylab as P


#### Calculate Social Interaction Graph ####
# sigs = cal_SIG('./Result/merged_flows.csv', tp='igraph')
# animate_SIGs(sigs, './Animation/')
# dump(sigs, './sigs.pkz')


#### Verify whether normal SIGs satisfies ERGM ####
# sigs = load('./sigs.pkz')
# bv = P.linspace(0.0005, 1, 100)
# deviations = verify_ERGM(sigs[0:200], 'igraph', beta_values=bv)
# P.plot(bv, deviations)
# P.grid(which='major')
# P.title('K-L divergence of normal degree dist and ERGM')
# P.xlabel('$\\beta$')
# P.ylabel('K-L divergence')
# P.savefig('./Result/fitness_normal_dd_ERGM.pdf')
# P.show()

#### Monitor the degree distribution ####
sigs = load('./sigs.pkz')
divs_KL = monitor_deg_dis(sigs, sigs[0:200], 'igraph', KL_div)
divs_LDP = monitor_deg_dis(sigs, sigs[0:200], 'igraph', LDP_div)
P.subplot(211)
P.plot(divs_KL)
P.title('K-L div. of all windows')
P.ylabel('K-L divergence')
P.subplot(212)
P.plot(divs_LDP)
P.title('LDP div. of all windows')
P.ylabel('LDP divergence')
P.xlabel('time (s)')
P.savefig('./Result/comparison_KL_LDP.pdf')
# P.title('')
P.show()
