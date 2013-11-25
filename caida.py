#! /usr/bin/env python
#! /home/wangjing/Apps/python/bin/python
####
from __future__ import print_function, division
from SBDet import *
import pylab as P
import networkx as nx
from subprocess import check_call
from Util import load, zload, dump

f_name = '/home/wangjing/LocalResearch/CyberData/CaidaData/t_shark/ddostrace.20070804_134936_tshark.txt'
# f_name = '/home/wangjing/LocalResearch/CyberData/CaidaData/t_shark/ddostrace.20070804_141436_tshark.txt'
data_file = HDF_tshark(f_name)

sigs = cal_SIG(data_file, interval=10.0,
               dur=10.0, rg=(0.0, 300.0),
               directed=False, tp='networkx')
# pk_f_name = '/home/wangjing/LocalResearch/CyberData/CaidaData/ddostrace.20070804_134936_sigs.pk'
# dump(sigs, pk_f_name)
import ipdb;ipdb.set_trace()
sigs = load(pk_f_name)
