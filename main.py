#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import parseToCoo, select_model
from SBDet import monitor_deg_dis, detect_botnet
from SBDet import mix

if __name__ == "__main__":
    normal_sigs, normal_nodes = parseToCoo('../pcap2sigs-loc6-20070501-2055.sigs',
            undirected=True)
    botnet_sigs, botnet_nodes = parseToCoo('../ddostrace.20070804_134936-10.sigs')
    mix_sigs, mix_nodes = mix((normal_sigs, normal_nodes), (botnet_sigs, botnet_nodes), 0)
    import ipdb;ipdb.set_trace()
    pass
