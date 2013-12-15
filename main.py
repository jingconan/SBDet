#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import parseToCoo, select_model
from SBDet import monitor_deg_dis, detect_botnet
from SBDet import mix, load, dump

if __name__ == "__main__":
    normal_sigs, normal_nodes = parseToCoo('../pcap2sigs-loc6-20070501-2055.sigs',
            undirected=True)
    botnet_sigs, botnet_nodes = parseToCoo('../ddostrace.20070804_134936-10.sigs', undirected=True)
    mix_sigs, mix_nodes = mix((normal_sigs, normal_nodes),
                              (botnet_sigs, botnet_nodes), 30)
    dump(dict(sigs=mix_sigs, nodes=mix_nodes), './mix_sigs.pk')
    # import ipdb;ipdb.set_trace()
    data = load('./mix_sigs.pk')
    # import ipdb;ipdb.set_trace()
    model, para = select_model(normal_sigs[:3])
    print('model: ', model, ' para: ', para)
    divs = monitor_deg_dis(data['sigs'][0:100], model, [para, 1e-10])
    print('divs', divs)
    # divs = monitor_deg_dis(data['sigs'][0:100], 'BA', [0.1, 1e-10])
    import matplotlib.pyplot as plt
    plt.plot(divs)
    import ipdb;ipdb.set_trace()

    pass
