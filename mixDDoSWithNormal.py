#!/usr/bin/env python
from __future__ import print_function, division

SADIT_ROOT = '/home/wangjing/Dropbox/Research/CyberSecurity/CommunityDetection/sadit_with_traffic_graph/sadit'
NORMAL_DATA = '/home/wangjing/LocalResearch/CyberSecurity/CommunityDetectionNormalData/scale_free_n_200_edge_642/n0_flow.txt'
DDOS_DATA = './Result/flows.txt'

import os
os.environ['SADIT_ROOT'] = SADIT_ROOT
import sys
sys.path.insert(0, SADIT_ROOT.rstrip('sadit'))
sys.path.insert(0, SADIT_ROOT)

import csv

from sadit.Detector.Data import HDF_FS
from sadit.Detector.Data import HDF_FlowExporter
from sadit.Detector.Data import PreloadHardDiskFile
import numpy as np
from Util import zdump, zload

class HDF_DumpFS(HDF_FS):
    def __init__(self, f_name):
        self.f_name = f_name
        self.table = zload(f_name)
        self.row_num = self.table.shape[0]

        self.t = np.array([t for t in self.get_rows('start_time')])
        t_idx = np.argsort(self.t)
        self.table = self.table[t_idx]
        self.t = self.t[t_idx]

        self.min_time = min(self.t)
        self.max_time = max(self.t)


def adjust_normal_data(f_name, out_f_name, fps, fsm):
    # nd = HDF_FS(f_name)
    nd = HDF_DumpFS('./Result/sampled_dump_orig.pkz')

    # adjust flow rate
    dur = nd.max_time - nd.min_time
    # sampled_row_nums = int(nd.row_num * 1.0 / 57.5)
    sampled_row_nums = int(dur * fps)
    rn = np.linspace(0, nd.row_num-1, sampled_row_nums).astype(int)
    samped_rows = nd.table[rn]
    # samped_rows['flow_size'] -= 4000
    f_max = np.max(samped_rows['flow_size'])
    f_min = np.min(samped_rows['flow_size'])
    samped_rows['flow_size'] -= f_min


    # adjust flow size
    fs = samped_rows['flow_size']
    ndm = np.mean(fs)
    samped_rows['flow_size'] *= (fsm * 1.0 / ndm)

    zdump(samped_rows, out_f_name)

# ndf = HDF_DumpFS('./Result/sampled_dump.pkz')
adjust_normal_data(NORMAL_DATA, './Result/sampled_dump.pkz', 8.8, 12494)
# adjust_normal_data(NORMAL_DATA, './Result/sampled_dump.pkz', 8.8, 1e5)

ip_to_dotted = lambda ip: ".".join(str(d)for d in ip)

class HDF_Merge(PreloadHardDiskFile):
    DT = np.dtype([
        ('start_time', np.float64, 1),
        ('src_ip', np.uint8, (4,)),
        ('dst_ip', np.uint8, (4,)),
        ('prot', np.str_, 5),
        ('flow_size', np.float64, 1),
        ('duration', np.float64, 1),
    ])
    def __init__(self, files):
        # self.table = zload(f_name)
        self.row_num = sum(f.row_num for f in files)
        self.table = np.zeros(shape=(self.row_num,), dtype=self.DT)
        cur_idx = 0
        for f in files:
            res = f.get_rows(fields=list(self.DT.names))
            # due to limitation of np.array. The attributes in each data
            # storage should have the same relative order
            # print('res.dtype.names', res.dtype.names)
            # print('self.DT.names', self.DT.names)
            assert(self.DT.names == res.dtype.names)
            self.table[cur_idx:(cur_idx + f.row_num)] = res
            cur_idx += f.row_num

        self.t = np.array([t for t in self.get_rows('start_time')])
        t_idx = np.argsort(self.t)
        self.table = self.table[t_idx]
        self.t = self.t[t_idx]

        self.min_time = min(self.t)
        self.max_time = max(self.t)

    def export(self, f_name):
        with open(f_name, 'w') as csv_f:
             writer = csv.writer(csv_f, delimiter=' ')
             # writer.writerows(self.table)
             for i in xrange(self.row_num):
                 writer.writerow([self.table[i]['start_time'],
                                  ip_to_dotted(self.table[i]['src_ip']),
                                  ip_to_dotted(self.table[i]['dst_ip']),
                                  self.table[i]['prot'],
                                  self.table[i]['flow_size'],
                                  self.table[i]['duration'],
                                  ])

normal = HDF_DumpFS('./Result/sampled_dump.pkz')
normal.shift_time(-1 * normal.t[0])
ddos = HDF_FlowExporter(DDOS_DATA)
ddos.shift_time(-1 * ddos.t[0])
ddos.shift_time(2000)

merge = HDF_Merge([normal, ddos])
merge.export('./Result/merged_flows.csv')
# import matplotlib.pyplot as plt
# plt.plot(merge.t, merge.get_rows('flow_size') / 1e3, 'b+')
# plt.ylim([0, 1e2])
# plt.xlabel('time')
# plt.ylabel('flow size (KB)')
# plt.savefig('flow_size_test_data_small_range.pdf')
# plt.savefig('flow_size_test_data.pdf')
# plt.show()
# import ipdb;ipdb.set_trace()


# Add data to some segment
