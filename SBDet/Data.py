"""  Defines all the wrapper classes for a variety of `data types`
    There are two categrories:
        1. Files in hard disk. the base class is the :class:`PreloadHardDiskFile`.
        2. MySQL database. The base class is the :class:`MySQLDatabase`.
"""
from __future__ import print_function, division, absolute_import
from .Util import abstract_method
from .Util import Find, DataEndException
from .Util import zload
# from Util import np
import numpy as np
import csv


class Data(object):
    """abstract base class for data. Data class deals with any implementation
    details of the data. it can be a file, a sql data base, and so on, as long
    as it supports the pure virtual methods defined here.
    """
    def get_rows(self, fields=None, rg=None, rg_type=None):
        """ get a slice of feature

        Parameters
        ---------------
        fields : string or list of string
            the fields we need to get
        rg : list of two floats
            is the range for the slice
        rg_type : str,  {'flow', 'time'}
            type for range

        Returns
        --------------
        list of list

        """
        abstract_method()

    def get_where(self, rg=None, rg_type=None):
        """ get the absolute position of flows records that within the range.

        Find all flows such that its belong to [rg[0], rg[1]). The interval
        is closed in the starting point and open in the ending pont.

        Parameters
        ------------------
        rg : list or tuple or None
            range of the the data. If rg == None, simply return position
            (0, row_num])
        rg_type : {'flow', 'time'}
            specify the type of the range.

        Returns
        -------------------
        sp, ep : ints
            flows with index such that sp <= idx < ed belongs to the range

        """
        abstract_method()

    def get_min_max(self, field_list):
        """  get the min and max value for fields

        Parameters
        ---------------
        field_list : a list of str

        Returns
        --------------
        miN, maX : a list of floats
            the mimium(maximium) value for each field in field_list
        """
        abstract_method()


import re

# @profile
def parse_records(f_name, FORMAT, regular_expression):
    flow = []
    with open(f_name, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            if line == '\n':  # Ignore Blank Line
                continue
            line = line.strip()
            item = re.split(regular_expression, line)
            f = tuple(h(item[pos]) for k, pos, h in FORMAT)
            flow.append(f)
    return flow

IP = lambda x:tuple(np.uint8(v) for v in x.rsplit('.'))

import pyximport; pyximport.install()
import numpy as np
from .CythonUtil import c_parse_records_tshark
# from CythonUtil import c_parse_records_fs

class PreloadHardDiskFile(Data):
    """ abstract base class for hard disk file The flow file into memory as a
    whole, so it cannot deal with flow file larger than your memery
    """

    RE = None
    """regular expression used to seperate each line into segments"""

    FORMAT = None
    """Format of the Data. Should be a list of tuple, each tuple has 3
    element (field_name, position, converter).
    """

    DT = None
    """ Specify how the data will be stored in Numpy array. Should be np.dtype
    See
    `Numpy.dtype
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html>`_
    for more information.
    """

    @property
    def FIELDS(self):
        return zip(*self.FORMAT)[0] if self.FORMAT is not None else None

    """  The name of all columns """

    def __init__(self, f_name):
        """ data_order can be flow_first | feature_first
        """
        self.f_name = f_name
        self._init()

    def parse(self):
        fea_vec = parse_records(self.f_name, self.FORMAT, self.RE)
        self.table = np.array(fea_vec, dtype=self.DT)
        self.row_num = self.table.shape[0]

    def _init(self):
        self.parse()

        self.t = np.array([t for t in self.get_rows('start_time')])
        t_idx = np.argsort(self.t)
        self.table = self.table[t_idx]
        self.t = self.t[t_idx]

        self.min_time = min(self.t)
        self.max_time = max(self.t)

    def shift_time(self, st):
        self.t += st

    def get_where(self, rg=None, rg_type=None):
        if not rg:
            return 0, self.row_num
        if rg_type == 'flow':
            sp, ep = rg
            if sp >= self.row_num:
                raise DataEndException()
        elif rg_type == 'time':
            sp = Find(self.t, rg[0] + self.min_time)
            ep = Find(self.t, rg[1] + self.min_time)
            assert(sp != -1 and ep != -1)
            if (sp == len(self.t) - 1 or ep == len(self.t) - 1):
                raise DataEndException()
        else:
            raise ValueError('unknow window type')
        return sp, ep

    def get_rows(self, fields=None, rg=None, rg_type=None, row_indices=None):
        if fields is None:
            fields = list(self.FIELDS)

        if row_indices is not None:
            return self.table[row_indices][fields]

        sp, ep = self.get_where(rg, rg_type)
        return self.table[sp:ep][fields]

    def get_min_max(self, feas):
        min_vec = []
        max_vec = []
        for fea in feas:
            dat = self.get_rows(fea)
            min_vec.append(min(dat))
            max_vec.append(max(dat))
        return min_vec, max_vec

import datetime
import time


def str_to_sec(ss, formats):
    """
    >>> str_to_sec('2012-06-17T16:26:18.300868', '%Y-%m-%dT%H:%M:%S.%f')
    14660778.300868
    """
    # x = time.strptime(ss,'%Y-%m-%dT%H:%M:%S.%f')
    x = time.strptime(ss,formats)

    ts = ss.rsplit('.')[1]
    micros = int(ts) if len(ts) == 6 else 0  # FIXME Add microseconds support for xflow
    return datetime.timedelta(
        days=x.tm_yday,
        hours=x.tm_hour,
        minutes=x.tm_min,
        seconds=x.tm_sec,
        microseconds=micros,
    ).total_seconds()


class HDF_FS(PreloadHardDiskFile):
    """  Data generated by `fs-simulator
    <http://cs-people.bu.edu/eriksson/papers/erikssonInfocom11Flow.pdf>`_ .
    HDF means Hard Disk File.
    """
    RE = '[\[\] :\->]'
    FORMAT = [
        ('start_time', 3, np.float64),
        ('end_time', 4, np.float64),
        ('src_ip', 5, IP),
        ('src_port', 6, np.int16),
        ('dst_ip', 8, IP),
        ('dst_port', 9, np.int16),
        ('prot', 10, np.str_),
        ('node', 12, np.str_),
        ('flow_size', 14, np.float64),
        ('duration', 13, np.float64),
    ]

    DT = np.dtype([
        ('start_time', np.float64, 1),
        ('end_time', np.float64, 1),
        ('src_ip', np.uint8, (4,)),
        ('src_port', np.int16, 1),
        ('dst_ip', np.int8, (4,)),
        ('dst_port', np.int16, 1),
        ('prot', np.str_, 5),
        ('node', np.str_, 5),
        ('flow_size', np.float64, 1),
        ('duration', np.float64, 1),
    ])

    # def parse(self):
    #     try: # try optimized parse method written in cython first
    #         self.table, self.row_num = c_parse_records_fs(self.f_name)
    #     except Exception as e:
    #         print('-' * 30)
    #         print(e)
    #         print('-' * 30)
    #         super(HDF_FS, self).parse()


class HDF_FlowExporter(PreloadHardDiskFile):
    """  Data generated FlowExporter. It is a simple tool to convert pcap to
    flow data. It is avaliable in tools folder.

    """
    RE = '[ \n]'
    FORMAT = [
        ('start_time', 0, np.float64),
        ('src_ip', 1, IP),
        ('dst_ip', 2, IP),
        ('prot', 3, np.str_),
        ('flow_size', 4, np.float64),
        ('duration', 5, np.float64),
    ]
    DT = np.dtype([
        ('start_time', np.float64, 1),
        ('src_ip', np.uint8, (4,)),
        ('dst_ip', np.uint8, (4,)),
        ('prot', np.str_, 5),
        ('flow_size', np.float64, 1),
        ('duration', np.float64, 1),
    ])


def seq_convert(args, arg_num, handlers):
    res = []
    i = 0
    for n, h in zip(arg_num, handlers):
        res.append(h(*args[i:(i + n)]))
        i += n
    return tuple(res)


class HDF_DumpFS(HDF_FS):
    def __init__(self, data):
        self.data = data
        super(HDF_DumpFS, self).__init__('')

    def shift_time(self, st):
        self.t += st

    def _init(self):
        data = self.data
        if isinstance(data, str):
            data = zload(data)
        self.table = data
        self.row_num = self.table.shape[0]

        self.t = np.array([t for t in self.get_rows('start_time')])
        t_idx = np.argsort(self.t)
        self.table = self.table[t_idx]
        self.t = self.t[t_idx]

        self.min_time = min(self.t)
        self.max_time = max(self.t)


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
        PreloadHardDiskFile.__init__(self, '')

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

    def _init(self):
        pass


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



ip_to_int = lambda x: np.uint64(x.replace(".", ""))
class HDF_tshark(PreloadHardDiskFile):
    """  Data generated FlowExporter. It is a simple tool to convert pcap to
    flow data. It is avaliable in tools folder.

    """
    RE = '\s+'
    FORMAT = [
        ('start_time', 1, np.float64),
        ('src_ip', 2, np.str_),
        ('dst_ip', 4, np.str_),
        ('prot', 5, np.str_),
        ('size', 6, np.float64),
    ]
    DT = np.dtype([
        ('start_time', np.float64),
        ('src_ip', np.str_, 15),
        ('dst_ip', np.str_, 15),
        ('prot', np.str_, 5),
        ('size', np.float64),
    ])

    def parse(self):
        self.table, self.row_num = c_parse_records_tshark(self.f_name)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
