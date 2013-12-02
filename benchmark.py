#!/usr/bin/env python
from __future__ import print_function, division
import sys
from SBDet.Data import HDF_FS, HDF_tshark


def main():
    for i in xrange(10):
        print('i', i)
        # HDF_FS('./n0_flow.txt')
        HDF_tshark('../ddostrace.20070804_135436_tshark.txt')
    print('done')
    pass



if __name__ == "__main__":
    main()
