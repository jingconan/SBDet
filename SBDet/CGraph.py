#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

from .Util import igraph
from .Util import abstract_method
from .Util import np_to_dotted
# from .Data import HDF_FlowExporter

# def pcap2csv(pcap_file_name, csv_name):
#     txt_f_name = pcap_file_name.rsplit('.pcap')[0] + '_tshark.txt'
#     export_to_txt(pcap_file_name, txt_f_name)
#     records, name = parse_txt(txt_f_name)
#     write_csv(records, csv_name)


def np_union2d(A, B):
    """  union of A, B. Each row is consider as a single element
    A and B must be array whose buffer is continuous.
    """
    # http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    nrows, ncols = A.shape
    dtype = {
        'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [A.dtype]
    }
    C = np.union1d(A.view(dtype), B.view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C





class TrafficGraph(object):
    """ Traffic Dispersion Graphs constructed from flow data.

    Parameters
    ---------------
    f_name : str
        name of the flow file
    node_info : dict
        additional data structure to store information related to nodes.
        two fields: 1 node_name, 2 node_ip.
    """

    def __init__(self, data=None, node_info=None, graph=None):
        self.data = data
        self.node_info = node_info


        # map ip to name
        if node_info is not None:
            self.ip_to_name = dict()
            for name, ip_table in zip(node_info['node_name'], node_info['node_ip']):
                for ip in ip_table:
                    self.ip_to_name[ip.rsplit('/')[0]] = name
        if graph is not None:
            self.graph = graph
        else:
            self._init()

    def _init(self):
        abstract_method()

    def add_vertices(self, ips):
        """  add vertices to the graph

        Parameters
        ---------------
        ips : list of dot-sparated string
            ip addresses for vertices

        Returns
        --------------
        None
        """
        abstract_method()

    def add_edges(self, edges):
        """  add edges to the graph

        Parameters
        ---------------
        edges :  a list of tuple.
            each tuple is (src_id, dst_id)

        Returns
        --------------
        None
        """
        abstract_method()

    def filter(self, prot='TCP', rg=None, rg_type=None):  # FIXME change it to more general later
        if prot is None:
            return self.data.get_rows(rg=rg, rg_type=rg_type)
        pkt_prot = self.data.get_rows(fields='prot', rg=rg, rg_type=rg_type)
        filter_indices = np.nonzero(pkt_prot == prot)[0]
        records = self.data.get_rows(row_indices=filter_indices)
        return records

    def get_vertices(self):
        """  calculate all unique ip address in the flow data. Treat each as a
        vertex.

        Returns
        --------------
        ips : np.2darray
            a list of unique ip addresses.
        """
        src_ip = self.data.get_rows(fields='src_ip')
        src_ip = np.array(src_ip, copy=True)  # make buffer continuous
        dst_ip = self.data.get_rows(fields='dst_ip')
        dst_ip = np.array(dst_ip, copy=True)  # make buffer continous
        # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        if isinstance(src_ip[0], str):
            self.ips = set(src_ip) | set(dst_ip)
        else:
            self.ips = np_union2d(src_ip, dst_ip)
        return self.ips

    # @profile
    def get_edges(self, records):
        """  get edge from flow records. There will be link from (i, j) if there is
        record from node i to node j.

        Parameters
        ---------------
        records : a list of **record**
            format: to be finished.

        Returns
        --------------
        edges : a list of tuple
            each tuple is (src_id, dst_id)
        """
        tran_sd = zip(records['src_ip'], records['dst_ip'])
        # get all edges
        # import ipdb;ipdb.set_trace()
        # tran_sd = []
        # for rec in records:
            # tran_sd.append((np_to_dotted(rec[1]), np_to_dotted(rec[2])))
            # tran_sd.append((rec[1], rec[2]))
        edges = Counter(tran_sd)
        self.edges = edges
        return edges


class Igraph(TrafficGraph):
    """  Traffic Graph that works above python-igraph bindings.

    See `TrafficGraph`
    """
    def _init(self):
        # self.data = HDF_FlowExporter(self.f_name)
        self.adj_mat = None
        self.graph = igraph.Graph(directed=False)
        # self.graph = igraph.Graph(directed=True)

    def add_vertices(self, ips):
        self.graph.add_vertices(ips)
        # for ip in ips:
            # self.graph.add_vertex(name=np_to_dotted(ip))

    def add_edges(self, edges):
        if len(edges) == 0:
            return
        self.i_edges, self.weights = zip(*edges.items())
        self.i_edges = list(self.i_edges)
        self.weights = list(self.weights)
        self.graph.add_edges(self.i_edges)

    def gen_layout(self):
        return self.graph.layout("circular")
        # return self.graph.layout("kk")

    def plot(self, *args, **kwargs):
        # vertex_label = [self.ip_to_name[ip] for ip in self.graph.vs['name']]
        vertex_label = self.graph.vs['name']
        # import ipdb;ipdb.set_trace()
        # vertex_label = [str(i) for i in xrange(len(self.ips))]
        igraph.plot(self.graph, *args,
                    # layout=layout,
                    vertex_label=vertex_label,
                    **kwargs)
                    # edge_arrow_size=2, edge_arrow_width=2,
                    # vertex_label= [str(x) for x in range(len(self.ips))],


class NetworkXGraph(TrafficGraph):
    """  Traffic Graph that works above networkx.

    See `TrafficGraph`
    """

    def _init(self):
        # self.data = HDF_FlowExporter(self.f_name)
        self.adj_mat = None
        # self.graph = nx.DiGraph()
        self.graph = nx.Graph()

    def add_vertices(self, ips):
        self.graph.add_nodes_from(ips)
        # for ip in ips:
            # self.graph.add_node(np_to_dotted(ip))

    def add_edges(self, edges):
        for (src, dst), weight in edges.iteritems():
            self.graph.add_edge(src, dst, weight=weight)

    def gen_layout(self):
        return nx.graphviz_layout(self.graph)

    def plot(self, f_name, layout, *args, **kwargs):
        plt.clf()
        nx.draw(self.graph, pos=layout, arrows=True)
        plt.savefig(f_name)
        # plt.show()
