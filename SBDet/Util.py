from __future__ import print_function, division, absolute_import
import sys


def abstract_method():
    """ This should be called when an abstract method is called that should
    have been implemented by a subclass. It should not be called in situations
    where no implementation (i.e. a 'pass' behavior) is acceptable. """
    raise NotImplementedError('Method not implemented!')


def binary_search(a, x, lo=0, hi=None):
    """
    Find the index of largest value in a that is smaller than x.
    a is sorted Binary Search
    """
    # import pdb;pdb.set_trace()
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        midval = a[mid]
        if midval < x:
            lo = mid + 1
        elif midval > x:
            hi = mid
        else:
            return mid
    return hi - 1

Find = binary_search


class DataEndException(Exception):
    pass

import numpy as np


try:
    import cPickle as pickle
except ImportError:
    import pickle


def dump(obj, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)


def load(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

import gzip
proto = pickle.HIGHEST_PROTOCOL


def zdump(obj, f_name):
    f = gzip.open(f_name, 'wb', proto)
    pickle.dump(obj, f)
    f.close()


def zload(f_name):
    f = gzip.open(f_name, 'rb', proto)
    obj = pickle.load(f)
    f.close()
    return obj


import matplotlib.pyplot as plt
import networkx as nx


def save_graph(graph, file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    del fig

import time

START_TIME = -1


def log(*args):
    if globals()['START_TIME'] == -1:
        globals()['START_TIME'] = time.time()
    msg = ' '.join([str(a) for a in args])
    print('[%f s]: --> %s' % (time.time() - globals()['START_TIME'], msg))


def adjust_pv_slow(prob, eps):
    """ adjust probability vector so that each value >= eps

    Parameters
    ---------------
    prob : list or tuple
        probability vector
    eps : float
        threshold

    Returns
    --------------
    prob : list
        adjusted probability vector

    Examples
    -------------------
    >>> adjust_pv([0, 0, 1], 0.01)
    [0.01, 0.01, 0.98]

    """
    assert(abs(sum(prob) - 1) < 1e-3)
    a = len(prob)
    # zero element indices
    zei = [i for i, v in zip(xrange(a), prob) if abs(v) < eps]
    if a == len(zei):  # all elements are zero
        return [eps] * a
    zei_sum = sum(prob[i] for i in zei)
    adjustment = (eps * len(zei) - zei_sum) * 1.0 / (a - len(zei))
    prob2 = [v - adjustment for v in prob]
    for idx in zei:
        prob2[idx] = eps
    if min(prob2) < 0:
        print('[warning] EPS is too large in adjust_pv')
        # return adjust_pv(prob2, eps / 2.0)
    return prob2


def adjust_pv(prob, eps):
    """ adjust_pv using numpy. It will change input parameter **prob**

    See Also
    --------------
    adjust_pv_slow

    """
    prob[prob == 0] = eps
    prob /= np.sum(prob)
    return prob


def KL_div(nu, mu, eps=None):
    """  Calculate the empirical measure of two probability vector nu and mu

    Parameters
    ---------------
    nu, mu : list or tuple
        two probability vector

    Returns
    --------------
    res : float
        the cross entropy

    Notes
    -------------
    The cross-entropy of probability vector **nu** with respect to **mu** is
    defined as

    .. math::

        H(nu|mu) = \sum_i  nu(i) \log(nu(i)/mu(i)))

    One problem that needs to be addressed is that mu may contains 0 element.

    Examples
    --------------
    >>> from numpy import array
    >>> print KL_div(array([0.3, 0.7, 0, 0]), array([0, 0, 0.3, 0.7]))
    45.4408375578

    """
    assert(len(nu) == len(mu))
    # a = len(nu)

    if eps:
        mu = adjust_pv(mu, eps)
        nu = adjust_pv(nu, eps)

    # H = lambda x, y: x * log(x * 1.0 / y)
    # return sum(H(a, b) for a, b in zip(nu, mu))
    return np.dot(nu, np.log(nu / mu))


def progress_bar(i):
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()


# def np_index(A, B):
#     """  index function of np.array

#     Parameters
#     ---------------
#     A : np.2darray
#         set of elements
#     B : np.array
#         one element of the form of numpy array
#     Returns
#     --------------
#     C : np.array
#         the row number of A which equals to B.
#     """
#     nrows, ncols = A.shape
#     dtype = {
#         'names': ['f{}'.format(i) for i in range(ncols)],
#         'formats': ncols * [A.dtype]
#     }
#     C = np.where(A.view(dtype) == B.view(dtype))
#     return C[0]


def np_to_dotted(ip):
    return '{0}.{1}.{2}.{3}'.format(*ip)
    # ip_s = str(ip)
    # return '.'.join(ip_s.strip('[] ').rsplit())

try:
    import igraph
except ImportError:
    igraph = False


try:
    import scipy as sp
except ImportError:
    sp = False

try:
    import scipy.linalg as la
except ImportError:
    la = False

try:
    import scipy.stats as stats
except ImportError:
    stats = False


lf = [
    0.000000000000000,
    0.000000000000000,
    0.693147180559945,
    1.791759469228055,
    3.178053830347946,
    4.787491742782046,
    6.579251212010101,
    8.525161361065415,
    10.604602902745251,
    12.801827480081469,
    15.104412573075516,
    17.502307845873887,
    19.987214495661885,
    22.552163853123421,
    25.191221182738683,
    27.899271383840894,
    30.671860106080675,
    33.505073450136891,
    36.395445208033053,
    39.339884187199495,
    42.335616460753485,
    45.380138898476908,
    48.471181351835227,
    51.606675567764377,
    54.784729398112319,
    58.003605222980518,
    61.261701761002001,
    64.557538627006323,
    67.889743137181526,
    71.257038967168000,
    74.658236348830158,
    78.092223553315307,
    81.557959456115029,
    85.054467017581516,
    88.580827542197682,
    92.136175603687079,
    95.719694542143202,
    99.330612454787428,
    102.968198614513810,
    106.631760260643450,
    110.320639714757390,
    114.034211781461690,
    117.771881399745060,
    121.533081515438640,
    125.317271149356880,
    129.123933639127240,
    132.952575035616290,
    136.802722637326350,
    140.673923648234250,
    144.565743946344900,
    148.477766951773020,
    152.409592584497350,
    156.360836303078800,
    160.331128216630930,
    164.320112263195170,
    168.327445448427650,
    172.352797139162820,
    176.395848406997370,
    180.456291417543780,
    184.533828861449510,
    188.628173423671600,
    192.739047287844900,
    196.866181672889980,
    201.009316399281570,
    205.168199482641200,
    209.342586752536820,
    213.532241494563270,
    217.736934113954250,
    221.956441819130360,
    226.190548323727570,
    230.439043565776930,
    234.701723442818260,
    238.978389561834350,
    243.268849002982730,
    247.572914096186910,
    251.890402209723190,
    256.221135550009480,
    260.564940971863220,
    264.921649798552780,
    269.291097651019810,
    273.673124285693690,
    278.067573440366120,
    282.474292687630400,
    286.893133295426990,
    291.323950094270290,
    295.766601350760600,
    300.220948647014100,
    304.686856765668720,
    309.164193580146900,
    313.652829949878990,
    318.152639620209300,
    322.663499126726210,
    327.185287703775200,
    331.717887196928470,
    336.261181979198450,
    340.815058870798960,
    345.379407062266860,
    349.954118040770250,
    354.539085519440790,
    359.134205369575340,
    363.739375555563470,
    368.354496072404690,
    372.979468885689020,
    377.614197873918670,
    382.258588773060010,
    386.912549123217560,
    391.575988217329610,
    396.248817051791490,
    400.930948278915760,
    405.622296161144900,
    410.322776526937280,
    415.032306728249580,
    419.750805599544780,
    424.478193418257090,
    429.214391866651570,
    433.959323995014870,
    438.712914186121170,
    443.475088120918940,
    448.245772745384610,
    453.024896238496130,
    457.812387981278110,
    462.608178526874890,
    467.412199571608080,
    472.224383926980520,
    477.044665492585580,
    481.872979229887900,
    486.709261136839360,
    491.553448223298010,
    496.405478487217580,
    501.265290891579240,
    506.132825342034830,
    511.008022665236070,
    515.890824587822520,
    520.781173716044240,
    525.679013515995050,
    530.584288294433580,
    535.496943180169520,
    540.416924105997740,
    545.344177791154950,
    550.278651724285620,
    555.220294146894960,
    560.169054037273100,
    565.124881094874350,
    570.087725725134190,
    575.057539024710200,
    580.034272767130800,
    585.017879388839220,
    590.008311975617860,
    595.005524249382010,
    600.009470555327430,
    605.020105849423770,
    610.037385686238740,
    615.061266207084940,
    620.091704128477430,
    625.128656730891070,
    630.172081847810200,
    635.221937855059760,
    640.278183660408100,
    645.340778693435030,
    650.409682895655240,
    655.484856710889060,
    660.566261075873510,
    665.653857411105950,
    670.747607611912710,
    675.847474039736880,
    680.953419513637530,
    686.065407301994010,
    691.183401114410800,
    696.307365093814040,
    701.437263808737160,
    706.573062245787470,
    711.714725802289990,
    716.862220279103440,
    722.015511873601330,
    727.174567172815840,
    732.339353146739310,
    737.509837141777440,
    742.685986874351220,
    747.867770424643370,
    753.055156230484160,
    758.248113081374300,
    763.446610112640200,
    768.650616799717000,
    773.860102952558460,
    779.075038710167410,
    784.295394535245690,
    789.521141208958970,
    794.752249825813460,
    799.988691788643450,
    805.230438803703120,
    810.477462875863580,
    815.729736303910160,
    820.987231675937890,
    826.249921864842800,
    831.517780023906310,
    836.790779582469900,
    842.068894241700490,
    847.352097970438420,
    852.640365001133090,
    857.933669825857460,
    863.231987192405430,
    868.535292100464630,
    873.843559797865740,
    879.156765776907600,
    884.474885770751830,
    889.797895749890240,
    895.125771918679900,
    900.458490711945270,
    905.796028791646340,
    911.138363043611210,
    916.485470574328820,
    921.837328707804890,
    927.193914982476710,
    932.555207148186240,
    937.921183163208070,
    943.291821191335660,
    948.667099599019820,
    954.046996952560450,
    959.431492015349480,
    964.820563745165940,
    970.214191291518320,
    975.612353993036210,
    981.015031374908400,
    986.422203146368590,
    991.833849198223450,
    997.249949600427840,
    1002.670484599700300,
    1008.095434617181700,
    1013.524780246136200,
    1018.958502249690200,
    1024.396581558613400,
    1029.838999269135500,
    1035.285736640801600,
    1040.736775094367400,
    1046.192096209724900,
    1051.651681723869200,
    1057.115513528895000,
    1062.583573670030100,
    1068.055844343701400,
    1073.532307895632800,
    1079.012946818975000,
    1084.497743752465600,
    1089.986681478622400,
    1095.479742921962700,
    1100.976911147256000,
    1106.478169357800900,
    1111.983500893733000,
    1117.492889230361000,
    1123.006317976526100,
    1128.523770872990800,
    1134.045231790853000,
    1139.570684729984800,
    1145.100113817496100,
    1150.633503306223700,
    1156.170837573242400,
]

# import math
# def log_fact(n):
#     if n < 0:
#         raise Exception()
#     elif n > 254:
#         x = n + 1
#         return (x - 0.5)* math.log(x) - x + \
                # 0.5*math.log(2*math.pi) + 1.0/(12.0*x);
#     else:
#         return lf[n];


def log_fact_mat(n):
    nm = np.max(n)
    if nm > 254:
        nv = n[n > 254]
        n[n > 254] = (nv - 0.5)*np.log(nv) - nv + \
            0.5*np.log(2*np.pi) + 1.0/(12.0*nv)
        nm = 254

    for i in xrange(int(nm)+1):
        n[n == i] = lf[i]
    return n


def warning(msg):
    print("warning: ", msg)


""" Utility Functions to Calculate Statistics
"""


def roc(trace):
    """  Calculate points of ROC curve given trace data

    Parameters
    ---------------
    trace : tuple of 6 lists
        each list are observation of
            tuple positive num
            false negative num
            true negative num
            false positive num
    Returns
    --------------
    fpr : list of float
        false positive rate
    tpr : list of float
        true positive rate
    """
    tpv, fnv, tnv, fpv, _, _ = trace
    tpr = [tp * 1.0 / (tp + fn) for tp, fn in zip(tpv, fnv)]
    # calculate the false positive rate
    fpr = [fp * 1.0 / (fp + tn) for fp, tn in zip(fpv, tnv)]
    print('fpr, ', fpr)
    print('tpr, ', tpr)
    return fpr, tpr


def get_quantitative(A, B, W, show=True):
    """
    """
    """

    Parameters
    ---------------
    A, B, W : set
        **A** is the referece, and **B** is the detected result, **W** is the
        whole set calculate the true positive, false negative, true negative
        and false positive
    show : bool
        when to print the results out.

    Returns
    --------------
    ret : 6-element tuple
        - true positive num,
        - false negative num,
        - true negative num,
        - false positive num,
        - false positive rate
        - true positive rante
    """
    A = set(A)
    B = set(B)
    W = set(W)
    # no of true positive, no of elements belongs to B and also belongs to A
    tp = len(set.intersection(A, B))

    # no of false negative no of elements belongs to A but doesn't belong to B
    fn = len(A - B)

    # no of true negative, no of element not belongs to A and not belong to B
    tn = len(W - set.union(A, B))
    # no of false positive. no of ele. not belongs to A but belongs to B
    fp = len(B - A)

    # sensitivity is the probability of a alarm given that the this flow is
    # anormalous
    # sensitivity = tp * 1.0 / (tp + fn)
    # specificity is the probability of there isn't alarm given that the flow
    # is normal
    # specificity = tn * 1.0 / (tn + fp)

    tpr = tp * 1.0 / (tp + fn)
    fpr = fp * 1.0 / (fp + tn)

    # ret = tp, fn, tn, fp, sensitivity, specificity
    ret = tp, fn, tn, fp, fpr, tpr
    if show:
        OUT_STRING = """tp: %f\t fn: %f\t tn: %f\t fp: %f
        fpr: %f\ttpr: %f
        """
        print(OUT_STRING % ret)
    return ret


def degree(G):
    return np.array(G.sum(axis=0), dtype=int).ravel()
