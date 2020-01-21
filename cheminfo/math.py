
import numpy as np


def get_compl(a,b):
    a_compl = []
    for ai in a:
        if ai not in b:
            a_compl.append(ai)
    return a_compl


def get_compl_u(a,b):
    a_compl = []
    bu = [ set(bi) for bi in b ]
    for ai in a:
        if set(ai) not in bu:
            a_compl.append(ai)
    return a_compl

def products(s, idxsOnly=False):
    """
    input `s is a list of sublists and each sublist is
    of different size, we want to enumerate the product
    of elements from each sublist.
    e.g., s = [ ['a','b','c'], ['d','e'], ['f','g','h','i'] ]
    --> [ ['a','d','f'], ['b','d','f'], ['c','d','f',]
          ['a','e','f',], ['b','e','f'], ['c','e','f'],
          ...
          ]
    The problems could be simplified to obtaining the indices
    first; then get the combinations
    ns = [3,2,4] --> combs = [(0,0,0),
                              (1,0,0),
                              (2,0,0),
                              (0,1,0),
                              (1,1,0),
                              (2,1,0),
                              ...
                              (0,1,3),
                              (1,1,3),
                              (2,1,3)
    """

    ns = [ len(si) for si in s]; N = len(ns)
    su = []; idxsu = []
    nt = np.int( np.product(ns) )
    for i in range(nt):
        si = []; idxsi = []; iu = i
        for j in range(N):
            if j == N - 1:
                dnm_1 = ns[j]; idx = iu%dnm_1
                si.append( s[j][idx] )
                idxsi.append(idx)
#                print iu, dnm_2, si
            else:
                dnm_2 = np.product(ns[j+1:]); idx = iu//dnm_2
                idxsi.append(idx)
                si.append( s[j][idx] )
                i0 = iu; iu = iu%dnm_2
#                print i0, iu, dnm_2, si
        su.append(si); idxsu.append(idxsi)

    ots = su
    if idxsOnly:
        idxsu.sort()
        ots = idxsu

    return ots


def merge_sets(sets):
    """
    merge any two sets sharing some elements
    """
    idxs_skipped = []
    n = len(sets)
    for i in range(n-1):
        if i not in idxs_skipped:
            set_i = sets[i]
            for j in range(i+1,n):
                set_j = sets[j]
                if set_i.intersection( set_j ) > set([]):
                    sets[i].update( set_j )
                    idxs_skipped.append( j )
    sets_u = [ sets[k] for k in np.setdiff1d(range(n), idxs_skipped).astype(np.int) ]
    return sets_u

def union(sets):
    s = set([])
    for set_i in sets: s.update( set_i )
    return s

def get_idx_set(i, sets):
    """
    find out the idx of set where `i is an element
    """
    idxs = []
    for j, set_j in enumerate(sets):
        if i in set_j: idxs.append(j)
    return idxs


