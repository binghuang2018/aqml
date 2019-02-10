
#import cheminfo.fortran.famon as fm
import networkx as nx
import numpy as np
import multiprocessing
import itertools as itl
import scipy.spatial.distance as ssd

T, F = True, False

def get_pl(ipt):
    """ get shortest path length between atom i and j """
    gnx, i, j = ipt
    pl = 0
    if nx.has_path(gnx,i,j):
        pl = nx.shortest_path_length(gnx,i,j)
    return pl


class Graph(object):

    def __init__(self, g, nprocs=1):
        n = g.shape[0] # number_of_nodes
        self.n = n
        g1 = (g > 0).astype(np.int)
        np.fill_diagonal(g1, 0)
        self.g  = g1
        self.nprocs = nprocs

    @property
    def nb(self):
        if not hasattr(self, '_nb'):
            self._nb = g1.sum()/2 # num of bonds (i.e., edges)
        return self._nb

    @property
    def gnx(self):
        if not hasattr(self, '_gnx'):
            self._gnx = nx.from_numpy_matrix(self.g)
        return self._gnx

    @property
    def bonds(self):
        if not hasattr(self, '_bonds'):
            self._bonds = [ list(edge) for edge in \
                  np.array( list( np.where(np.triu(self.g)>0) ) ).T ]
        return self._bonds

    def get_number_of_rings(self):
        g = self.g
        gu = (g > 0); v = gu.shape[0];
        # in some case, u might have graph with weighted nodes
        # the safe way is  to first assign diagonal elements to 0
        for i in range(v): gu[i,i] = 0
        e = gu.ravel().sum()/2
        r = 2**(e - v + 1) - 1
        return r

    @property
    def is_connected(self):
        if not hasattr(self, '_iconn'):
            self._iconn = self.is_connected_graph()
        return self._iconn

    def is_connected_graph(self):
        gobj = nx.Graph(self.g.astype(np.int))
        return nx.is_connected(gobj)

    @property
    def has_standalone_atom(self):
        """ is there any standalone atom in this mol? """
        if not hasattr(self, '_ialone'):
            tf = F
            for c in self.find_cliques():
                if len(c) % 2 == 1:
                    tf = T
                    break
            self._ialone = tf
        return self._ialone

    @property
    def cliques(self):
        if not hasattr(self, '_clq'):
            self._clq = self.find_cliques()
        return self._clq

    def find_cliques(self):
        """
        the defintion of `clique here is not the same
        as that in graph theory, which states that
        ``a clique is a subset of vertices of an
        undirected graph such that every two distinct
        vertices in the clique are adjacent; that is,
        its induced subgraph is complete.''
        However, in our case, it's simply a connected
        subgraph, or a fragment of molecule. This is useful
        only for identifying the conjugated subset of
        atoms connected all by double bonds or bonds of
        pattern `double-single-double-single...`
        """
        g1 = self.g
        n = self.n
        cliques = []
        G = nx.Graph(g1)
        if nx.is_connected(G):
            cliques = [ list(range(n)), ]
        else:
            sub_graphs = nx.connected_component_subgraphs(G)
            for i, sg in enumerate(sub_graphs):
                cliques.append( list(sg.nodes()) )
        #nc, iass = fm.connected_components(g1)
        #for i in range(nc):
        #    ias = iass[:,i] - 1
        #    cliques.append( ias[ias >= 0] )
        return cliques


    @property
    def is_connected(self):
        if not hasattr(self, '_ic'):
            self._ic = nx.is_connected(self.gnx)
        return self._ic

    def get_pls(self):
        """
        get shortest path length between all atoms
        """
        gx = self.gnx
        ias = np.arange(self.n)
        ipts = []
        for i,j in itl.combinations(ias,2):
            ipts.append([gx,i,j])
        _ts = []
        #print('g=', self.g)
        if self.nprocs > 1:
            print(' now parallelly computing path lengths...')
            pool = multiprocessing.Pool(processes=self.nprocs)
            _ts = pool.map(get_pl, ipts) #
            print(' done!')
        else:
            for ipt in ipts:
                _ts.append(get_pl(ipt))
        pls = ssd.squareform(_ts)
        return pls

    @property
    def pls(self):
        if not hasattr(self, '_pls'):
            self._pls = self.get_pls()
        return self._pls

    def get_shortest_path(self, i, j):
        return list( nx.shortest_path(self.gnx, i, j) )

    def get_shortest_paths(self, i, j):
        """ return shortest paths connecting two nodes i & j """
        paths = []
        for p in nx.all_shortest_paths(self.gnx, source=i, target=j):
            paths.append( list(p) )
        return paths


