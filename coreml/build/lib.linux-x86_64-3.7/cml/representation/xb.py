
import os, sys, ase
import numpy as np
from aqml.cheminfo.rw.xyz import *
import itertools as itl

import scipy.spatial.distance as ssd
from functools import reduce
import cml.fslatm as sl
#np.set_printoptions(precision=4)

T,F = True,False


"""
representing bond (for densitry matrix) in a molecule

A new implementation of SLATM is used, instead of using
the slatm module in qml (where only SLATM of atom is available)
"""

def get_nzs(ias1, ias2, zs, zsu):
    nzs = []
    nm = len(ias1)
    for i in range(nm):
        ib, ie = ias1[i], ias2[i]
        zsi = zs[ib:ie]
        nzsi = []
        for _zj in zsu:
            nzsi.append( (_zj == np.array(zsi,np.int)).sum() )
        nzs.append(nzsi)
    return np.array(nzs, np.int)

def get_neighbors(rbcut, coords):
    na = len(coords)
    ds = ssd.squareform( ssd.pdist( coords ) )
    ias = np.arange(na)
    _nbrs = []; ns = []
    nmax = 0
    for i in range(na):
        nbrs_i = ias[ds[i]<= rbcut]; nb = len(nbrs_i)
        if nb > nmax: nmax = nb
        _nbrs.append(nbrs_i); ns.append(nb)
    nbrs = - np.ones((na,nmax)).astype(np.int)
    for i in range(na):
        nbrs[i,:ns[i]] = _nbrs[i]
    return nmax, nbrs


class slatm(object):

    def __init__(self, nas, zs, coords, wd=None):
        """ slatm object """
        if wd is None:
            self.wd = os.environ['PWD']
        else:
            _wd = wd[:-1] if wd[-1] == '/' else wd
            self.wd = _wd
        self.nm = len(nas)
        self.zs = zs
        zsu = np.unique(self.zs).astype(np.int)
        self.zsu = zsu
        self.nas = nas
        ias2 = np.cumsum(self.nas)
        ias1 = np.concatenate( ([0], ias2[:-1]) )
        self.ias2 = ias2
        self.ias1 = ias1
        self.nat = sum(nas)
        self.nzs = get_nzs(ias1, ias2, zs, zsu)
        self.coords = np.array(coords)

    def get_slatm_mbtypes(self):
        """ get slatm many-body types"""
        zsmax = self.zsu
        nzsmax = np.max(self.nzs, axis=0)
        if self.pbc != '000':
            # the PBC will introduce new many-body terms, so set
            # nzmax to 3 if it's less than 3
            nzsmax[ nzsmax <= 2 ] = 3
        boas = [ [zi,] for zi in zsmax ]
        bops = [ [zi,zi] for zi in zsmax ] + list( itl.combinations(zsmax,2) )
        bots = []
        for i in zsmax:
            for bop in bops:
                j,k = bop
                tas = [ [i,j,k], [i,k,j], [j,i,k] ]
                for tasi in tas:
                    if (tasi not in bots) and (tasi[::-1] not in bots):
                        nzsi = [ (zj == tasi).sum() for zj in zsmax ]
                        if np.all(nzsi <= nzsmax):
                            bots.append( tasi )
        self.mbs1 = np.array(boas,np.int)
        self.mbs2 = np.array(bops,np.int)
        self.mbs3 = np.array(bots,np.int)


    def init_dm_pair(self):
        nbmax = 0
        nbrs = []; ns = []
        for i in range(self.nm):
            ib, ie = self.ias1[i], self.ias2[i]
            _nbmax, _nbrs = get_neighbors(self.rbcut, self.coords[ib:ie])
            assert _nbmax>0
            ns.append(_nbmax)
            nbmax = max(nbmax,_nbmax)
            nbrs += list(_nbrs)
        t = - np.ones((self.nat, nbmax)).astype(np.int)
        #print ' ** ias1, ias2 = ', self.ias1,self.ias2
        for i in range(self.nm):
            ib, ie = self.ias1[i], self.ias2[i]; #print ' ** ', nbrs[ib:ie]
            t[ib:ie, :ns[i]] = nbrs[ib:ie]
        self.nbmax = nbmax
        self.nbrs = t

    def init_grids(self):
        nsx = [0,0,0]; r0 = 0.1
        # set up grid
        nsx[0] = int((self.racut - r0)/self.dgrids[0]) + 1
        nsx[1] = int((self.rbcut - r0)/self.dgrids[0]) + 1
        #d2r = pi/dble(180) ! degree to rad
        #a0 = -20.0*d2r
        #a1 = pi + 20.0*d2r
        nsx[2] = int((np.pi + 40.0*(np.pi/180.))/self.dgrids[0]) + 1
        self.n = self.n1 + self.n2*nsx[0] + self.n3*nsx[2]
        self.nu = self.n1 + self.n2*nsx[1] + self.n3*nsx[2]
        self.nsx = nsx

    def get_x(self, rb=T, mbtypes=None, param={'racut':3.2, 'rbcut':4.2}):
        """ molecular repr

        rb -- represent bond?

        racut -- cutoff radius for atom
        rbcut -- cutoff radius for bond
        """
        # init_param first
        _param = {'local':T, 'nbody':3, 'dgrids': [0.04,0.04],  'sigmas':[0.05,0.05],\
                  'racut':3.2, 'rbcut':4.2, 'alchemy':False, 'iBoA':True, 'rpower2':6, 'coeffs':[1.], \
                  'rpower3': 3, 'ws':[1.,1./2,1./3], 'pbc':'000', 'kernel':'gaussian', \
                  'saves':[F,F,F], 'reuses':[F,F,F]}
        for key in list(param.keys()):
            if param[key] != _param[key]:
                _param[key] = param[key]
        self.savex,self.saved,self.savek = _param['saves']
        self.reusex,self.reused,self.reusek = _param['reuses']
        keys = ['dgrids','sigmas','racut','rbcut','rpower2','rpower3', 'ws','kernel','pbc']
        self.dgrids,self.sigmas,self.racut,self.rbcut,self.rpower2,self.rpower3,self.ws,self.kernel,self.pbc = \
                          [ _param[key] for key in keys ]
        if mbtypes is None:
            self.get_slatm_mbtypes()
            mbs1 = self.mbs1
            mbs2 = self.mbs2
            mbs3 = self.mbs3; #print ' ** mbs2, mbs3 = ', mbs2.shape, mbs3.shape
        else:
            mbs1, mbs2, mbs3 = mbtypes
        self.n1, self.n2, self.n3 = [len(mbsi) for mbsi in [mbs1, mbs2, mbs3]]
        self.mbtypes = [mbs1,mbs2,mbs3]

        if rb: # represent bonds (in Density matrix ML)
            self.init_dm_pair()
        else:
            self.nbmax = 1
            self.nbrs = - np.ones((self.nat, self.nbmax)).astype(np.int)

        self.init_grids()

        zs, nas = self.zs, self.nas
        n,nu,n1,n2,n3 = self.n,self.nu,self.n1,self.n2,self.n3
        nsx, dgrid, sigma, rpower2, rpower3 = self.nsx, self.dgrids[0], self.sigmas[0], self.rpower2, self.rpower3
        w1,w2,w3 = self.ws
        nbmax = self.nbmax
        rscut = [self.racut, self.rbcut]
        #print ' ++ nbmax = ', nbmax
        xsa = []; xsb = []; labels = []
        IA, IB = 0, -1
        for i in range(self.nm):
            ib, ie = self.ias1[i], self.ias2[i]
            _nbrs = self.nbrs[ib:ie]
            zsi = zs[ib:ie]
            _xs1,_xsb1 = sl.fslatm.fget_local_spectrum(zsi,self.coords[ib:ie].T, _nbrs.T, \
                           n,nu, mbs1,mbs2.T,mbs3.T, nsx, rscut,dgrid,sigma, \
                           w2,rpower2,w3,rpower3, IA, IB)
            xsa += list(_xs1)
            #print ' ** shape(_xs), shape(_xsb) = ', _xs.shape, _xsb.shape
            if rb:
                _xs = _xs1.T; _xsb = _xsb1.T
                for ia in range(nas[i]):
                    ## Note that the case [ A_I, A_I ] is automatically included below
                    li = list(_nbrs[ia])
                    for ja in _nbrs[ia]:
                        if ja > -1:
                            lj = list(_nbrs[ja])
                            label = [i,ia,ja,zsi[ia],zsi[ja]]; #print ' ** label = ', label
                            labels.append(label)
                            _ja= li.index(ja)
                            _ia= lj.index(ia)
                            xi = np.concatenate( (_xs[ia],_xs[ja], _xsb[ia,_ja,:],_xsb[ja,_ia,:]), axis=0 )
                            xsb.append(xi)
        self.xsb = xsb
        self.xsa = xsa
        self.labels = np.array(labels, np.int)


    def get_idx(self, keys, ims=None, labels=None, opt='ii'):
        """
        Call this function only when `rb = True (used __only__ for ML density-matrix elements

        opt = 'ii' or 'zz'
        """

        if labels is None:
            labels = self.labels
        nm = np.max(labels[:,0])
        nlb = len(labels)
        tidx = np.arange(nlb)
        if ims is None: ims = np.arange(nm)
        idxs = []
        if opt in ['ii']:
            for im in ims:
                for key in keys:
                    ia, ja =  key
                    entry = tidx[ reduce(np.logical_and, (labels[:,0]==im, labels[:,1]==ia, labels[:,2]==ja)) ]
                    if len(entry) == 0:
                        print('#ERROR: no entry found! R_IJ may exceed Rcut')
                        idxs.append(None)
                    else:
                        idxs.append( entry[0] )
        elif opt in ['zz','z']:
            same_atm = (labels[:,1]==labels[:,2]) if opt=='z' else (labels[:,1]!=labels[:,2])
            for im in ims:
                for key in keys:
                    zi, zj = key
                    entries = tidx[ reduce(np.logical_and, (same_atm, labels[:,0]==im, labels[:,3]==zi, labels[:,4]==zj)) ]
                    idxs += list(entries)
        else:
            raise '#ERROR:'
        return np.array(idxs,np.int)

if __name__ == '__main__':
    import ase.io as aio

    m = aio.read('c08h10.xyz')
    nas = [len(m)]; zs = m.numbers; coords = m.positions
    obj = slatm(nas, zs, coords)
    obj.get_x()


