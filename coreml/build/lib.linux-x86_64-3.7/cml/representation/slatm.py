
from __future__ import print_function

import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np

from .fslatm import calc_sbot
from .fslatm import calc_sbot_local
from .fslatm import calc_sbop
from .fslatm import calc_sbop_local
import ase

T, F = True, False


def get_boa(z1, zs_):
    return z1*np.array( [(zs_ == z1).sum(), ])
    #return -0.5*z1**2.4*np.array( [(zs_ == z1).sum(), ])


def get_sbop(mbtype, obj, cg=None, izeff=F, iloc=False, ia=None, normalize=True, sigma=0.05, \
             rcut=4.8, dgrid=0.03, pbc=F, rpower=6):
    """
    two-body terms

    :param obj: molecule object, consisting of two parts: [ zs, coords ]
    :type obj: list
    """

    z1, z2 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    #print('    bop:  ia=', ia)
    zs1 = zs.copy()
    coords1 = coords.copy()
    if pbc:
        #if rcut < 9.0: raise '#ERROR: rcut too small for systems with pbc'
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        mobj = MolPBC(zs, coords, c, rcut=rcut)
        zs1, coords1 = mobj.get_cluster(ia)
        #print('zs=',zs1, 'coords=',coords1) # ase.Atoms(zs1,coords1) )
        #print('zs=',zs1, '; coords=', [list(csi) for csi in coords1], '\nm=ase.Atoms(zs,coords)') # ase.Atoms(zs1,coor
        #if cg is None: raise Exception('Todo: connectivity between atoms in solid <- voronoi diagram')
        # after update of `m, the query atom `ia will become the first atom
        #ia = 0
    #print('          ia=', ia)

    na = len(zs1)
    _cg = np.ones((na,na), dtype=int)
    #if (cg is not None) and (ia is not None):
    #    #assert isinstance(cg,dict), '#ERROR: input `cg is not a dict?'
    #    _cg = np.zeros((na,na), dtype=int)
    #    ias_sl = cg[ia]
    #    _cg[np.ix_(ias_sl,ias_sl)] = 1

    # bop potential distribution
    r0 = 0.1
    nx = int((rcut - r0)/dgrid) + 1


    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

    if iloc:
        iatm = 0 if pbc else ia
        ys = calc_sbop_local(coords1, zs1, iatm, _cg, izeff, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)
    else:
        ys = calc_sbop(coords1, zs1, _cg, izeff, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)

    return ys

def get_sbot(mbtype, obj, cg=None, izeff=F, iloc=F, ia=None, normalize=T, sigma=0.05, \
             rcut=4.8, dgrid=0.0262, pbc=F):

    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """

    z1, z2, z3 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    #print('    bot:  ia=', ia)

    zs1 = zs.copy()
    coords1 = coords.copy()
    if pbc:
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        mobj = MolPBC(zs, coords, c, rcut=rcut)
        zs1, coords1 = mobj.get_cluster(ia)
        #print('zs=',zs1, '; coords=', [list(csi) for csi in coords1], '\nm=ase.Atoms(zs,coords)') # ase.Atoms(zs1,coords1) )
        #print('atoms=', ase.Atoms(zs1,coords1) )
        #if cg is None: raise Exception('Todo: connectivity between atoms in solid <- voronoi diagram')
        # after update of `m, the query atom `ia will become the first atom
        #ia = 0
    #print('          ia=', ia)

    na = len(zs1)
    _cg = np.ones((na,na), dtype=int)
    #if (cg is not None) and (ia is not None):
    #    #assert isinstance(cg,dict), '#ERROR: input `cg is not a dict?'
    #    _cg = np.zeros((na,na), dtype=int)
    #    ias_sl = cg[ia]
    #    _cg[np.ix_(ias_sl,ias_sl)] = 1


    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

    # Setup grid in Python
    d2r = np.pi/180 # degree to rad
    a0 = -20.0*d2r
    a1 = np.pi + 20.0*d2r
    nx = int((a1-a0)/dgrid) + 1

    if iloc:
        iatm = 0 if pbc else ia
        ys = calc_sbot_local(coords1, zs1, iatm, _cg, izeff, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)
    else:
        ys = calc_sbot(coords1, zs1, _cg, izeff, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)

    return ys



class MolPBC(object):

    def __init__(self, zs, coords, cell, rcut=9.0):
        """
         a mol obj with pbc
        """
        na = len(zs)
        self.na = na
        self.zs = zs
        self.coords = coords
        self.cell = np.array(cell)
        self.coords_f = np.linalg.solve(self.cell.T, coords.T).T # scaled (fractional) coords
        ls = np.linalg.norm(cell, axis=0)
        self.ls = ls
        self.a = ls[0]
        self.b = ls[1]
        self.c = ls[2]
        self.ias = np.arange(na)
        self.rcut = rcut


    def get_ext_cell(self, idx):
        """
         get a new extended cell with index `idx
        """
        i, j, k = idx
        if i==j==k==0:
            return self.zs, self.coords
        zs = self.zs
        coords = self.coords + np.dot(idx, self.cell)
        return zs, coords


    def get_nmax(self, ia, axis, sign):
        """
          Get maximal num of cells to be repeat along `axis and direction `sign

          axis: 0, 1, 2 <--> x, y, z
          sign: +1 (position x/y/z direction), -1 (negative x/y/z direction)
        """
        n = sign
        while T:
            if self.ls[axis] * np.abs(n - self.coords_f[ia][axis]) > self.rcut:
                n += sign
                break
            else:
                n += sign
        return n


    def get_cluster(self, ia):
        """
         get all neighbors within a cutoff radius of `rcut
        """

        ns = []
        for axis in [0, 1, 2]:
            for sign in [-1, 1]:
                ns.append( self.get_nmax(ia, axis, sign) )
        #print('ns=',ns)
        nx1, nx2, ny1, ny2, nz1, nz2 = ns
        n1s = np.arange(nx1,nx2+1)
        n2s = np.arange(ny1,ny2+1)
        n3s = np.arange(nz1,nz2+1)

        zs = [ self.zs[ia] ]
        coords = [ self.coords[ia] ]
        for idx in itl.product(n1s,n2s,n3s):
            _zs, _coords = self.get_ext_cell(idx)
            dsi = ssd.cdist([self.coords[ia]], _coords)[0]
            for ja in self.ias[dsi <= self.rcut]:
                if (ja == ia) and (idx==(0,0,0)):
                    continue
                zs += [ _zs[ja] ]
                coords += [ _coords[ja] ]
        assert np.all( ssd.pdist(coords) > 0 )
        return [zs, coords]



def update_m(obj, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom `ia
    for periodic systems (or very large system)
    """
    zs, coords, c = obj
    v1, v2, v3 = c
    vs = ssd.norm(c, axis=0)

    ds = ssd.squareform( ssd.pdist(coords) )

    nns = []
    for i,vi in enumerate(vs):
        n1_doulbe = rcut/vi
        n1 = int(n1_doulbe)
        if n1 - n1_doulbe == 0:
            n1s = range(-n1, n1+1) if pbc[i] else [0,]
        elif n1 == 0:
            n1s = [-1,0,1] if pbc[i] else [0,]
        else:
            n1s = range(-n1-1, n1+2) if pbc[i] else [0,]

        nns.append(n1s)

    n1s,n2s,n3s = nns

    n123s_ = np.array( list( itl.product(n1s,n2s,n3s) ) )
    n123s = []
    for n123 in n123s_:
        n123u = list(n123)
        if n123u != [0,0,0]: n123s.append(n123u)

    nau = len(n123s)
    n123s = np.array(n123s, np.float)

    na = len(zs)
    cia = coords[ia]

    zs_u = []; coords_u = []
    zs_u.append( zs[ia] ); coords_u.append( coords[ia] )
    for i in range(na) :
        di = ds[i,ia]
        if (di > 0) and (di <= rcut):
            zs_u.append(zs[i]); coords_u.append(coords[i])

# add new coords by translation
            ts = np.zeros((nau,3))
            for iau in range(nau):
                ts[iau] = np.dot(n123s[iau],c)

            coords_iu = coords[i] + ts #np.dot(n123s, c)
            dsi = ssd.norm(coords_iu - cia, axis=1);
            filt = np.logical_and(dsi > 0, dsi <= rcut); nx = filt.sum()
            zs_u += [zs[i],]*nx
            coords_u += [ list(ci) for ci in coords_iu[filt,:] ]

    #for ci in coords_u: print(ci)

    obj_u = [np.array(zs_u,dtype=int), np.array(coords_u)]
    assert np.all( ssd.pdist(coords_u) > 0 )

    return obj_u


class NBody(object):

    def __init__(self, obj, pbc=F, rcut=4.8):

        self.obj = obj
        self.pbc = pbc
        self.rcut = rcut

    def get_slatm_mbtypes(self):
        """ get slatm many-body types"""
        nzs = self.obj.nzs
        zsu = self.obj.zsu
        nzmax = np.max(nzs, axis=0)
        zsu = self.obj.zsu
        boas = [ [zi,] for zi in zsu ]
        bops = []
        for zi in zsu:
            #if nzmax[zi==zsu] > 1:
            bops.append( [zi,zi] )
        bops += list( itl.combinations(zsu,2) )

        obsolete = """
        bops = []
        if self.pbc:
            for bop in _bops:
                if self.iexist_2body(bop):
                    bops.append(bop)
        else:
            for bop in _bops:
                if (zi!=zj):
        _bots = []
        for ti in itl.product(zsu, repeat=3):
            if (ti not in _bots) and (ti[::-1] not in _bots):
                if self.iexist_3body(ti):
                    _bots.append(ti)
        bots = [ list(ti) for ti in _bots ] """

        # Note that the code below need to be replaced by the code commented
        # out above, for periodic systems in particular!!
        bots = []
        for i in zsu:
            for bop in bops:
                j,k = bop
                tas = [ [i,j,k], [i,k,j], [j,i,k] ]
                for tasi in tas:
                    if (tasi not in bots) and (tasi[::-1] not in bots):
                        nzsi = [ (zj == np.array(tasi)).sum() for zj in zsu ]
                        if np.all(nzsi <= nzmax):
                            bots.append( tasi )
        mbtypes = boas + bops + bots
        #nsx = np.array([len(mb) for mb in [boas,bops,bots]],np.int)
        #ins2 = np.cumsum(nsx)
        #ins1 = np.array([0,ins2[0],ins2[1]],np.int)
        return mbtypes

    def iexist_2body(self, mb):
        iok = F
        return iok

    def iexist_3body(self, mb):
        iok = F
        return iok

