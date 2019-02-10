
from __future__ import print_function

import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np


T, F = True, False


class molecule(object):

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
        a, b, c = np.norm(cell, axis=0)
        self.a = a
        self.b = b
        self.c = c
        #self.pbc = pbc
        if na == 1:
            ds = np.array([[0.]])
        else:
            ds = ssd.squareform( ssd.pdist(coords) )
        self.ds = ds
        self.ias = np.arange(na)


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
            for sign in [1, -1]:
                ns.append( self.get_nmax(ia, axis, sign) )

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
        return [zs, coords]



def update_m(obj, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom `ia
    for periodic systems (or very large system)
    """
    zs, coords, c = obj
    v1, v2, v3 = c
    vs = ssd.norm(c, axis=0)

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

    return obj_u


