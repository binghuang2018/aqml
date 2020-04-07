

import os,sys,ase
import ase.data as ad
import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np
import io2.data as dt

import multiprocessing
import copy_reg
import types as _TYPES

global h2kc, h2e
h2kc = 627.5094738898777
h2e = 27.211386024367243

## register instance method
## otherwise, the script will stop with error:
## ``TypeError: can't pickle instancemethod objects
def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(_TYPES.MethodType, _reduce_method)


def s2z(obj):
    return ad.chemical_symbols.index(obj)

class XYZFile(object):

    def __init__(self, fin):
        cs = file(fin).readlines()
        na = int(cs[0])
        self.property_str = cs[1]
        geom = np.array( [ ci.split()[:4] for ci in cs[2:na+2] ] )
        self.symbs = list(geom[:,0])
        self.coords = geom[:,1:4].astype(np.float)
        self.atoms = ase.Atoms(self.symbs, self.coords)
        self.zs = self.atoms.numbers

    def read_property(self, qcl, property_name):
        """
        for extended xyz file
        """
        zs = self.zs
        L2c = self.property_str.strip().split()
        nprop = len(L2c)
        properties = np.array(L2c[2:]).astype(np.float)

        properties_u = []
        if nprop == 14: #QM7b dataset
            properties_u = np.array(L2c).astype(np.float)
        else:
            if nprop == 24: # my format
                [A, B, C, mu, alpha, homo, lumo, gap, r2, zpe, U0, U, H, G, Cv,  E,Enn, Href,Gref,U0ref,Uref, omega1] = properties
            elif nprop == 17: # Raghu's format
                [A, B, C, mu, alpha, homo, lumo, gap, r2, zpe, U0, U, H, G, Cv] = properties
                Enn = 0.0
                dic = dt.retrieve_esref(qcl); symbs = dic.keys(); zsR = np.array([ s2z(si) for si in symbs ])
                esR = np.array([ dic[si] for si in symbs ])
                nzR = len(zsR); idxsZ = np.arange(nzR)
                ots = []
                for zi in zs:
                    izs = idxsZ[zsR==zi]
                    ots.append( esR[izs[0]] )
                ots = np.array(ots).sum(axis=0)
                #print ' -- ots = ', ots*h2kc
                U0ref, Uref, Href, Gref = ots
                omega1 = 0.0
            else:
                raise ' # nprop unknown??'

            Enn, U0, U, H, G, zpe = np.array([Enn, U0, U, H, G, zpe])*h2kc
            U0ref, Uref, Href, Gref = np.array([ U0ref, Uref, Href, Gref ])*h2kc
            homo, lumo, gap = np.array([homo, lumo, gap])*h2e
            E = U0 - zpe

            if property_name == 'E': # total energy
                properties_u = [E, ] # Eref = U0ref
            elif property_name == 'AE': # atomization of energy
                properties_u = [E - U0ref, ] # Eref = U0ref
            elif property_name == 'AH': # atomization of enthalpy
                properties_u = [H - Href, ] if H != 0.0 else [E ]
            else:
                #               1  2   3  4   5    6    7      8   9    10    11   12 13 14 15 16   17    18   19    20   21      22
                properties_u = [H, G, U0, U, Cv, zpe,  mu, alpha, r2, homo, lumo, gap, A, B, C, E, Enn, Href,Gref,U0ref,Uref, omega1];

        self.properties = properties_u

    def write_xyz(self, filename, Line2):
        fid = open(filename, 'w')
        aio.xyz.simple_write_xyz(fid, [self.atoms,], Line2)


class XYZFiles(object):

    def __init__(self, fs, nproc=1):

        self.n = len(fs)
        if nproc == 1:
            self.objs = []
            for i,f in enumerate(fs):
                #print i+1, f
                self.objs.append( self.processInput(f) )
        else:
            pool = multiprocessing.Pool(processes=nproc)
            self.objs = pool.map(self.processInput, fs)

    def processInput(self, f):
        obj = XYZFile(f)
        return obj


    def get_statistics(self):
        """
        `zs, `nas, `nhass, `coords, etc
        """
        zs = []
        zsr = []
        nas = []
        nhass = []
        zsu = set([])
        coords = []
        for i in range(self.n):
            obj_i = self.objs[i]
            coords.append( obj_i.coords )
            zsi = obj_i.zs
            nhass.append( (np.array(zsi)>1).sum() )
            nas.append( len(zsi) )
            zsu.update( zsi )
            zsr += list(zsi)
            zs.append( zsi )
        zsu = list(zsu)
        nzu = len(zsu)
        zsu.sort()
        nzs = np.zeros((self.n, nzu), np.int32)
        for i in range(self.n):
            for iz in range(nzu):
                ioks = np.array(zs[i]) == zsu[iz]
                nzs[i,iz] = np.sum(ioks)
        self.nzs = nzs
        self.zsu = zsu
        self.zs = zs
        self.zsr = np.array(zsr,np.int32)
        self.nas = np.array(nas,np.int32)
        self.nhass = np.array(nhass,np.int32)
        self.coords = coords

