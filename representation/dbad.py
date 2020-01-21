
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolDescriptors
import os, sys
import numpy as np
import scipy.spatial.distance as ssd
import ase.units as au
import ase, copy
import multiprocessing
import copyreg
import types as _TYPES

#global const
#const = 1./np.sqrt(2.*np.pi)

## register instance method
## otherwise, the script will stop when calling the function
## multiprocessing() with error:
## ``TypeError: can't pickle instancemethod objects
def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__self__.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)
copyreg.pickle(_TYPES.MethodType, _reduce_method)


def get_mbtypes(zs):
    """
    get many-body types
    """
    # atoms that cannot be J in angle IJK or J/K in dihedral angle IJKL
    zs1 = [1,9,17,35,53]

    zs.sort()
    nz = len(zs)

    # 2-body
    mbs2 = []
    mbs2 += [ '%d-%d'%(zi,zi) for zi in zs ]
    for i in range(nz):
        for j in range(i+1,nz):
            mbs2.append( '%d-%d'%(zs[i],zs[j]) )

    # 3-body
    mbs3 = []
    zs2 = list( set(zs).difference( set(zs1) ) )
    zs2.sort()
    nz2 = len(zs2)
    for j in range(nz2):
        for i in range(nz):
            for k in range(i,nz):
                type3 = '%d-%d-%d'%(zs[i],zs2[j],zs[k])
                if type3 not in mbs3: mbs3.append( type3 )

    # 4-body
    mbs4 = []
    for j in range(nz2):
        for k in range(j,nz2):
            for i in range(nz):
                for l in range(nz):
                    zj,zk = zs2[j],zs2[k]
                    zi,zl = zs[i],zs[l]
                    if j == k:
                        zi,zl = min(zs[i],zs[l]), max(zs[i],zs[l])
                    type4 = '%d-%d-%d-%d'%(zi,zj,zk,zl)
                    if type4 not in mbs4: mbs4.append( type4 )
    return [mbs2,mbs3,mbs4]


class dbads(object):
    """
    generate dbad for all molecules
    """

    def __init__(self, fs, rcut=9.0, nprocs=[1,1], ibrute=False, cdistr=True):

        self.objs = []
        self.nm = len(fs)
        ms = []; zsr = set()
        for i in range(self.nm):
            m = Chem.MolFromMolFile(fs[i], removeHs=False); ms.append(m)
            zsr.update( [ ai.GetAtomicNum() for ai in m.GetAtoms() ] )
        print(' * all files processed')
        zsr = list(zsr)
        self.zsr = zsr
        if ibrute:
            # get all possible many-body types by brute-force, i.e.,
            # enumerating all possible combinations
            mbs = get_mbtypes( zsr )
        else:
            # get all many-body types from molecules
            mbs = [ ]
            imbt = True
            objs = []
            if nprocs[0] == 1:
                for i,mi in enumerate(ms):
                    ipt = [mi,mbs,rcut,imbt]
                    objs.append( self.processInput(ipt) )
            else:
                pool = multiprocessing.Pool(processes=nprocs[0])
                ipts = [ [mi,mbs,rcut,imbt] for mi in ms ]
                objs = pool.map(self.processInput, ipts)
            mbs2 = set([]); mbs3 = set([]); mbs4 = set([])
            for obj in objs:
                mbs2.update( obj.mbs2 )
                mbs3.update( obj.mbs3 )
                mbs4.update( obj.mbs4 )
            for _mbs in [mbs2,mbs3,mbs4]:
                t = list(_mbs); t.sort(); mbs.append(t)
        print(' * many-body types obtained')
        self.mbs = mbs

        if cdistr:
            imbt = False
            if nprocs[1] == 1:
                for i,mi in enumerate(ms):
                    print('%d/%d'%(i+1,self.nm))
                    ipt = [mi,mbs,rcut,imbt]
                    self.objs.append( self.processInput(ipt) )
            else:
                pool = multiprocessing.Pool(processes=nprocs[1])
                ipts = [ [mi,mbs,rcut,imbt] for mi in ms ]
                self.objs = pool.map(self.processInput, ipts, 5)


    def processInput(self, ipt):
        mi, mbs, rcut, imbt = ipt
        obj = dbad(mi, mbs, imbt, rcut=rcut)
        obj.get_all()
        return obj



class dbad(object):
    """
    distribution of bond, angle and dihedral angles
    """
    def __init__(self,m,mbs,imbt,rcut=9.0,dgrids=[0.03,0.03,0.03],sigmas=[0.05,0.05,0.05]):
        """
        vars
        =================================
        m -- rdkit molecule object
        mbs -- [mbs2,mbs3,mbs3], `mbs2: 2-body types
        """

        self.m = m
        self.rcut = rcut
        self.imbt = imbt # get many-body types only?
        self.dgrids = dgrids
        self.sigmas = sigmas
        self.na = m.GetNumAtoms()
        self.nb = m.GetNumBonds()
        self.zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ], np.int)
        self.iasb = []
        for ib in range(self.nb):
            bi = self.m.GetBondWithIdx(ib)
            ia1,ia2 = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
            self.iasb.append( [ia1,ia2] )

        coords = get_coords(m)
        self.ds = ssd.squareform( ssd.pdist(coords) )

        if self.imbt:
            self.mbs2, self.mbs3, self.mbs4 = [], [], []
        else:
            mbs2, mbs3, mbs4 = mbs
            self.mbs2, self.mbs3, self.mbs4 = mbs2, mbs3, mbs4
            dic2 = {}; dic3 = {}; dic4 = {}
            for mb2 in mbs2: dic2[mb2] = []
            for mb3 in mbs3: dic3[mb3] = []
            for mb4 in mbs4: dic4[mb4] = []
            self.dic2, self.dic3, self.dic4 = dic2, dic3, dic4

            n2 = int( (self.rcut+1)/self.dgrids[0] )
            xs2 = np.linspace(0,rcut+1.0,n2+1)
            self.xs2 = xs2

            n3 = int( (np.pi*(7./6.)/self.dgrids[1]) )
            xs3 = np.linspace(0,np.pi*(7./6),n3+1)
            self.xs3 = xs3

            n4 = int( (np.pi*(8./6.)/self.dgrids[2]) )
            xs4 = np.linspace(-np.pi/6.,np.pi*(7./6),n4+1)
            self.xs4 = xs4


    def get_2body(self,conn=False):
        """
        the atomic pair contrib

        vars
        ====================
        conn: should the pair of atoms be connected?
        """
        for ia in range(self.na):
            for ja in range(ia+1,self.na):
                dij = self.ds[ia,ja]
                zi, zj = [ self.zs[idx] for idx in [ia,ja] ]
                if zi > zj:
                    t = zi; zi = zj; zj = t
                type2 = '%d-%d'%(zi,zj)
                if self.imbt:
                    if type2 not in self.mbs2: self.mbs2.append(type2)
                    continue
                #assert type2 in self.dic2.keys()
                if dij <= self.rcut: self.dic2[type2] += [dij]

        if not self.imbt:
            distr2 = []
            for mb2 in self.mbs2:
                self.gaussian(self.xs2, self.dic2[mb2], self.sigmas[0])
                distr2.append( self.ys )
            self.distr2 = distr2


    def get_3body(self):
        """
        3-body parts: angles spanned by 3 adjacent atoms,
                      must be a valid angle in forcefield
        """
        for aj in self.m.GetAtoms():
            j = aj.GetIdx()
            zj = self.zs[j]
            neibs = aj.GetNeighbors()
            nneib = len(neibs)
            if zj > 1 and nneib > 1:
                  for i0 in range(nneib):
                    for k0 in range(i0+1,nneib):
                        i, k = neibs[i0].GetIdx(), neibs[k0].GetIdx()
                        ias = [i,j,k]
                        if self.zs[i] > self.zs[k]: ias = [k,j,i]
                        zsi = [ self.zs[ia] for ia in ias ]
                        type3 = '-'.join(['%d'%zi for zi in zsi ])
                        if self.imbt:
                            if type3 not in self.mbs3: self.mbs3.append(type3)
                            continue
                        #assert type3 in self.dic3.keys()
                        _theta = rdMolTransforms.GetAngleRad(self.m.GetConformer(), ias[0],ias[1],ias[2])
                        theta = _theta*(-1.) if _theta < 0. else _theta
                        #assert theta <= np.pi
                        if theta > np.pi:
                            raise '#ERROR: `thea > np.pi?'
                        self.dic3[type3] += [theta]

        if not self.imbt:
            distr3 = []
            for mb3 in self.mbs3:
                self.gaussian(self.xs3, self.dic3[mb3], self.sigmas[1])
                distr3.append( self.ys )
            self.distr3 = distr3


    def get_4body(self):
        """
        4-body parts: dihedral angles in forcefield
        """
        for ib in range(self.nb):
            j, k = self.iasb[ib]
            if self.zs[j] > self.zs[k]:
                tv = k; k = j; j = tv
            neibs1 = self.m.GetAtomWithIdx(j).GetNeighbors(); n1 = len(neibs1);
            neibs2 = self.m.GetAtomWithIdx(k).GetNeighbors(); n2 = len(neibs2);
            for i0 in range(n1):
                for l0 in range(n2):
                    i = neibs1[i0].GetIdx(); l = neibs2[l0].GetIdx()
                    ias = [i,j,k,l]
                    if len(set(ias)) == 4:
                        if self.zs[j] == self.zs[k]:
                            if self.zs[i] > self.zs[l]:
                                ias = [l,k,j,i]
                        zsi = [ self.zs[ia] for ia in ias ]
                        type4 = '-'.join([ '%d'%zi for zi in zsi ])
                        if self.imbt:
                            if type4 not in self.mbs4: self.mbs4.append(type4)
                            continue
                        #assert type4 in self.dic4.keys()
                        _tor = rdMolTransforms.GetDihedralRad(self.m.GetConformer(), ias[0],ias[1],ias[2],ias[3])
                        tor = _tor*(-1.) if _tor < 0. else _tor
                        #assert tor <= np.pi
                        if tor > np.pi:
                            #print type4, tor
                            raise '#ERROR:'
                        self.dic4[type4] += [tor]
        if not self.imbt:
            distr4 = []
            for mb4 in self.mbs4:
                self.gaussian(self.xs4, self.dic4[mb4], self.sigmas[2])
                distr4.append( self.ys )
            self.distr4 = distr4


    def gaussian(self, xs, rs, sigma):
        ys = np.zeros( xs.shape )
        for r in rs:
            ys += np.exp( -(xs-r)**2/(2.*sigma*sigma) )/(sigma * np.sqrt(2.*np.pi))
        self.ys = ys


    def get_all(self):
        self.get_2body()
        self.get_3body()
        self.get_4body()
        if not self.imbt:
            self.distr = np.concatenate( (self.distr2 + self.distr3 + self.distr4), axis=0 )


def get_coords(m):
    na = m.GetNumAtoms()
    zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ])
    #assert m.GetNumConformers() == 1, '#ERROR: more than 1 Conformer exist?'
    c0 = m.GetConformer(-1)
    coords = []
    for i in range(na):
        coords_i = c0.GetAtomPosition(i)
        coords.append([ coords_i.x, coords_i.y, coords_i.z ])
    return np.array(coords)








