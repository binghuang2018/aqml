#!/usr/bin/env python

"""
graph of graph, i.e., connectivity between amons
"""


from functools import reduce
import os, sys, re, copy, ase, time
import numpy as np
import aqml.cheminfo as co
from aqml.cheminfo.rw.ctab import write_ctab
import multiprocessing
import itertools as itl
from aqml.cheminfo.molecule.elements import Elements
import aqml.cheminfo.oechem.core as coc

rscov0 = Elements().rcs
rsvdw0 = Elements().rvdws

T,F = True,False


                    obsolete = """
                    if not iqg:
                        if not use_f90:
                            ggo = GG(obj, lms, k2, nasv, nprocs)
                            gc = ggo.gc
                            gv = ggo.gv
                    """

class GG(object):

    """
    Graph of Graph
    I.e., mol complex
    """

    def __init__(self, q, ms, k2, nasv=None, nprocs=1):

        self.q = q  # newmol class
        self.ms = ms  # list of newmol's
        self.nm = len(ms)
        self.nprocs = nprocs
        if nasv is None:
            nasv = []
            for mi in ms:
                nasv.append(mi.nav)
        self.nasv = np.array(nasv, dtype=int)
        nasp = self.nasv[..., np.newaxis] + [self.nasv]
        np.fill_diagonal(nasp, 999)
        fl = (nasp <= k2)
        bs = np.array( np.where(np.triu(fl)), dtype=int).T
        self.bs = bs
        nb = len(bs)
        print('    Total number of amons pairs with N_I<=k2: ', nb)
        self.nb = nb
        self.ibs = np.arange(nb)

        ipts = []
        for b in self.bs:
            mi = self.ms[b[0]]; mj = self.ms[b[1]]
            ipts.append( [q, mi, mj] )
        self.ipts = ipts


    @property
    def ibsn(self):
        """
        non-bonds
        I.e., a pair of amons that are too distant
        """
        if not hasattr(self, '_bsn'):
            pool = multiprocessing.Pool(processes=self.nprocs)
            _bsn = pool.map(get_is_nbij, self.ipts)
            self._bsn = np.array(_bsn, dtype=int)
            print('   Total number of non-bonded amon pairs: ', (self._bsn==1).sum())
        return self._bsn


    @property
    def ibsc(self):
        """
        bonds that are covalent
        I.e., `bond` here refer to a pair of amons that approximated
        in query mol
        """
        if not hasattr(self, '_bsc'):
            pool = multiprocessing.Pool(processes=self.nprocs)
            _bsc = pool.map(get_is_covij, self.ipts)
            self._bsc = np.array(_bsc, dtype=int)
            print('   Total number of cov-bonded amon pairs: ', (self._bsc==1).sum())
        return self._bsc



    @property
    def ibsv(self):
        """
        bonds that are bonded through vdw interaction
        I.e., a pair of amons containing some vdw bond
        in query mol
        """
        if not hasattr(self, '_bsv'):
            _ibsv = np.zeros(self.nb).astype(int)
            ibs = self.ibs[ np.logical_not( np.logical_or(self.ibsn, self.ibsc) ) ]
            print('    Initial number of vdw bonds to be further assessed: ', len(ibs) )
            ipts = [ self.ipts[ib] for ib in ibs ]
            pool = multiprocessing.Pool(processes=self.nprocs)
            _bsv = pool.map(get_is_vdwij, ipts)
            bsv = np.array(_bsv, dtype=int)
            _ibsv[ibs[bsv==1]] = 1
            self._bsc = _ibsv
        return self._bsc


    @property
    def gc(self):
        """
        covalent graph
        """
        if not hasattr(self, '_gc'):
            g = np.zeros( (self.nm, self.nm) )
            for ib in self.ibs[self.ibsc==1]:
                i, j = self.bs[ib]
                g[i,j] = g[j,i] = 1
            self._gc = g
        return self._gc


    @property
    def gv(self):
        """
        covalent graph
        """
        if not hasattr(self, '_gv'):
            g = np.zeros( (self.nm, self.nm) )
            for ib in self.ibs[self.ibsv==1]:
                i, j = self.bs[ib]
                g[i,j] = g[j,i] = 1
            self._gv = g
        return self._gv



def get_is_vdwij(ipt):
    """
    get get vdw connectivity between the i- and j-th amons
    This seperate function is necessary for multithreading!!
    """
    obj, mi, mj = ipt
    pls0 = obj.pls
    g0 = obj.g
    vc = 0 # connected by vdw bond?

    # now bsv of current mol complex
    mc = molcplx([mi,mj])
    c = coc.newmol(mc.zs, mc.chgs, mc.bom, mc.coords)
    #c.write_ctab('temp.sdf')
    nbv = 0; nbv0 = 0
    bsv = []; bsv0 = []
    for b in c.ncbs:
        a1,a2 = b
        # include only intermolecular vdw bond of __this__ `comb
        if c.pls[a1,a2]==0:
            bsv.append(b)
            nbv += 1
            b0 = [ mc.iasq[a0] for a0 in b ]
            if np.all( [a0!=None for a0 in b0] ):
                b0.sort()
                bsv0.append(b0)
                nbv0 += 1

    nbo1 = c.nbo1 #mi.nbo1 + mj.nbo1
    if nbv0 > 0 and nbv0==nbv:
        if nbv==1: # and nbv0>=1:
            bv0 = bsv0[0]
            #if bv0 in obj.chbs_ext:
            #    print("    found co-planar HB's in q, but only 1 of such HB exist in this amon, skip!")
            #elif bv0 in obj.hbs:
            #    print("    found HB in", mc.can0)
            #    if nbo1 <= 2:
            #        vc = 1
            #else:
            #    print("    found vdw bond (non-HB) in", mc.can0)
            #    if nbo1 <= 2: # 1
            #        vc = 1
            if (bv0 not in obj.chbs_ext) and nbo1 <= 2:
                #print('        nbv=1 and nbo1<=2 for ', mc.can0, 'accept!')
                vc = 1
        else:
            ioks = [ bvi in obj.chbs for bvi in bsv0 ]
            if np.any(ioks):
                if np.sum(ioks) >= 2:
                    print("   found more than 2 HB's in a conj env", mc.can0, 'keep it!')
                    vc = 1
            else:
                if nbo1 <= 2*nbv-1:
                    print("    found multiple vdw bonds in", mc.can0,  'bsv=',bsv)
                    vc = 1

    return vc


def get_is_covij(ipt):
    """
    get get cov connectivity between the i- and j-th amons
    This seperate function is necessary for multithreading!!
    """
    obj, mi, mj = ipt
    pls0 = obj.pls
    g0 = obj.g
    cc = 0 # connected by covalent bond? (I.e., some atom from mi is bonded to some atom in mj)

    na = len(np.intersect1d(mi.iasvq, mj.iasvq))
    if na > 0:
        cc = 1
    else:
        gij = g0[mi.iasvq][:,mj.iasvq]
        plij = pls0[mi.iasvq][:,mj.iasvq]
        if np.any(gij > 0) or (np.all(plij > 0) and np.any(plij <= 4)):
            cc = 1
    return cc


def get_is_nbij(ipt):
    """
    non-bond formed between `mi and `mj
    """
    obj, mi, mj = ipt
    # not-bonded?
    nb = 0
    iasvq = mi.iasvq; rsi = rsvdw0[obj.zs[iasvq]] # mi,mj: ExtM class
    jasvq = mj.iasvq; rsj = rsvdw0[obj.zs[jasvq]]
    dsij = obj.ds[iasvq][:,jasvq]
    dsijmax = ( rsi[...,np.newaxis]+[rsj] ) * 1.25 # 1.20 was used instead to get `ncbs
    if np.all(dsij > dsijmax):
        nb = 1
    return nb



def get_iexist_vdw_bond(ipt):
    """
    check if a given mol pair contain any vdw bond, which exists
    in the query mol. Note that input mol pairs must have cc=0.
    """
    obj, mi, mj = ipt
    iok = F
    if np.any( [ set(b) <= set(mi.iasq+mj.iasq) for b in obj.ncbs ] ):
        iok = T
    return iok



