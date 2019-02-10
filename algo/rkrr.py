#!/usr/bin/env python

import os,sys
import numpy as np
from cheminfo import *
from cheminfo.rw.xyz import *
from representation.xb import get_nzs

T,F = True,False

chemical_symbols = ['X',  'H',  'He', 'Li', 'Be',
                    'B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si',
                    'P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se',
                    'Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',
                    'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                    'At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu',
                    'Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']
s2z = dict( list(zip(chemical_symbols, list(range(len(chemical_symbols))) )) )

pi2 = np.pi*np.pi
a2b = 1.8897261258369282  # angstrom to bohr

np.set_printoptions(precision=2)

class rkrr(object):
    """
    recursive krr
    """
    def __init__(self, ys):
        """ Note that the column of `ys represents the same property
        generated from different level of theory """
        self._ys = np.array(ys)
        self.nm = len(ys)
        self.nl = len(ys[0]) # number of levels
        # initialize test set size `n2 to 0
        self.n2 = 0

    def init_m(self, obj):
        """ initialize molecules """
        #assert type(obj) is molecules
        self._zsu = np.unique( obj.zs )
        self._zs = obj.zs
        self._nas = obj.nas
        self._nsheav = obj.nsheav
        self._coords = obj.coords
        _ias2 = np.cumsum(self._nas)
        _ias1 = np.array([0,]+list(_ias2[:-1]),np.int)
        self._ias1 = _ias1
        self._ias2 = _ias2
        # get energy of dressed atom
        self._nzs = get_nzs(_ias1,_ias2,self._zs,self._zsu)

    def get_idx(self,idx=-1,n1s=[],n1max=-1,seed1=1,namin=1,namax=7):
        """
        get training/test idx

        i) AML: idx=-n ([n] target mols)
                namax=0 -> one training set, i.e., all amons
                namax=7 -> 7 training sets, the i-th set is {N_I = i} (i=1,2,...,7)
                namax=-5 -> one training set, comprising of all amons with N_I <= -namax
        ii) random sampling: idx>0 (__not__ quite useful?!)
        iii) manually select training set: idx=[0,3,10,...],
        """

        if isinstance(idx,int):
            if idx == 0:
                # only k1 is to be calculated, then we can randomly choose training data
                # from the molecules and shuffle many times to make an averaged LC
                _idx1 = np.arange(self.nm)
                idx2 = []
            elif idx > 0:
                np.random.seed(seed1) # fix the random sequence
                tidxs = np.random.permutation(self.nm)
                _idx1 = tidxs[:-idx]; idx2 = tidxs[-idx:]
                if n1max > 0:
                    _idx1 = _idx1[:n1max] # choose a subset
            else:
                tidxs = np.arange(self.nm); _idx1 = tidxs[:idx]; idx2 = tidxs[idx:]
        elif isinstance(idx,list):
            # idx as the maximal training set
            _idx1 = np.array(idx,np.int)
            tidxs = np.arange(self.nm)
            idx2 = np.setdiff1d(tidxs,_idx1)
        else:
            print('#ERROR: unsupported type of `idx')
            raise

        n2 = len(idx2) # now test set is fixed!!
        self.n2 = n2

        # now get smaller training set sizes
        nsu_heav = np.unique( self._nsheav ) # already in ascending order
        self.nsu_heav = nsu_heav
        idx1 = _idx1
        nn1 = len(n1s)
        aml = True
        if namax == 0:
            aml = False
            if nn1 == 0:
                # use only one training set, including all amons
                idxs1 = [_idx1]
            else:
                idxs1 = [ _idx1[:n1] for n1 in n1s ]
        elif namax < 0:
            # use only one training set, {N_I <= -namax}
            idx1 = _idx1[ np.logical_and(self._nsheav[_idx1]>=namin, self._nsheav[_idx1] <= -namax) ]
            idx2 = np.setdiff1d(tidxs,idx1)
            idxs1 = [idx1]; n1s = [len(idx1)]
        else:
            # use `namax-`namin+1 training sets to generate a LC
            idxs1 = []
            n1s = []
            # Note that `nheav may not be of the same order as the mol idx in filename,
            # so we'd better sort the idx of filenames! E.g., na=5 in frag_01.xyz while na=3 in frag_05.xyz
            idx1_sorted = []
            t = self._nsheav[_idx1]
            cnt = 0
            for na in range(namin,namax+1):
                if np.any(na==nsu_heav):
                    idx_i = _idx1[t==na]
                    leni = len(idx_i)
                    idx1_sorted += list(idx_i)
                    cnt += leni
                    n1s.append(cnt)
                else:
                    if cnt == 0:
                        n1s.append(np.nan) #
                    else:
                        n1s.append(cnt)
            print(' ** initial n1s = ', n1s)
            idx1 = idx1_sorted
        self.aml = aml
        self.n1s = n1s

        tidx = np.concatenate((idx1,idx2)).astype(np.int)
        n1,n2 = len(idx1),len(idx2)
        nt = n1+n2
        self.ys = self._ys[tidx]

        null = np.array([],np.int)
        _nas = self._nas[tidx]
        _coords = []; _zs = []
        for i1 in tidx:
            ib1,ie1 = self._ias1[i1],self._ias2[i1]
            _coords += list(self._coords[ib1:ie1])
            _zs += list(self._zs[ib1:ie1])
        _coords = np.array(_coords)
        _zs = np.array(_zs,np.int)

        self.nas1 = _nas[:n1]
        self.nas2 = null if n2 == 0 else _nas[n1:]
        self.nas = _nas

        self.nzs = self._nzs[tidx]
        self.nsheav = self._nsheav[tidx]

        # atomic index
        ias_e = np.cumsum(_nas)
        ias_b = np.concatenate(([0],ias_e[:-1]))
        ias1 = np.concatenate( [ np.arange(ias_b[i],ias_e[i]) for i in range(n1) ] )
        ias2 = null if n2 == 0 else np.concatenate( [ np.arange(ias_b[i],ias_e[i]) for i in range(n1,nt) ] )
        iast = np.concatenate((ias1,ias2))
        self.coords = _coords[iast]
        self.zs1 = _zs[ias1]
        self.zs2 = _zs[ias2]
        self.zs = np.concatenate((self.zs1,self.zs2))
        self.nat1 = len(ias1)
        self.nat2 = len(ias2)


    def calc_ae_dressed(self,nzs1,ys1,nzs2,ys2):
        esb = np.linalg.lstsq(nzs1,ys1)[0]
        ys1p = np.dot(nzs1,esb)
        #print ' +++++ ys1.shape, ys1p.shape = ', ys1.shape, ys1p.shape
        dys1 = ys1 - ys1p
        ys2_base = np.dot(nzs2,esb)
        dys2 = ys2 - ys2_base
        return dys1,dys2, ys2_base

    def calc_e_base(self,nzs1,ys1,nzs2):
        esb = np.linalg.lstsq(nzs1,ys1,rcond=None)[0]
        ys1p = np.dot(nzs1,esb)
        #print ' +++++ ys1.shape, ys1p.shape = ', ys1.shape, ys1p.shape
        dys1 = ys1 - ys1p
        ys2_base = np.dot(nzs2,esb)
        return dys1, ys2_base

    def run(self, mks, usebl=T, llambda=1e-10, iprt=True):
        """do KRR training & test

        vars
        ================
        """
        if isinstance(mks, (list,tuple)):
            tmpk1,tmpk2 = mks;
        elif isinstance(mks, str):
            if os.path.exists(mks):
                dic = np.load(mks)
                tmpk1,tmpk2 = [ dic[key].copy() for key in ['k1','k2'] ]
        else:
            print('#ERROR: unknow input')
            raise
        mk1,mk2 = tmpk1.copy(), tmpk2.copy()

        n2 = self.n2
        tidxs = np.arange(self.nm)
        idxs1 = tidxs[:-n2]; idxs2 = tidxs[-n2:]

        # now train many krr models
        nl = self.nl # number of levels
        mods = []
        n1sr = self.n1s[::-1] # order reversed
        ys2p = np.zeros(n2)
        ns_nested = []
        for l in range(nl-1):
            #lvi = levels[i]
            _ims1 = list(range(n1sr[l]))
            ims1 = _ims1 #[ np.logical_not( np.isnan(self.ys[_ims1,l]) ) ] # e.g., if E_QMC is not avail, was set to NaN
            n1 = len(ims1)
            ns_nested.append(n1)
            nzs1, nzs2 = self.nzs[ims1], self.nzs[idxs2]
            k1 = mk1[ims1][:,ims1]
            k1[np.diag_indices_from(k1)] += llambda
            k2 = mk2[:,ims1]
            if l==0: # one model
                _ys1 = self.ys[ims1,l]
                ys1, ys2_base = self.calc_e_base(nzs1,_ys1,nzs2)
            else: # i <= nl-2:
                #_ys1_l = self.ys[ims1,l]; _ys1_lm1 = self.ys[ims1,l-1]
                _ys1_l = self.ys[ims1,l] - self.ys[ims1,l-1]
                ys1_l, ys2_base_l = self.calc_e_base(nzs1,_ys1_l,nzs2)
                #ys1_lm1, ys2_base_lm1 = self.calc_e_base(nzs1,_ys1_lm1,nzs2)
                ys1 = ys1_l; ys2_base = ys2_base_l
                #ys1 = ys1_l - ys1_lm1; ys2_base = 0.
            alphas = np.linalg.solve(k1,ys1)
            ys2p_l = np.dot(k2,alphas) + ys2_base
            ys2p += ys2p_l
        print(' ** ns_nested = ', ns_nested )

        # last machine
        _n1s = self.n1s
        l = nl-1
        maes = []
        n1s = []
        for i,_n1 in enumerate(_n1s):
            _ims1 = np.arange(_n1)
            #print ' ims1 = ', _ims1, self.ys[_ims1,l]
            ims1 = _ims1[ np.logical_not( np.isnan(self.ys[_ims1,l]) ) ] # e.g., if E_QMC is not avail, was set to NaN
            n1 = len(ims1)
            nzs1, nzs2 = self.nzs[ims1], self.nzs[idxs2]
            if np.linalg.matrix_rank(nzs1) < len(nzs1[0]):
                print('%6s'%('None'))
                n1s.append(np.nan)
                maes.append(np.nan) #; rmses.append(np.nan)
                continue

            k1 = mk1[ims1][:,ims1]
            k1[np.diag_indices_from(k1)] += llambda
            k2 = mk2[:,ims1]
            #_ys1_l = self.ys[ims1,l]; _ys1_lm1 = self.ys[ims1,l-1]
            _ys1_l = self.ys[ims1,l] - self.ys[ims1,l-1]
            ys1_l, ys2_base_l = self.calc_e_base(nzs1,_ys1_l,nzs2)
            #ys1_lm1, ys2_base_lm1 = self.calc_e_base(nzs1,_ys1_lm1,nzs2)
            ys1 = ys1_l
            #ys1 = ys1_l - ys1_lm1
            ys2_base = ys2_base_l
            alphas = np.linalg.solve(k1,ys1)
            #print(' --> ys2p, k2*alpha, ys2_base = ',ys2p, np.dot(k2,alphas), ys2_base)
            ys2p_final = ys2p + np.dot(k2,alphas) + ys2_base

            dys = ys2p_final - self.ys[idxs2,l]
            print('%6d %.2f'%(len(ims1), dys[0]))
            maes.append( dys[0] )
            n1s.append(n1)
        self.maes = maes
        self.n1s = n1s

