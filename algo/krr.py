#!/usr/bin/env python

import os,sys
import numpy as np
import cheminfo.core as cc
from cheminfo.rw.xyz import *
import scipy.spatial.distance as ssd
import itertools as itl
from .efit import initb_g

T, F = True, False
pi2 = np.pi*np.pi
a2b = 1.8897261258369282  # angstrom to bohr
# e.g., for S, [ 1s^2 2s^2 2px^2 2py^2 2pz^2 ], i.e., 5 subshells
n0 = 0
nshl_core = {1:0, 2:0, 3:1, 4:1, 5:1, 6:1+n0, 7:1+n0, 8:1+n0, 9:1+n0, 16:5+n0, 17:5+n0}

np.set_printoptions(precision=2)


class krr(object):

    def __init__(self, ys):
        _ys = np.array(ys)
        if len(_ys.shape) == 2:
            nr, nc = _ys.shape
            assert nr > nc, '#ERROR: each column of `ys should correspond to one property!!'
        self._ys = _ys
        assert np.logical_not(np.any(np.isnan(_ys))), '#ERROR: property is NaN, check your sdf file'
        self._nm = len(ys)
        # initialize test set size `n2 to 0
        self.n2 = 0

    def init_m(self, obj, nbody=1, icn=F, nproc=1):
        """ initialize molecules """
        #assert type(obj) is molecules
        self.obj = obj
        self._zsu = np.unique( obj.zs )
        self._zs = obj.zs
        self._nas = obj.nas
        self._nsheav = obj.nsheav
        self._coords = obj.coords
        _ias2 = np.cumsum(self._nas)
        _ias1 = np.array([0,]+list(_ias2[:-1]),np.int)
        self._ias1 = _ias1
        self._ias2 = _ias2
        self._imcs = obj.imcs # is molecular complex?
        # get energy of dressed atom
        if nbody == 1 and (not icn):
            self._nzs = obj.nzs
        else:
            self._nzs = self.calc_mbts(nbody=nbody, icn=icn, nproc=nproc)
        if self._nm != len(self._nas):
            raise Exception('#ERROR: len(ys)!= len(_nas)??')

    def get_idx(self,idx=1,n2=None,n1s=[],iaml=T,seed1=1,namin=1,namax=7):
        """
        get training/test idx

        :param iaml: is amons-based krr? True or False. If set to False, then training set will be randomly selected
        :type iaml: bool
        :param idx: specifying the training & test molecules
        :type idx: integer
        """
        self.iaml = iaml
        assert isinstance(idx,int), '#ERROR: idx should be a int'
        if idx < 0: idx = -idx
        if iaml:
            tidxs = np.arange(self._nm)
            tidx1 = tidxs[:-idx]; idx2 = tidxs[-idx:]
        else:
            np.random.seed(seed1) # fix the random sequence
            _tidx = np.random.permutation(self._nm)
            tidx1 = _tidx[:-idx];
            if n2:
                idx2 = _tidx[-n2:]
            else:
                print(' ** choose all mols[-idx:] as test set')
                idx2 = _tidx[-idx:]

        self.nm = self._nm

        n2 = len(idx2) # now test set is fixed!!
        # now get all training sets
        nsu_heav = np.unique( self._nsheav ) # already in ascending order
        self.nsu_heav = nsu_heav
        nn1 = len(n1s)
        #print(' ######################## n1s = ', n1s)
        if nn1 != 0:
            idxs1 = [ tidx1[:n] for n in n1s ]
        else:
            # Note that `nheav of mols may not be sorted by numerical
            # value. Here we sort it for convinience of reference later.
            tidx1_sorted = []
            t = self._nsheav[tidx1]
            cnt = 0
            for na in range(namin,namax+1):
                if np.any(na==nsu_heav):
                    i = tidx1[t==na]
                    tidx1_sorted += list(i)
                    cnt += len(i)
                    n1s.append(cnt)
                else:
                    if cnt == 0:
                        n1s.append(np.nan) #
                    else:
                        n1s.append(cnt)
                #print('na,cnt,n1s=',na,cnt,n1s)
            #print('cnts = ', cnts)
            tidx1 = tidx1_sorted
            idxs1 = [ tidx1[:n] for n in n1s ]
        self.n1s = n1s
        #print('*************n1s=', n1s)

        tidx = np.concatenate((tidx1,idx2)).astype(np.int)
        n1, n2 = len(tidx1), len(idx2)
        self.n2 = n2
        nt = n1 + n2
        self.ys = self._ys[tidx]
        #print('tidx=',tidx.shape, 'ys=',self.ys.shape)

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
        self.nas2 = null if n2 == 0 else _nas[-n2:]
        self.nas = np.concatenate((self.nas1,self.nas2))

        self.nzs = self._nzs[tidx]
        self.imcs = self._imcs[tidx]
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

    def calc_mbts(self, nbody=2, ival=F, icn=T, nproc=1):
        """
        calc many-body terms, including types and nzs
        """
        bts = initb_g(self.obj, icount=T, ival=ival, nbody=nbody, \
                            iconn=T, icn=icn, nproc=nproc)
        mbts1, nmbts1, mbts2, nmbts2 = bts[:4]
        nmbts = nmbts1 if nbody == 1 else nmbts2
        return np.array(nmbts,dtype=int) # np.concatenate((nmbts1,nmbts2), axis=1)

    @staticmethod
    def calc_ae_dressed(imcs1,ns1,ys1,ns2,ys2):
        """
        calc dressed atom or/and bond energies as baseline

        :imcs1 -- mols that are molecular complexs are not used
                  for regression of atomic energy
        """
        flt = np.logical_not(imcs1)
        esb,_,rank,_ = np.linalg.lstsq(ns1[flt],ys1[flt],rcond=None)
        ys2_base = np.zeros(np.array(ys2).shape)
        if rank < len(ns1[0]):
            # number of molecules .le. number of elements involved,
            # that is, lstsq won't be effective
            return ys1,ys2,ys2_base
        ys1p = np.dot(ns1,esb)
        #print ' +++++ ys1.shape, ys1p.shape = ', ys1.shape, ys1p.shape
        dys1 = ys1 - ys1p
        ys2_base = np.dot(ns2,esb)
        dys2 = ys2 - ys2_base
        return dys1,dys2, ys2_base

    def run(self, mks, usek1o=F, n1sr=None, n2r=None, idx2r=None, \
            exclude=None, seed2r=None, usebl=T, llambda=1e-10, iprt=T):
        """do KRR training & test

        vars
        ========================================================
        case 1:
        usek1o : use k1 only? Used for random sampling
        seed2 : used to choose a fixed test set (with size `_n2)
                 if usek1o=T or self.n2=0
        exclude : a set of molecules specified by indices to be excluded
                 for training. Used for debugging purpose only.
        """

        if isinstance(mks, (list,tuple)):
            tk1,tk2 = mks;
        elif isinstance(mks, str):
            if os.path.exists(mks):
                dic = np.load(mks)
                tk1,tk2 = [ dic[key].copy() for key in ['k1','k2'] ]
        else:
            raise Exception('#ERROR: `mks should be either [ks1,ks2] or *.npz file')

        # encountered some unexpected behavior before, i.e.,
        # somehow the var ks1,ks2 were modified after calling
        # run() once...
        # Here we make a copy of the original data for safety!
        _mk1,_mk2 = tk1.copy(), tk2.copy()
        n1a, n2a = _mk1.shape[0], np.array(_mk2).shape[0]
        assert n1a <= self.nm

        if self.n2==0: usek1o = T
        if usek1o:
            assert n1sr!=None, '#ERROR: `n1sr is None!!'
            assert idx2r!=None, '#ERROR: both n2r/idx2r are None!'
            n1s = n1sr
            if isinstance(idx2r,(np.ndarray,list,tuple)):
                idx2 = idx2r
            elif isinstance(idx2r,int): # assume an integer
                assert seed2r!=None, '#ERROR: `seed2r is None!'
                np.random.seed(seed2r)
                tidx = np.random.permutation(n1a)
                idx2 = tidx[-idx2r:]
            else:
                raise Exception(' type of `idx2r not supported')
            idx2.sort()
            n2 = len(idx2)
            idx1 = np.setdiff1d(tidx,idx2)
            mk1 = _mk1[idx1][:,idx1]
            mk2 = _mk1[idx2][:,idx1]
        else:
            tidx = np.arange(self.nm)
            mk1 = _mk1; mk2 = _mk2
            n1s = self.n1s; n2 = self.n2
            idx1 = tidx[:-n2]; idx2 = tidx[-n2:]

        _ys1 = self.ys[idx1]
        _ys2 = self.ys[idx2]
        if usebl:
            _nzs1 = self.nzs[idx1]
            _nzs2 = self.nzs[idx2]
            _imcs1 = self.imcs[idx1]
            _imcs2 = self.imcs[idx2]

        n1so = []; maes = []; rmses = []; dys = []; errsmax = []
        #print(' n1s = ', n1s)
        for i,n1 in enumerate(n1s):
            #print('n1=',n1)
            if n1 in [np.nan,None]:
                print('%6s'%('None'))
                maes.append(np.nan); rmses.append(np.nan)
                continue

            if exclude:
                _idx1 = np.setdiff1d(np.arange(n1),exclude)
            else:
                _idx1 = np.arange(n1)

            if len(_idx1) == 0:
                continue

            # now update n1
            n1u = len(_idx1)
            n1so.append(n1u)
            #print('n1u=',n1u)

            #print('idx1=',_idx1, 'ys.size=',len(_ys1))
            _ys1i = _ys1[_idx1]
            _ys2i = _ys2
            #print('ys1i.shape = ', _ys1i.shape, 'ys2i.shape = ', _ys2i.shape)
            ys2i_base = 0.
            if usebl:
                ns1 = _nzs1[_idx1]
                imcs1 = _imcs1[_idx1]
                rank1 = np.linalg.matrix_rank(ns1)
                if rank1 < len(ns1[0]):
                    print('%6s # rank(ns1)=%d less than len(nzu)=%d'%('None', rank1, len(ns1[0])))
                    #print('             ')
                    maes.append(np.nan); rmses.append(np.nan)
                    continue
                else:
                    ns2 = _nzs2
                    if not np.all( [ si > 1 for si in ns1.shape ] ): continue
                    ys1i, ys2i, ys2i_base = krr.calc_ae_dressed(imcs1, ns1,_ys1i,ns2,_ys2i)
                    y1m,y2m = [ np.mean(np.abs(_)) for _ in [ys1i,ys2i]]
                    #print('n1=', n1, 'linear regress of E: MAE=%.2f (train), %.2f (test)'%(y1m,y2m))
            else:
                ys1i, ys2i = _ys1i, _ys2i

            k1 = mk1[_idx1][:,_idx1] #[:n1][:,:n1]
            k1[np.diag_indices_from(k1)] += llambda
            alphas = np.linalg.solve(k1,ys1i)
            k2 = mk2[:,_idx1] #:n1]
            #print ' k2.shape = ', k2.shape
            ys2i_p = np.dot(k2,alphas)
            #ys2i_ps.append( ys2i_p + ys2i_base )
            #print 'ys2i_p = ', ys2i_p
            dys2i = ys2i_p-ys2i
            dys.append( dys2i )
            dys2ia = np.abs(dys2i)
            errmax = np.unique( dys2i[np.max(dys2ia)==dys2ia] )[0]
            if n2 == 1:
                mae = dys2i[0]; rmse = abs(mae)
            else:
                mae = np.sum(np.abs(dys2i))/n2
                rmse = np.sqrt(np.sum(dys2i*dys2i)/n2)
            if iprt:
                #if self.iaml:
                #    nha = self.nsu_heav[i]
                #    print '%6d %6d %12.4f %12.4f'%(nha,n1,mae,rmse)
                #else:
                print('%6d %12.4f %12.4f %12.4f'%(n1u,mae,rmse,errmax))
                #print '     dys = ', np.squeeze(dys)
            maes.append(mae)
            rmses.append(rmse)
            errsmax.append(errmax)
        self.dys = dys
        self.errsmax = errsmax
        self.maes = np.array(maes)
        self.rmses = np.array(rmses)
        self.n1so = np.array(n1so, dtype=int)


    def run2(self, mks, nshuffle):
        """ shuffle the training set `nsfl times
        """
        if nshuffle > 1:
            maes = 0. # np.nanmean(_maes,axis=0) #sum(_maes,axis=0)/nshuffle
            rmses = 0. # np.sqrt( np.nanmean(_rmses**2, axis=0) )
        else:
            pass

        self.mmaes = []
        self.mrmses = []

