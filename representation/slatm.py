#!/usr/bin/env python

import os, sys, ase
import numpy as np
from cheminfo.rw.xyz import *
import itertools as itl
import cml.fslatm as sl

np.set_printoptions(precision=4)
T,F = True,False

class slatm(object):

    def __init__(self, obj, ck=False, wd=None, param={}, use_query_dmax=F):
        """
        ready mol info
        """
        if wd is None:
            self.wd = os.environ['PWD']
        else:
            _wd = wd[:-1] if wd[-1] == '/' else wd
            self.wd = _wd
        self.zs = obj.zs
        self.zsu = np.unique(self.zs).astype(np.int)
        self.nas = obj.nas
        self.nas1 = obj.nas1
        self.n1 = len(self.nas1)
        self.nas2 = obj.nas2

        self.ias2 = np.cumsum(self.nas)
        self.ias1 = np.concatenate( ([0], self.ias2[:-1]) )
        self.nat1 = obj.nat1
        self.nat2 = obj.nat2
        self.nat = self.nat1+self.nat2
        self.nzs = obj.nzs
        self.coords = obj.coords
        if ck:
            self.get_mk(param=param, use_query_dmax=use_query_dmax)

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

    def get_mk(self, param={}, use_query_dmax=F):
        """ molecular kernel """
        _param = {'local':T, 'nbody':3, 'dgrids': [0.04,0.04],  'sigmas':[0.05,0.05],\
                   'racut':4.8, 'rbcut':4.8, 'alchemy':False, 'iBoA':T, 'rpower2':6, \
                   'coeffs':[1.], 'rpower3': 3, 'ws':[1.,1./2,1./3], 'pbc':'000', \
                   'kernel':'gaussian', 'saves':[T,T,T], 'reuses':[T,T,T]}
        for key in list(param.keys()):
            if param[key] != _param[key]:
                _param[key] = param[key]
        savex,saved,savek = _param['saves']
        reusex,reused,reusek = _param['reuses']
        coeffs = _param['coeffs']
        keys = ['local','nbody','dgrids','sigmas','racut','rbcut','alchemy','iBoA', \
                'rpower2','rpower3', 'ws','kernel','pbc']
        local,nbody,dgrids,sigmas,racut,rbcut,alchemy,iBoA,rpower2,rpower3,ws,kernel,pbc = \
                [ _param[key] for key in keys ]
        rscut = np.array([racut,rbcut])
        if kernel in ['linear']:
            assert not local, '#ERROR: for linear kernel, consider using global repr only!'
        srep = 'aslatm' if local else 'slatm'
        w1, w2, w3 = ws
        self.pbc = pbc
        na = self.nas.sum()

        fk = self.wd + '/k_%s_%s.npz'%(kernel, srep)
        if os.path.isfile(fk) and reusek:
            _dt = np.load(fk)
            mk1 = _dt['mk1']; mk2 = _dt['mk2']
        else:
            zs, nas = self.zs, self.nas
            self.get_slatm_mbtypes()
            mbs1 = self.mbs1
            mbs2 = self.mbs2
            mbs3 = self.mbs3

            if use_query_dmax:
                naq = self.nas[-1]; nas_q = np.array([naq],np.int)
                ib, ie = self.ias1[-1], self.ias2[-1]
                zs_q = self.zs[ib:ie]
                coords_q = self.coords[ib:ie]
                nbmax = 1
                _nbrs = - np.ones((naq,nbmax)).astype(np.int) #(na, nbmax)).astype(np.int)
                print(' * now dmax...')
                dmax = sl.fslatm.fget_dij_max(local, nas_q,zs_q,coords_q.T, _nbrs.T, \
                             mbs1,mbs2.T,mbs3.T, rscut,dgrids[0],sigmas[0], w2,rpower2, \
                             w3,rpower3)
                print(' * dmax = ', dmax)
            else:
                raise '#ERROR: not know how to get dmax'

            if kernel[0] == 'g':
                kwds = np.array(coeffs) * dmax/np.sqrt(2.*np.log(2.0))
            elif kernel[0] == 'l':
                kwds = np.array(coeffs) * dmax/np.log(2.)

            nas = self.nas; nas1 = nas[:self.n1]
            zs = self.zs; zs1 = zs[:self.nat1]
            coords = self.coords; coords1 = coords[:self.nat1]
            nbmax=1; _nbrs = - np.ones((self.nat1,nbmax)).astype(np.int)
            print(' ** now mk1')
            mk1 = sl.fslatm.fget_mk1(local,nas1,zs1,coords1.T,_nbrs.T,mbs1,mbs2.T,mbs3.T, \
                                rscut,dgrids[0],sigmas[0], w2,rpower2,w3,rpower3, kwds)
            print(' ** mk1 done')
            mk2 = [ np.array([]) ]
            if len(self.nas2) > 0:
                nbmax=1; _nbrs = - np.ones((self.nat,nbmax)).astype(np.int)
                print(' ** now mk2')
                mk2 = sl.fslatm.fget_mk2(local,self.n1,nas,zs,coords.T,_nbrs.T,mbs1,mbs2.T,mbs3.T, \
                                rscut,dgrids[0],sigmas[0], w2,rpower2,w3,rpower3, kwds)
            if savek: np.savez(fk, mk1=mk1, mk2=mk2)
        self.mk1 = mk1; self.mk2 = mk2


