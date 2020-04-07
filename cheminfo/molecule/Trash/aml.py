#!/usr/bin/env python

import os, sys, ase
import numpy as np
from aqml.cheminfo.rw.xyz import *
import itertools as itl
#import representation.fslatm as sl
from aqml.cheminfo.lo.dmx as sl

#np.set_printoptions(precision=4)
T,F = True,False

class slatm(object):

    def __init__(self, nas, zs, coords, wd=None, params={'racut':4.8}):

        if wd is None:
            self.wd = os.environ['PWD']
        else:
            _wd = wd[:-1] if wd[-1] == '/' else wd
            self.wd = _wd

        obj = sl.slatm(nas, zs, coords)
        obj.get_x(rb=F, param=param)

        self.zs = zs
        self.zsu = np.unique(zs).astype(np.int)
        self.nas = nas
        self.ias2 = np.cumsum(self.nas)
        self.ias1 = np.concatenate( ([0], self.ias2[:-1]) )
        self.nzs = obj.nzs
        self.coords = obj.coords

        params0 = {'local':True, 'nbody':3, 'dgrids': [0.04,0.04],  'sigmas':[0.05,0.05],\
                   'rcut':4.8, 'alchemy':False, 'iBoA':True, 'rpower2':6, 'coeffs':[1.], \
                   'rpower3': 3, 'ws':[1.,1./2,1./3], 'pbc':'000', 'kernel':'gaussian', \
                   'saves':[True,True,True], 'reuses':[True,True,True]}
        for key in params.keys():
            if params[key] != params0[key]:
                params0[key] = params[key]
        savex,saved,savek = params0['saves']
        reusex,reused,reusek = params0['reuses']
        self.coeffs = params0['coeffs']
        keys = ['local','nbody','dgrids','sigmas','rcut','alchemy','iBoA', \
                'rpower2','rpower3', 'ws','kernel','pbc']
        local,nbody,dgrids,sigmas,rcut,alchemy,iBoA,rpower2,rpower3,ws,kernel,pbc = \
                [ params0[key] for key in keys ]
        self.pbc = pbc
        if kernel in ['linear']:
            assert not local, '#ERROR: for linear kernel, consider using global repr only!'
        self.kernel = kernel
        self.srep = 'aslatm' if local else 'slatm'
        self.local = local
        if ck:
            self.get_mk(self.coeffs,savek=savek,reusek=reusek)


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


    def get_mk(self, local=True, coeffs=[1.], sigma=0.05, dgrid=0.03, \
              rcut=4.8, rpower2=6.0, rpower3=3.0, savek=False, reusek=False):
        """ molecular kernel """
        fk = self.wd + '/k_%s_%s.npz'%( self.kernel, self.srep )
        if os.path.isfile(fk) and reusek:
            _dt = np.load(fk)
            mk1 = _dt['mk1']; mk2 = _dt['mk2']
        else:
            zs, nas = self.zs, self.nas
            self.get_slatm_mbtypes()
            mbs1 = self.mbs1
            mbs2 = self.mbs2
            mbs3 = self.mbs3

            mk1,kwds = sl.fslatm.fget_mk1(local,nas,zs,coords,mbs1,mbs2,mbs3, \
                           rcut,dgrid,sigma, rpower2,rpower3, coeffs,kernel)
            mk2 = [ np.array([]) ]
            if self.n2 > 0:
                mk2 = sl.fslatm.fget_mk2(local,n1,nas,zs,coords,mbs1,mbs2,mbs3, \
                           rcut,dgrid,sigma, rpower2,rpower3, kwds,kernel)
            if savek: np.savez(fk, mk1=mk1, mk2=mk2)
        self.mk1 = mk1; self.mk2 = mk2


