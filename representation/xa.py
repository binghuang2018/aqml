#!/usr/bin/env python

"""
generated SLATM for atom only
(use the relevant module in qmlcode)
"""

import os, sys, ase
import numpy as np
import scipy.spatial.distance as ssd
from scipy.interpolate import CubicSpline
from cheminfo.rw.xyz import *
from qml.representations import generate_slatm
import qml.distance as qd
import qml.kernels as qk
from qml.math import cho_solve
from qml.kernels import gaussian_kernel, get_local_kernels_gaussian

import itertools as itl

np.set_printoptions(precision=4)


T,F = True,False


class slatm(object):

    def __init__(self, obj, cx=F, ck=F, wd=None, label='', param={}):

        if wd is None:
            self.wd = os.environ['PWD']
        else:
            _wd = wd[:-1] if wd[-1] == '/' else wd
            self.wd = _wd
        self.label = label

        self.zs = obj.zs
        self.zsu = np.unique(self.zs).astype(np.int)
        self.nas = obj.nas
        self.nas1 = obj.nas1
        self.n1 = len(self.nas1)
        self.nas2 = obj.nas2
        self.n2 = len(self.nas2)
        self.ias2 = np.cumsum(self.nas)
        self.ias1 = np.concatenate( ([0], self.ias2[:-1]) )
        self.nat1 = obj.nat1
        self.nat2 = obj.nat2
        self.nzs = obj.nzs
        self.coords = obj.coords

        _param= {'local':True, 'nbody':3, 'dgrids': [0.04,0.04],  'sigmas':[0.05,0.05],\
                   'rcut':4.8, 'alchemy':False, 'iBoA':True, 'rpower2':6, 'coeffs':[1.], \
                   'rpower3': 3, 'ws':[1.,1.,1.], 'pbc':'000', 'kernel':'gaussian', \
                   'saves':[True,True,True], 'reuses':[True,True,True]}
        for key in list(param.keys()):
            if _param[key] != param[key]:
                _param[key] = param[key]
        savex,saved,savek = _param['saves']
        reusex,reused,reusek = _param['reuses']
        self.coeffs = _param['coeffs']
        keys = ['local','nbody','dgrids','sigmas','rcut','alchemy','iBoA', \
                'rpower2','rpower3', 'ws','kernel','pbc']
        local,nbody,dgrids,sigmas,rcut,alchemy,iBoA,rpower2,rpower3,ws,kernel,pbc = \
                [ _param[key] for key in keys ]
        self.pbc = pbc
        if kernel in ['linear']:
            assert not local, '#ERROR: for linear kernel, consider using global repr only!'
        self.kernel = kernel
        self.srep = 'aslatm' if local else 'slatm'
        self.local = local
        if ck: cx = T
        if cx:
            self.get_x(sigmas=sigmas, dgrids=dgrids, rcut=rcut, \
                      rpower2=rpower2, rpower3=rpower3, savex=savex,reusex=reusex)
        if ck:
            self.get_dmax(saved=saved,reused=reused)
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
        self.mbtypes = boas + bops + bots
        nsx = np.array([len(mb) for mb in [boas,bops,bots]],np.int)
        ins2 = np.cumsum(nsx)
        self.ins1 = np.array([0,ins2[0],ins2[1]],np.int)
        self.ins2 = ins2
        self.nsx = nsx

    def get_x(self, sigmas=[0.05,0.05], dgrids=[0.04,0.04], \
              rcut=4.8, rpower2=6.0, rpower3=3.0, savex=False, reusex=False):
        """ generate (a)SLATM representation """
        npz = self.wd+'/x.npz'
        if os.path.isfile(npz) and reusex:
            _dt = np.load(npz)
            self.x = _dt['x']
        else:
            self.get_slatm_mbtypes()
            x = []
            for i in range(len(self.nas)):
                ib, ie = self.ias1[i], self.ias2[i]
                _zs = self.zs[ib:ie]
                _coords = self.coords[ib:ie]
                xi = generate_slatm(_coords, _zs, self.mbtypes, unit_cell=None,
                                    local=self.local, sigmas=sigmas, dgrids=dgrids,
                                    rcut=rcut, alchemy=False, pbc=self.pbc, rpower=rpower2)
                x.append(xi)
            x = np.concatenate(x) if self.local else np.array(x)
            ########### IMPORTANT! Multiply `X by dgrid
            _dgrid = dgrids[0]
            if self.kernel in ['g','gaussian']:
                _dgrid = np.sqrt(dgrids[0])
            n1 = self.nsx[0] # note that there are `n1 unique Z's
            x[:,n1:] *= _dgrid
            ###########"""
            self.x = x
            if savex: np.savez(npz, x=self.x)

    def get_idx1(self, im, ia):
        """ get idx of atom in `zs """
        return self.ias1[i]+ia

    def get_idx2(self, im, ia):
        """ get idx of atom in `zs """
        return self.ias2[i]+ia

    def get_ds(self):
        """ calc (a)SLATM distance between atoms/molecules """
        if self.local:
            x1 = self.x[:self.nat1]; x2 = self.x[-self.nat2:]
        else:
            x1 = self.x[:self.n1]; x2 = self.x[-self.n2:]
        ds1 = qd.l2_distance(x1, x1)
        ds2 = qd.l2_distance(x2, x1)
        return ds1, ds2

    def get_dmax(self, saved=False, reused=False):
        """calc `dmax between aSLATM of two atoms"""
        fdmax = self.wd+'/%s_dmax_%s_%s.txt'%(self.label,self.kernel, self.srep)
        x = self.x[:self.nat1]
        zs = self.zs[:self.nat1]
        if os.path.exists(fdmax) and reused:
            dmax = eval( file(fdmax).readlines()[0] )
        else:
            if self.local:
                dsmax = []
                _zsu = list(self.zsu); Nz = len(_zsu)
                if Nz == 2:
                  filt1 = (zs == _zsu[0]); filt2 = (zs == _zsu[1])
                  ds = qd.l2_distance(x[filt1,:],x[filt2,:])
                  dmax_i = np.max(ds); dsmax.append( dmax_i )
                else:
                  for i in range(Nz-1):
                    # `i starts from 1 instead of 0 (i.e., 'H' atom) due to that
                    # d(H,X) << d(X,X'), where X stands for any heavy atom
                    for j in range(i+1,Nz):
                        filt1 = (zs == _zsu[i]); filt2 = (zs == _zsu[j])
                        ds = qd.l2_distance(x[filt1],x[filt2])
                        dmax_i = np.max(ds); dsmax.append( dmax_i )
                dmax = max(dsmax)

            else:
                if self.kernel in ['gaussian','g',]:
                    ds = np.max( qd.l2_distance(x,x) )
                    self.ds = ds
                    dmax = np.max(ds)
                elif self.kernel in ['l','laplacian']:
                    ds = np.max( qd.manhattan_distance(x,x) )
                    self.ds = ds
                    dmax = np.max(ds)
                else:
                    dmax = np.inf
            if saved: open(fdmax,'w').write('%.8f'%dmax)
        self.dmax = dmax

    def get_mk(self, coeffs=[1.,], savek=False, reusek=False):
        """ molecular kernel """
        fk = self.wd + '/%s_k_%s_%s.npz'%(self.label, self.kernel, self.srep )
        if os.path.isfile(fk) and reusek:
            _dt = np.load(fk)
            mk1 = _dt['mk1']; mk2 = _dt['mk2']
        else:
            zs, nas = self.zs, self.nas
            if self.local:
                x1 = self.x[:self.nat1]; x2 = self.x[-self.nat2:]
            else:
                x1 = self.x[:self.n1]; x2 = self.x[-self.n2:]
            if self.kernel in ['gaussian','g', 'laplacian','l']:
                if self.kernel[0]=='g':
                    _coeffs = np.array(coeffs)/np.sqrt(2.0*np.log(2.0))
                else:
                    _coeffs = np.array(coeffs)/np.log(2.0)
                sigmas = self.dmax * _coeffs
                if self.local:
                    func = qk.get_local_kernels_gaussian if self.kernel[0] == 'g' else qk.get_local_kernels_laplacian
                    mk1 = func(x1,x1,self.nas1,self.nas1,sigmas)
                    if len(self.nas2)> 0:
                        mk2 = func(x2,x1,self.nas2,self.nas1,sigmas)
                    else:
                        mk2 = [ np.array([]) ] * len(sigmas)
                else:
                    func = qk.gaussian_kernel if self.kernel[0] == 'g' else qk.laplacian_kernel
                    mk1 = [ func(x1,x1,sigma) for sigma in sigmas ]
                    if len(self.nas2)> 0:
                        mk2 = [ func(x2,x1,sigma) for sigma in sigmas ]
                    else:
                        mk2 = [ np.array([]) ] * len(sigmas)
            elif self.kernel in ['linear',]: # global  repr
                mk1 = qk.linear_kernel(x1,x1)
                if len(self.nas2)> 0:
                    mk2 = qk.linear_kernel(x2,x1)
                else:
                    mk2 = np.array([])
            else:
                raise '#ERROR: not implemented yet'
            if savek: np.savez(fk, mk1=mk1, mk2=mk2)
        self.mk1 = mk1; self.mk2 = mk2


