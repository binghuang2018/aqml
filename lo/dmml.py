#!/usr/bin/env python

import numpy as np
from pyscf import lib, gto, scf, dft, cc, ao2mo
from cheminfo.core import *
import os, sys, io2
import scipy.spatial.distance as ssd
#import torch
from representation.xb import *
import cml.fkernels as qk
import cml.fdist as qd
from functools import reduce
#from qml.math import cho_solve

home = os.environ['HOME']
np.set_printoptions(precision=3,suppress=True)

T, F = True, False

UN = io2.Units()
h2e = UN.h2e
h2kc = UN.h2kc


class dmml(object):

    def __init__(self, xd): #, yd):
        self.__dict__ = xd.__dict__.copy()

    def init_YData(self,yd):
        self.yd = yd
        self.yobj = yd.yobj

    def krr(self, x1,y1,x2, kernel='g', icenter=F, c=1.0,l=1e-8):
        #kf = qk.gaussian_kernel if kernel[0] == 'g' else qk.laplacian_kernel
        g = T
        if kernel == 'g':
            df = qd.fl2_distance;
            #metric = 'euclidean';
            c2 = c/np.sqrt(2.0*np.log(2.0))
        else:
            g = F
            df = qd.manhattan_distance
            #metric = 'cityblock'
            c2 = c/np.log(2.0)
        #ds11 = ssd.squareform(ssd.pdist(x1,metric));
        ds11 = df(x1.T,x1.T); ds21 = df(x2.T,x1.T)
        dmax = np.max(ds11)
        #ratio = np.max(ds21)/dmax
        #print ' ----- max(ds11), max(ds21), ratio = ', np.max(ds11), np.max(ds21), ratio
        #if ratio > 1e2: c2 = ratio * 10
        sigma = c2 * dmax

        #print ' ** ds11 = ', ds11
        #print ' dmax = ', np.max(ds11)
        #print ' ** sigma = ', sigma
        #ds21 = ssd.cdist(x2,x1,metric);
        #print ' ** ds21 = ', ds21
        #k1 = kf(x1,x1,sigma); k2 = kf(x2,x1,sigma)
        if g:
            k1 = np.exp( -0.5 * ds11**2/sigma**2 )
            k2 = np.exp( -0.5 * ds21**2/sigma**2 )
        else:
            k1 = np.exp( -ds11/sigma )
            k2 = np.exp( -ds21/sigma )
        k1[np.diag_indices_from(k1)] += l

        #icenter=T #F
        if icenter:
            _y1c = np.mean(y1,axis=0)
        else:
            _y1c = np.zeros(y1.shape[1])
        y1c = _y1c[np.newaxis,...]
        alpha = np.linalg.solve(k1,y1-y1c); #print ' ** alpha = ', alpha #.shape
        #print ' ** size(k1), size(y1), size(a), size(k2) = ', k1.shape, y1.shape, alpha.shape, k2.shape
        #print ' ** k2 = ', k2
        y2_pred = np.dot(k2,alpha)
        return ds21, y2_pred + y1c

    def run_zbz(self, ims1, ims2, c=1., l=1e-8):
        """
        zbz: Z by Z, i.e., use (Z_I,Z_J) pair to select training & test set
        """
        zsu = self.xobj.zsu
        nzu = len(zsu)
        xs, ys = np.array(self.xobj.xsb), np.array(self.yobj.ys)
        #print ' xs, ys = ', xs.shape, ys.shape
        xlabels, ylabels = self.xobj.labels, self.yobj.labels; #print ' + xlabels = ', xlabels
        xims, yims = xlabels[:,0], ylabels[:,0]
        #print ' xims, yims = ', xims.shape, yims.shape
        ims1, ims2 = np.array(ims1, np.int), np.array(ims2,np.int)

        ys_pred = np.zeros( ys.shape ) # note that only the entries related to `ims2 need your concern

        nml = 0
        for zi in zsu:
            for zj in zsu:
                keys = [ [zi,zj] ]
                for opt in ['z','zz']:
                    if zi != zj and opt=='z': continue
                    print(' ** now working on %d-%d-%s'%(zi,zj,opt))
                    idxs_x1 = self.xobj.get_idx(keys, ims=ims1, opt=opt); #print 'idsx_x1 = ', idxs_x1.shape
                    idxs_x2 = self.xobj.get_idx(keys, ims=ims2, opt=opt)
                    idxs_y1 = self.xobj.get_idx(keys, ims=ims1, opt=opt, labels=ylabels)
                    idxs_y2 = self.xobj.get_idx(keys, ims=ims2, opt=opt, labels=ylabels)
                    #x1filt = np.any(xims[np.newaxis,...] == ims1[...,np.newaxis], axis=0); print 'x1filt = ', x1filt.shape
                    #x2filt = np.any(xims[np.newaxis,...] == ims2[...,np.newaxis], axis=0)
                    #y1filt = np.any(yims[np.newaxis,...] == ims1[...,np.newaxis], axis=0)
                    #y2filt = np.any(yims[np.newaxis,...] == ims2[...,np.newaxis], axis=0)
                    x1, x2 = xs[idxs_x1], xs[idxs_x2]
                    y1, y2 = ys[idxs_y1], ys[idxs_y2]

                    # get some LCs
                    percents = [0.1, 0.2, 0.4, 0.8, 1.] # [1.]
                    n1t = len(x1); ridxs = np.random.permutation(n1t)
                    for percent in percents:
                      n1 = int(n1t*percent)
                      if n1 <= 2: continue
                      _i = ridxs[:n1]
                      ds2, y2_est = self.krr(x1[_i],y1[_i],x2,c=c,l=l)
                      #print ' has NaN in x? ',
                      assert not np.any(np.isnan(xs))
                      #print ' y1 = ', y1[:,0]
                      #print ' y2 = ', y2[:,0]
                      ns2 = self.yobj.ns[idxs_y2]; nm2 = len(ns2); n2 = nm2 * ns2[0,2]
                      #print ', y2.shape = ', y2.shape
                      dy2 = np.abs(y2_est-y2)
                      print('  n1,  mae, rmse, delta_max = ', n1, np.sum(dy2)/n2, np.sqrt(np.sum(dy2**2)/n2), np.max(dy2))
                    ir2 = np.where( dy2==np.max(dy2) )[0][0]
                    print('    when reaching delta_max, we have')
                    print('                          entries: ', idxs_y2[ir2], ',  atom labels: ', ylabels[idxs_y2[ir2]])
                    dmin = np.min(ds2[ir2])
                    ic1 = np.where( ds2[ir2] == dmin )[0][0]
                    print('       closest training instances (dmin=%.4f): '%dmin, idxs_x1[ic1], ',  atom labels: ', xlabels[idxs_x1[ic1]])
                    nr1,nc1,nn1 = ns2[ic1]
                    nr2,nc2,nn2 = ns2[ir2]
                    print('         corresp. dm used for training: \n', y1[ic1][:nn1].reshape((nr1,nc1)))
                    print('                  corresp. dm for test: \n', y2[ir2][:nn2].reshape((nr2,nc2)))
                    ys_pred[ idxs_y2 ] = y2_est
                    nml += len(idxs_y2)
        self.ys_pred = ys_pred
        for im2 in ims2:
            print(' ** now test on mol %d'%(1+im2))
            yfilt = (yims==im2)
            _labels = ylabels[yfilt]
            _vs = ys_pred[yfilt]
            #print ' ++ nml, nml0 = ', nml, len(_vs)
            ib, ie = self.xobj.ias1[im2], self.xobj.ias2[im2]
            _zs, _coords = self.xobj.zs[ib:ie], self.xobj.coords[ib:ie]
            _obj = density_matrix(_zs, _coords, basis=self.yd.basis, meth=self.yd.meth, \
                                 spin=self.yd.spin, verbose=self.yd.verbose, iprt=self.yd.iprt)
            dm1_hao = _obj.reconstruct_dm( _labels, _vs )
            #dm1_hao = _obj.reconstruct_dm(_labels, self.yobj.ys[yfilt])
            #obj.calc_ca_dm(idx=idx, idx2=idx2)
            #print ' ### ', self.yobj.props
            props_r = self.yobj.props[im2]
            _obj.get_diff(dm=dm1_hao, props_r=props_r, hao=T)

    def run_aba(self, ims1, ims2, xs1=None, xs2=None, rot=None, c=1., l=1e-8):
        """
        aba: atom-by-atom, i.e., we treat one pair of atoms each time and choose
             bonds of similar type as training set.
        test: target bond, must be specified as a list/tuple of size 2
        """
        zsu = self.xobj.zsu
        nzu = len(zsu)
        xs, ys = np.array(self.xobj.xsb), np.array(self.yobj.ys)
        #print ' xs, ys = ', xs.shape, ys.shape
        xlabels, ylabels = self.xobj.labels, self.yobj.labels; #print ' + xlabels = ', xlabels
        xims, yims = xlabels[:,0], ylabels[:,0]
        #print ' xims, yims = ', xims.shape, yims.shape
        ims1, ims2 = np.array(ims1, np.int), np.array(ims2,np.int)

        ys_pred = np.zeros( ys.shape ) # note that only the entries related to `ims2 need your concern

        nml = 0
        for zi in zsu:
            for zj in zsu:
                keys = [ [zi,zj] ]
                for opt in ['z','zz']:
                    if zi != zj and opt=='z': continue
                    print(' ** now working on %d-%d-%s'%(zi,zj,opt))
                    idxs_x1 = self.xobj.get_idx(keys, ims=ims1, opt=opt); #print 'idsx_x1 = ', idxs_x1.shape
                    idxs_x2 = self.xobj.get_idx(keys, ims=ims2, opt=opt)
                    idxs_y1 = self.xobj.get_idx(keys, ims=ims1, opt=opt, labels=ylabels)
                    idxs_y2 = self.xobj.get_idx(keys, ims=ims2, opt=opt, labels=ylabels)
                    #x1filt = np.any(xims[np.newaxis,...] == ims1[...,np.newaxis], axis=0); print 'x1filt = ', x1filt.shape
                    #x2filt = np.any(xims[np.newaxis,...] == ims2[...,np.newaxis], axis=0)
                    #y1filt = np.any(yims[np.newaxis,...] == ims1[...,np.newaxis], axis=0)
                    #y2filt = np.any(yims[np.newaxis,...] == ims2[...,np.newaxis], axis=0)
                    x1, x2 = xs[idxs_x1], xs[idxs_x2]
                    y1, y2 = ys[idxs_y1], ys[idxs_y2]

                    # get some LCs
                    percents = [0.1, 0.2, 0.4, 0.8, 1.] # [1.]
                    n1t = len(x1); ridxs = np.random.permutation(n1t)
                    for percent in percents:
                      n1 = int(n1t*percent)
                      if n1 <= 2: continue
                      _i = ridxs[:n1]
                      ds2, y2_est = self.krr(x1[_i],y1[_i],x2,c=c,l=l)
                      #print ' has NaN in x? ',
                      assert not np.any(np.isnan(xs))
                      #print ' y1 = ', y1[:,0]
                      #print ' y2 = ', y2[:,0]
                      ns2 = self.yobj.ns[idxs_y2]; nm2 = len(ns2); n2 = nm2 * ns2[0,2]
                      #print ', y2.shape = ', y2.shape
                      dy2 = np.abs(y2_est-y2)
                      print('  n1,  mae, rmse, delta_max = ', n1, np.sum(dy2)/n2, np.sqrt(np.sum(dy2**2)/n2), np.max(dy2))
                    ir2 = np.where( dy2==np.max(dy2) )[0][0]
                    print('    when reaching delta_max, we have')
                    print('                          entries: ', idxs_y2[ir2], ',  atom labels: ', ylabels[idxs_y2[ir2]])
                    dmin = np.min(ds2[ir2])
                    ic1 = np.where( ds2[ir2] == dmin )[0][0]
                    print('       closest training instances (dmin=%.4f): '%dmin, idxs_x1[ic1], ',  atom labels: ', xlabels[idxs_x1[ic1]])
                    nr1,nc1,nn1 = ns2[ic1]
                    nr2,nc2,nn2 = ns2[ir2]
                    print('         corresp. dm used for training: \n', y1[ic1][:nn1].reshape((nr1,nc1)))
                    print('                  corresp. dm for test: \n', y2[ir2][:nn2].reshape((nr2,nc2)))
                    ys_pred[ idxs_y2 ] = y2_est
                    nml += len(idxs_y2)
        self.ys_pred = ys_pred
        for im2 in ims2:
            print(' ** now test on mol %d'%(1+im2))
            yfilt = (yims==im2)
            _labels = ylabels[yfilt]
            _vs = ys_pred[yfilt]
            #print ' ++ nml, nml0 = ', nml, len(_vs)
            ib, ie = self.xobj.ias1[im2], self.xobj.ias2[im2]
            _zs, _coords = self.xobj.zs[ib:ie], self.xobj.coords[ib:ie]
            _obj = density_matrix(_zs, _coords, basis=self.yd.basis, meth=self.yd.meth, \
                                 spin=self.yd.spin, verbose=self.yd.verbose, iprt=self.yd.iprt)
            dm1_hao = _obj.reconstruct_dm( _labels, _vs )
            #dm1_hao = _obj.reconstruct_dm(_labels, self.yobj.ys[yfilt])
            #obj.calc_ca_dm(idx=idx, idx2=idx2)
            #print ' ### ', self.yobj.props
            props_r = self.yobj.props[im2]
            _obj.get_diff(dm=dm1_hao, props_r=props_r, hao=T)


