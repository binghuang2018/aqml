#!/usr/bin/env python

import contextlib
import multiprocessing
import numpy as np
import scipy.spatial.distance as ssd
import aqml.cheminfo.core as cc
import aqml.cheminfo.molecule.nbody as MB
import aqml.cheminfo.molecule.core as cmc
import matplotlib.pylab as plt

T, F = True, False

class RawM(object):
    """
    molecule object with only `zs & `coords
    """
    def __init__(self, mol):
        self.zs = mol.zs
        self.coords = mol.coords
        self.mol = mol

    def generate_coulomb_matrix(self,inorm=False,wz=False,rpower=1.0):
        """ Coulomb matrix
        You may consider using `cml1 instead of `cm """
        na = len(self.zs)
        mat = np.zeros((na,na))
        ds = ssd.squareform( ssd.pdist(self.coords) )
        np.fill_diagonal(ds, 1.0)
        if wz:
            X, Y = np.meshgrid(self.zs, self.zs)
            diag = -1. * np.array(self.zs)**2.4
        else:
            X, Y = [1., 1.]
            diag = np.zeros(na)
        mat = X*Y/ds**rpower
        np.fill_diagonal(mat, diag)
        L1s = np.linalg.norm(mat, ord=1, axis=0)
        ias = np.argsort(L1s)
        cm = L1s[ias] if inorm else mat[ias,:][:,ias].ravel()
        return cm

    def get_lbob(self, plcut=2, iconn=T, wz=F, rpower=1.):
        """
        get local bob, i.e., a bob vector per atom 
        """
        mc = cmc.RawMol(self.mol)
        o1 = MB.NBody(mc, g=mc.g, pls=mc.pls, iconn=iconn, plcut=plcut, bob=T)
        x = []
        for i in range(mc.na):
            bs = o1.get_bonds([i])
            for k in bs:
                v = bs[k]
                const = np.product( np.array(k.split('-'),dtype=float) ) if wz else 1.0
                v2 = list( const/np.array(v)**rpower )
                v2.sort()
                bs[k] = v2
            x.append( bs )
        return x


class RawMs(object):

    #def __init__(self, mols, repr='cml1', param={'inorm':T, 'wz':F, 'rpower':1.0}):
    def __init__(self, mols, repr='bob', param={'iconn':T, 'plcut':2, 'wz':F, 'rpower':1.}, debug=F):
        self.mols = mols
        self.debug = debug
        self.nm = len(mols.nas)
        self.repr = repr
        self.param = param

    @property
    def x(self):
        if not hasattr(self, '_x'):
            self._x = self.get_x()
        return self._x

    def get_x(self):
        xs = []
        wz = self.param['wz']
        rp = self.param['rpower']
        if self.repr in ['cml1']:
            inorm = self.param['inorm']
            for i in range(self.nm):
                mi = self.mols[i]
                rmol = RawM(mi)
                xi = rmol.generate_coulomb_matrix(inorm=inorm,wz=wz,rpower=rp)
                xs.append(xi)
        elif self.repr in ['bob']:
            iconn = self.param['iconn']
            plcut = self.param['plcut']
            for i in range(self.nm):
                mi = self.mols[i]
                rmol = RawM(mi)
                xi = rmol.get_lbob(plcut=plcut, wz=wz, rpower=rp)
                xs.append(xi)
        return xs

    @property
    def ds(self):
        if not hasattr(self, '_ds'):
            ims = np.arange(self.nm)
            self._ds = self.cdist(ims,ims)
        return self._ds

    def cdist_lbob(self, xi, xj): 
        """ calculate distance between two local BoB's of atoms i and j"""
        d = 0.
        ks = list(xi.keys()) + list(xj.keys())  
        for k in ks:
            vi = xi[k] if k in xi else []
            ni = len(vi)
            vj = xj[k] if k in xj else []
            nj = len(vj)
            n = max(len(vi), len(vj))
            vi2 = np.array( [0.]*(n-ni)+vi )
            vj2 = np.array( [0.]*(n-nj)+vj )
            d += np.sum( np.abs( vi2-vj2 ))
        return d
        
    def cdist(self, ims, jms, ncpu=None):
        """ pair-wise distance """
        ni = len(ims)
        nj = len(jms)
        iap = F # incude all atom pairs 
        if ni==nj:
            if np.all(ims==jms): iap = T
        pairs = []; apairs = []
        for i0 in range(ni): 
            for j0 in range(nj):
                i = ims[i0]
                j = jms[j0]
                if iap:
                    if j0>i0:
                        pairs.append([i,j])
                        apairs.append([i0,j0])
                else:
                    pairs.append([i,j])
                    apairs.append([i0,j0])
        print('pairs=', pairs)
        ds = np.zeros((ni,nj))
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=ncpu)
        dsr = pool.map(self.get_dij, pairs)
        for ip, pair in enumerate(apairs):
            i,j = pair
            if iap:
                ds[i,j] = ds[j,i] = dsr[ip]
            else:
                ds[i,j] = dsr[ip]
        return ds

    def get_dij(self, ij):
        i,j = ij
        xi = self.x[i]
        xj = self.x[j]
        zsi = self.mols[i].zs; ias = np.arange(len(zsi))
        zsj = self.mols[j].zs; jas = np.arange(len(zsj))
        zsu = list(set(zsi))
        if set(zsi)!=set(zsj):
            print(' ** elements differ!')
        dij = 0.
        for z in zsu:
            iasz = ias[z==zsi]
            jasz = jas[z==zsj]
            dsz = []
            nazi = len(iasz)
            nazj = len(jasz)
            for ii,ia in enumerate(iasz):
                dsi = []
                for jj,ja in enumerate(jasz):
                    daij = self.cdist_lbob( xi[ia], xj[ja] )
                    dsi.append(daij)
                di = np.min(dsi)
                dsz.append(di)
                jac = jasz[dsi.index(di)]
                if self.debug: print('z=',z, 'im,ia=',i,ia, 'jm,ja=',j,jac, 'd=',di)
            dz = np.max(dsz)
            dij = max(dij,dz)
        return dij

    def remove_redundant(self, thresh=0.03):
        idx = [0]
        for i in range(1,self.nm):
            if np.all(self.ds[i,idx] > thresh):
                idx.append(i)
        return idx

    def vb(self, i,ia, j,ja, keys=None, sigma=0.1, xlim=None, ylim=None):
        """ visualize bob """
        rs = np.linspace(0, 4.8, 1000)
        wz = self.param['wz']
        rp = self.param['rpower']
        ys = []
        ys0 = np.zeros(len(rs))
        
        xi = self.x[i][ia]
        xj = self.x[j][ja]
        if keys is None:
            keys = list( set(list(xi.keys())+list(xj.keys())) )
        colors = ['k', 'b', 'r', 'g']
        legends = []
        for ik, key in enumerate(keys):
            ysi = ys0.copy()
            const = np.product( np.array(key.split('-'),dtype=float) ) if wz else 1.0
            rsi = const/np.array(xi[key]) if rp==1. else (const/np.array(xi[key]))**(1./rp)
            for ir,ri in enumerate(rsi):
                ysi += np.exp( - 0.5 * (rs-ri)**2/sigma**2 ) * xi[key][ir]
            ysj = ys0.copy()
            rsj = const/np.array(xj[key]) if rp==1. else (const/np.array(xj[key]))**(1./rp)
            for jr,rj in enumerate(rsj):
                ysj += np.exp( - 0.5 * (rs-rj)**2/sigma**2 ) * xj[key][jr]
            #ys.append([ ysi,ysj ])
        
            plt.plot(rs, ysi, '-'+colors[ik], rs, ysj,'--'+colors[ik])
            legends += keys[ik:ik+1] + ['']
        plt.legend( legends )
        if xlim:
            plt.xlim(xlim[0],xlim[1])
        if ylim:
            plt.ylim(ylim[0],ylim[1])
        return plt


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

def printa(a, precision=2, suppress=T, *args):
    with printoptions(precision=precision, suppress=T):
        print(a)


if __name__ == "__main__":

    import os, sys, io2
    import argparse as ap

    args = sys.argv[1:]

    ps = ap.ArgumentParser()

    ps.add_argument('-plcut', nargs='?', default=3, type=int, help='path length cutoff')
    ps.add_argument('-rcut','--rcut', nargs='?', default=2.7, type=float, help='SLATM cutoff radius, default is 4.8 Ang')

    ps.add_argument('-rp', '-rpower', nargs='?', type=int, default=1, help='r^rpower in BoB / SLATM')
    ps.add_argument('-thresh', nargs='?', type=float, float=0.03, help='threshold distance between mols')
    ps.add_argument('-debug', action='store_true')
    ps.add_argument('-iconn', nargs='?', type=str, default='T')
    ps.add_argument('-i', dest='idxs', type=int, nargs='*')

    ag = ps.parse_args(args)
    ag.iconn = {'T':True, 'F':False}[ag.iconn]

    debug = ag.debug
    thresh = ag.thresh #0.03
    so = ''
    for i in ag.idxs:
        fsi = io2.cmdout('ls frag_%s*z'%i)
        ms1 = RawMs( cc.molecules(fsi), repr='bob', param={'iconn':ag.iconn, 'plcut':ag.plcut, 'wz':F, 'rpower':rp}, debug=debug )
        idx = ms1.remove_redundant(thresh)
        so += ' '.join([ fsi[j][:-4]+'*' for j in idx ])
        so += ' '
        #print('nm=', ms1.nm, 'f=',fs[idx[0]], 'idx=', idx) 
        #print('ds=')
        #printa(ms1.ds)
    print( so )

