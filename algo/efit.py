

import io2
import pandas as pd
import itertools as itl
import numpy as np

from lmfit import minimize, Parameters
from scipy.optimize import curve_fit

from cheminfo.core import *
import scipy.spatial.distance as ssd

import cheminfo.molecule.core as cmc
import numpy as np

T, F = True, False

def initb(_mols, iint, iecol=F):
    """
    initialization of all possible bonds (cov+vdw) in a water cluster

    vars
    =============================================================
    iint: id of interaction,
          iint=0 corresponds to a coarse grained model, in
                 which d(O-O) is the only degree of freedom
          iint=1, HB only,
          iint=2, intermolecular, all vdw bond
          iint=3, intramolecular, cov only
          iint=4, intramolecular, cov+vdw (i.e., OH single bond as well
                  as H---H vdw bond)
    """

    ints = [ 'coarse-grained', 'HB only', 'inter, vdw', 'intra, cov only', 'intra,cov+vdw']
    print('interaction type: ', ints[iint])

    iast2 = np.cumsum(_mols.nas)
    iast1 = np.concatenate(([0], iast2[:-1]))
    zsu = np.unique(_mols.zs)
    nz = len(zsu)
    bts = []
    for i in range(nz):
        for j in range(i,nz):
            zi, zj = zsu[i], zsu[j]
            bt = [zi,zj]
            if bt not in bts: bts.append(bt)
    nbt = len(bts)
    print('all possible bond types = ', bts)
    rs = []; nbs = []
    hb = set([1,8])
    for im,na in enumerate(_mols.nas):
        iab, iae = iast1[im], iast2[im]
        _coords = _mols.coords[iab:iae]
        _zs = _mols.zs[iab:iae]
        rawm = cmc.RawMol( (_zs, _coords) )
        rawm.connect()
        g,g2,pls = rawm.g, rawm.g2, rawm.pls
        #print(' molecule #%d, #num vdw bonds = %d'%(im+1, (g2>0).sum()/2) )
        ds_i = rawm.ds #ssd.squareform(ssd.pdist(coords_i))
        ias = np.arange(na)
        _rs = []
        _nbs = []
        for i in range(nz):
            for j in range(i,nz):
                zi,zj = zsu[i],zsu[j]
                rsij = 0. if iecol else []
                nbij = 0
                for ia in ias[zi==_zs]:
                    for ja in ias[zj==_zs]:
                        if zi==zj and ja<=ia: continue
                        if iint == 0:
                            # coarse-grained model, i.e., each h2o is an entity and
                            # he only var considered is d_OO
                            #print('##')
                            if not (zi==8 and zj==8): continue
                        elif iint == 1: # HB bond only
                            if set([zi,zj]) != hb or g2[ia,ja] == 0: continue
                        elif iint == 2:
                            # intermolecular, only between O in one water and H in another
                            if set([zi,zj]) != hb or g[ia,ja] > 0: continue
                        elif iint == 2.5:
                            # intermolecular, there is no path connecting ia,ja
                            if pls[ia,ja] > 0: continue # skip all bonds (cov or vdw) within any h2o
                        elif iint == 3: # intramolecular, cov only
                            if g[ia,ja] == 0: continue
                        elif iint == 4: # intramolecular, cov + vdw
                            if pls[ia,ja] <= 0: continue
                        else:
                            raise Exception('#ERROR: unknown interaction type')
                        if iecol: # calc 1/rij
                            rsij += 1/ds_i[ia,ja]
                        else:
                            nbij += 1
                            rsij += [ ds_i[ia,ja] ]
                _rs.append( rsij )
                _nbs.append(nbij)
        rs.append(_rs)
        nbs.append(_nbs)
    if iecol:
        rs2 = rs
        nbtype = len(rs)
    else:
        # now update rs
        nbs = np.array(nbs)
        ioks = np.any(nbs!=0, axis=0)
        bts1 = []
        for ibt,bt in enumerate(bts):
            if ioks[ibt]: bts1.append(bt)
        nbt1 = len(bts1)
        print('existing bond types = ', bts1)
        nmb2max = int(np.max(nbs))
        print('nmb2max=',nmb2max)
        rs2 = []
        for i in range(_mols.nm):
            _rs = []
            for j, rs_j in enumerate(rs[i]):
                if ioks[j]:
                    _rs = _rs + rs_j + [16.0]*(nmb2max-len(rs_j))
            rs2.append(_rs)
    return nbt1, np.array(rs2)


import cheminfo.molecule.nbody as nbd
import multiprocessing as mt

def get_mbts_i(ipt):
    """ get many-body terms for a single mol """
    mi, nbody, isub, param = ipt
    rpad, icn, iconn, iconj, icnb, ivdw = param
    rawm = cmc.RawMol(mi, ivdw=ivdw)
    g = rawm.g
    pls = rawm.pls
    #g2,pls = rawm.g2, rawm.pls
    #print('g=',g)
    ds_i = rawm.ds #ssd.squareform(ssd.pdist(coords_i))
    obj = nbd.NBody(mi, g=g, pls=pls, rpad=rpad, icn=icn, iconn=iconn, iconj=iconj, icnb=icnb)
    obj.get_all(nbody, isub=isub)
    return obj

def get_mbts_count_i(ipt):
    bts, nmbts, bts0 = ipt
    counts = np.zeros(len(bts0)).astype(int)
    for ib,bt in enumerate(bts):
        if bt in bts0:
            i = bts0.index(bt)
            counts[i] = nmbts[ib]
    return counts


class initb_g(object):

    def __init__(self, _mols, rpad=F, nbody=2, isub=T, icount=F, ival=T, ivdw=F, \
            icn=T, iconn=T, iconj=F, icnb=F, plmax4conj=3, nproc=1):
        """
        rpad:   if set to T, then when nbody=3, pad `angs matrix with
                [np.nan,np.nan,np.nan] if corresponding angle type is absent.
                Otherwise, pad with [np.nan]

        ival:   regroup `nobdy values from each mol in a db to form a matrix of
                size `nm x `nmb2max?
        icount: calculte counts of each type of atom, bond or angle
        isub  : if set to T, when nbody=3, nbody=1,2 will also be computed
        """
        iast2 = np.cumsum(_mols.nas)
        iast1 = np.concatenate(([0], iast2[:-1]))
        zsu = np.unique(_mols.zs)
        nz = len(zsu)
        nm = _mols.nm

        nmb2max = 0
        param = [rpad,icn,iconn,iconj,icnb,ivdw]
        if nproc > 1:
            pool = mt.Pool(processes=nproc)
            objs = pool.map(get_mbts_i, [ [_mols[i],nbody,isub,param] for i in range(nm) ])
        else:
            objs = []
            for im in range(nm):
                objs += [ get_mbts_i([_mols[im],nbody,isub,param]) ]

        cmbts1 = []; cmbts2 = []; cmbts3 = []; cmbts4 = [] # combination of mb terms for 2- (3-)body
        cmbvs1 = []; cmbvs2 = []; cmbvs3 = []; cmbvs4 = [] # comb of values of mb terms
        nmbts1 = []; nmbts2 = []; nmbts3 = []; nmbts4 = []
        _mbts1 = set(); _mbts2 = set(); _mbts3 = set(); _mbts4 = set()
        nmb2max, nmb3max, nmb4max = 0, 0, 0
        cmbtpsidx = [] # comb of topological idx
        for obj in objs:
            mbts1_i = obj.mbs1.keys()
            cmbts1.append( list(mbts1_i) )
            cmbvs1.append( [ obj.mbs1[k] for k in mbts1_i ] )
            nmbts1.append( [len(obj.mbs1[k]) for k in mbts1_i] )
            _mbts1.update( mbts1_i )

            mbts2_i = obj.mbs2.keys()
            cmbts2.append( list(mbts2_i) )
            cmbvs2.append( [ obj.mbs2[k] for k in mbts2_i ] )
            nmbts2.append( [ len(obj.mbs2[k]) for k in mbts2_i ] )
            cmbtpsidx.append( [ obj.tpsidx[k] for k in mbts2_i ] )
            _mbts2.update( mbts2_i )
            if nbody > 2:
                mbts3_i = obj.mbs3.keys()
                cmbts3.append( list(mbts3_i) )
                cmbvs3.append( [ obj.mbs3[k] for k in mbts3_i ] )
                nmbts3.append( [len(obj.mbs3[k]) for k in mbts3_i] )
                _mbts3.update( mbts3_i )

            if nbody > 3:
                mbts4_i = obj.mbs4.keys()
                n4i = len(mbts4_i)
                cmbts4.append( list(mbts4_i) )
                cmbvs4.append( [ obj.mbs4[k] for k in mbts4_i ] )
                if n4i == 0: # e.g., for CH4, no 4-body torsional terms
                    nmbts4.append( [0] )
                else:
                    nmbts4.append( [len(obj.mbs4[k]) for k in mbts4_i] )
                    _mbts4.update( mbts4_i )

        mbts1 = list(_mbts1); mbts1.sort(); nmbt1 = len(mbts1); counts1 = []
        mbts2 = list(_mbts2); mbts2.sort(); nmbt2 = len(mbts2); counts2 = []
        mbts3 = list(_mbts3); mbts3.sort(); nmbt3 = len(mbts3); counts3 = []
        mbts4 = list(_mbts4); mbts4.sort(); nmbt4 = len(mbts4); counts4 = []

        if icount:
            if nproc == 1:
                for im in range(nm):
                    counts1.append( get_mbts_count_i([cmbts1[im],nmbts1[im],mbts1]) )
                    counts2.append( get_mbts_count_i([cmbts2[im],nmbts2[im],mbts2]) )
            else:
                pool = mt.Pool(processes=nproc)
                counts1 = pool.map(get_mbts_count_i, [ [cmbts1[im],nmbts1[im],mbts1] for im in range(nm) ])
                counts2 = pool.map(get_mbts_count_i, [ [cmbts2[im],nmbts2[im],mbts2] for im in range(nm) ])
            self.counts1, self.counts2 = counts1, counts2
        if not ival:
            return

        # get maximal number of bonds
        #print('nmbts2=',nmbts2)
        nmb2max = int(np.max([ np.max(nsi) for nsi in nmbts2])) if len(nmbts2) > 0 else 0
        #print('nmbts3=',nmbts3)
        nmb3max = int(np.max([ np.max(nsi) for nsi in nmbts3])) if len(nmbts3) > 0 else 0
        #print('nmbts4=',nmbts4)
        nmb4max = int(np.max([ np.max(nsi) for nsi in nmbts4])) if len(nmbts4) > 0 else 0

        mbts2.sort() # e.g., ['1-6', '6-6']
        self.mbts2 = mbts2

        mbts3.sort()
        mbts4.sort()
        rs = []; chis = [] #
        angs = []
        dangs = []
        for im in range(nm):
            rsi = []; chis_i = []
            mbts2_i = cmbts2[im]
            for bt in mbts2:
                if bt in mbts2_i:
                    ib = mbts2_i.index(bt)
                    #print('nmb2max, cmbvs2[ib] = ', nmb2max, cmbvs2[im][ib] )
                    npad = nmb2max-len(cmbvs2[im][ib])
                    rsi = rsi + cmbvs2[im][ib] + [24.]*npad
                    chis_i = chis_i + cmbtpsidx[im][ib] + [0]*npad
                else:
                    rsi += [24.]*nmb2max
            rs.append(rsi)
            chis.append(chis_i)

            angsi = []
            if nbody > 2:
                mbts3_i = cmbts3[im]
                pad = [np.nan, np.nan, np.nan] if rpad else np.nan
                for at in mbts3:
                    if at in mbts3_i:
                        ia = mbts3_i.index(at)
                        #print('nmb2max, cmbvs2[ib] = ', nmb2max, cmbvs2[im][ib] )
                        angsi = angsi + cmbvs3[im][ia] + [pad]*(nmb3max-len(cmbvs3[im][ia]))
                    else:
                        angsi += [pad]*nmb3max
                angs.append(angsi)

            dangsi = []
            if nbody > 3:
                mbts4_i = cmbts4[im]
                pad = [np.nan,]*6 if rpad else np.nan
                for dat in mbts4:
                    if dat in mbts4_i:
                        ida = mbts4_i.index(dat)
                        dangsi = dangsi + cmbvs4[im][ida] + [pad]*(nmb4max-len(cmbvs4[im][ida]))
                    else:
                        dangsi += [pad]*nmb4max
                dangs.append(dangsi)

        print('bonds          : nbt,  bts =', len(mbts2), mbts2)
        print('angles         : nat,  ats =', len(mbts3), mbts3)
        print('dihedral_angles: ndat, dats=', len(mbts4), mbts4)
        self.rs, self.angs, self.dangs = np.array(rs), np.array(angs), np.array(dangs)
        self.chis = np.array(chis)
        #return mbts2, rs, mbts3, angs, mbts4, dangs


class fpoly(object):

    """ polynomial potential energy fitting using bond order as parameter

    Two body energy is expressed as :

      bo_IJ = exp( - R_{IJ} )
      E_2 = \sum_{i} \sum_j a^{(IJ)}_j * bo_IJ ^ j

    """

    def __init__(self, mols):
        self.mols = mols

    @staticmethod
    def eij(rij, cij):
        #t = 0.0 if rij > 16.0 else
        t = np.exp(-cij[1]*(rij-cij[2]))
        return cij[0]*(t*t - 2.0*t)


from scipy.optimize import differential_evolution


class fmorse(object):

    """ Morse curve fitting """

    def __init__(self, mols):
        self.mols = mols

    @staticmethod
    def eij(rij, cij):
        #t = 0.0 if rij >= 12.0 else
        t = np.exp(-cij[1]*(rij-cij[2]))
        return cij[0]*(t*t - 2.0*t)

    @staticmethod
    def eij_ext(r, c):
        #t = 0.0 if r >= 12.0 else
        t = np.exp(-c[1] * r**c[2] + c[3])
        return c[0]*(t*t - 2.0*t)

    def morse_ext(self, p0):
        """
        Extended Morse potential
           n = exp( - a * r^b + c )
           V(r) = U0 * [ n*n - 2*n ]
        Note that U0 here is likely to be possitive!!
        """
        n = 4 # number of param for each morse pot
        nbt = int(len(p0)/n) #self.N
        n1 = len(self._x)
        nav = int(len(self._x[0])/nbt)
        p = p0.reshape( (nbt,n) )
        x1 = self._x.reshape((n1,nbt,nav))
        es = []
        for i in range(n1):
            ei = 0.
            for j in range(nbt):
                xij = x1[i,j,:]
                u, a, b, c = p[j]
                ei += np.sum( u * ( np.exp( -a * xij[xij<12.0]**b + c ) - 1.)**2 - 1. )
            es.append( ei )
        return np.array(es)

    def morse_g(self, p0):
        """
        morse potential
        """
        n = 3 # number of param for each morse pot
        nbt = int(len(p0)/n) #self.N
        n1 = len(self._x)
        nav = int(len(self._x[0])/nbt)
        p = p0.reshape( (nbt,n) )
        x1 = self._x.reshape((n1,nbt,nav))
        es = []
        for i in range(n1):
            ei = 0.
            for j in range(nbt):
                xij = x1[i,j]
                ei += np.sum( p[j,0] * ( np.exp( -p[j,1]*(xij[xij<12.0] - p[j,2]) ) - 1.)**2 - 1. )
            es.append(ei)
        return np.array(es)

    def vec_loss_morse_ext(self, p0): # vectorized comp
        """
        Use with caution. It seems to be problematic for dealing with r>=12.0 cases.
        I.e., when fitting Morse pot, V(r>=12.0) may still be non-negaligible
        """
        n = 4 # parameter
        x1, y1 = self._x, self._y
        n1 = len(x1); nbt = self.N; nav = int(len(x1[0])/n)
        p = p0.reshape( (self.N,4) )
        u0, a, b, c = p[:,0], p[:,1], p[:,2], p[:,3]
        bos = np.exp(-a[np.newaxis, ..., np.newaxis] * x1.reshape((n1,nbt,nav))**b[np.newaxis,...,np.newaxis] + \
                       c[np.newaxis,...,np.newaxis])
        return np.sum( (np.sum( -u0[np.newaxis,...,np.newaxis] * (bos*bos - 2.*bos), axis=(1,2)) - y1)**2 )

    """
    Note that the curve_fit or lstsqfit in scipy requires that:

          The model function, f(x, ...). It must take the independent
          variable as the first argument and the parameters to fit
          as separate remaining arguments.
    """

    def prepare_vars(self, _idx1, N, xs):
        idx = np.arange(self.mols.nm)
        if isinstance(_idx1, int):
            idx1 = idx[:_idx1]
        elif isinstance(_idx1, (tuple,list,np.ndarray)):
            idx1 = _idx1
        n1 = len(idx1)
        n2 = self.mols.nm - n1
        idx2 = np.setdiff1d(idx,idx1)
        ys1, ys2 = self.mols.ys[idx1], self.mols.ys[idx2]
        #ms1 = self.mols._slice(idx1); ms2 = self.mols._slice(idx2)
        xs1 = xs[idx1]
        xs2 = xs[idx2] # []
        #if n1 < self.mols.nm:
        #    xs2 = xs[n1:]
        self.n1 = n1
        self.n2 = n2
        self.N = N
        self.xs1 = xs1
        self.ys1 = ys1
        self.xs2 = xs2
        self.ys2 = ys2

        # for regressor
        self._param = [ -60., 2.0, 1.2 ] * N
        self._lower = [ -300., 0.001, 0.5 ] * N
        self._upper = [ -1.,   9.0,   5.0 ] * N
        self._param_ext = [ 60., 2.0, 1.2 ] * N
        self._lower_ext = [ 1., 0.001, 0.001, 0.5 ] * N
        self._upper_ext = [ 300., 10.0,  10.0, 10.0 ] * N


    def get_errors(self, dys, string=''):
        mae, rmse, errmax = np.mean(np.abs(dys)), np.sqrt(np.mean(dys**2)), np.max(np.abs(dys))
        print(' %s: n=%6d, mae=%12.4f, rmse=%12.4f, errmax=%12.4f'%(string, len(dys), mae,rmse,errmax))
        return mae, rmse, errmax


    def regressor_ga(self, lp=2, nproc=1, seed=3, itest=T, iext=F):
        """ fit using GA """
        if iext:
            fun = self.morse_ext
            bounds = tuple(zip(self._lower_ext, self._upper_ext))
        else:
            fun = self.morse_g
            bounds = tuple(zip(self._lower, self._upper))
        self._x = self.xs1.copy()
        self._y = self.ys1.copy()
        def loss(p):
            return np.sum( (self.ys1 - fun(p))**lp )

        updating = 'deferred' if nproc > 1 else 'immediate'
        res = differential_evolution(loss, bounds, seed=seed, updating=updating, workers=nproc)
        param = res.x
        dys1 = fun(param) - self.ys1
        mae1, rmse1, errmax1 = self.get_errors(dys1, 'training')
        if itest:
            self._x = self.xs2.copy()
            dys2 = fun(param) - self.ys2
            mae2, rmse2, errmax2 = self.get_errors(dys2, ' test')
        return param


    def morse_lsq(self, x1, y1, p0):
        """
        morse potential for lsq fit
        """
        n = 3 # number of param for each morse pot
        nbt = int(len(p0)/n) #self.N
        n1 = len(x1)
        nav = int(len(x1[0])/nbt)
        p = p0.reshape( (nbt,n) )
        x1 = x1.reshape((n1,nbt,nav))
        es = []
        for i in range(n1):
            ei = 0.
            for j in range(nbt):
                xij = x1[i,j]
                ei += np.sum( p[j,0] * ( np.exp( -p[j,1]*(xij[xij<12.0] - p[j,2]) ) - 1.)**2 - 1. )
            es.append(ei)
        return np.sum( (np.array(es) - y1)**2 )

    def regressor_lsq(self, itest=T):
        """ least square fit by scipy """
        param, _ = curve_fit(self.morse_lsq, self.xs1, self.ys1, p0=self._param, \
                              bounds=(self._lower, self._upper))
        self._x = self.xs1.copy()
        dys1 = self.morse_g(param) - self.ys1
        mae1, rmse1, errmax1 = self.get_errors(dys1, 'training')
        if itest:
            self._x = self.xs2.copy()
            dys2 = fmorse.morse_g(param) - self.ys2
            mae2, rmse2, errmax2 = self.get_errors(dys2, ' test')
        return param


    @staticmethod
    def objective_lm(params, x, y0, nbt):
        """
        nbt: number of bond types
        """
        a = np.array([params['a_%i' % (i+1)].value for i in range(nbt)]) #[..., np.newaxis]
        b = np.array([params['b_%i' % (i+1)].value for i in range(nbt)])[..., np.newaxis]
        c = np.array([params['c_%i' % (i+1)].value for i in range(nbt)])[..., np.newaxis]
        nm, nmb2max = x.shape
        nb = int(nmb2max/nbt)
        y1 = np.zeros(nm)
        for im in range(nm):
            xi = x[im].reshape((nbt,nb))
            y1[im] = np.sum( a * np.sum( (np.exp( - b * (xi - c)) -1)**2 -1, axis=1) )
            #y1[im] = np.sum( a*np.exp(-b*xi) - c*np.exp(-2.*b*xi) ) #, axis=1 )
        return y1 - y0


    def regressor_lm(self, itest=T, iprt=F, cs=None, cs3=None, \
                     params0=None, check_boundary=T):
        """ use lmfit module for regression """
        params = Parameters()
        varyc3 = T
        if cs is None:
            cs1, cs2 = None, None
        else:
            nn = len(cs)
            cs1 = None; cs2 = None
            if nn == 3:
                cs1, cs2, cs3 = cs
            elif nn == 2 and (cs3 is not None):
                cs1, cs2 = cs
                varyc3 = F
            else:
                raise Exception('#invalid comb of `cs and `cs3')
        cs_min = []
        cs_max = []
        for i in range(self.N):
            if cs1 is not None:
                c1min, c1max = 80, 300
                params.add('a_%i'%(i+1), value=cs1[i], min=c1min, max=c1max) # set init values here
            else:
                c1min, c1max = 2, 300
                params.add('a_%i'%(i+1), value=100, min=2, max=300) # set init values here
            if cs2 is not None:
                c2min, c2max = 0.1, 100 #25 #5
                params.add('b_%i'%(i+1), value=cs2[i], min=c2min, max=c2max)
            else:
                c2min, c2max = 0.1, 5
                params.add('b_%i'%(i+1), value=2.0, min=c2min, max=c2max)
            #c3min, c3max = 2, 300 #0.7, 2.7
            c3min, c3max = 0., 10 # 0.7, 2.7
            if cs3 is not None:
                params.add('c_%i'%(i+1), value=cs3[i], vary=varyc3, min=c3min, max=c3max)
            else:
                params.add('c_%i'%(i+1), value=1.6, min=c3min, max=c3max)
                #params.add('c_%i'%(i+1), value=100, min=c3min, max=c3max)
            cs_min.append( [c1min,c2min,c3min] )
            cs_max.append( [c1max,c2max,c3max] )
        cs_min, cs_max = [ np.array(csi) for csi in [cs_min,cs_max] ]
        if params0: # set params to input values
            params2 = params0
        else:
            result = minimize(fmorse.objective_lm, params, method='leastsq', args=(self.xs1,self.ys1,self.N))
            params2 = result.params
            csu = np.array([ [ params2['%s_%d'%(key,k+1)].value for key in ['a','b','c'] ] \
                                      for k in range(self.N)  ])
            # now check if optimized params are identical to boundary values
            if np.any(np.abs(csu-cs_min)<0.1) or np.any(np.abs(csu-cs_max)<0.1):
                if check_boundary:
                    raise Exception('#ERROR: some optimized param are boundary values?? Check params!!')
                else:
                    print(' ** warning: some optimized params reached boundary values!!')
        dys1 = fmorse.objective_lm(params2, self.xs1, self.ys1, self.N)
        self.dys1 = dys1
        mae1, rmse1 = np.mean(np.abs(dys1)), np.sqrt(np.mean(dys1**2))
        so = ' regress: n1,mae,rmse,errmax= %5d  %9.4f %9.4f %9.4f\n'%(self.n1,mae1,rmse1,np.max(np.abs(dys1)))
        if iprt:
            so1 = ''
            for dy1 in dys1: so1 += '%.2f '%dy1
            print('dys1=',so1)
        if itest:
            dys2 = fmorse.objective_lm(params2, self.xs2, self.ys2, self.N)
            self.dys2 = dys2
            mae2, rmse2 = np.mean(np.abs(dys2)), np.sqrt(np.mean(dys2**2))
            so += '   test: n2,mae,rmse,errmax= %5d %9.4f %9.4f %9.4f'%(self.n2,mae2,rmse2,np.max(np.abs(dys2)))
            if iprt:
                so2 = ''
                for dy2 in dys2: so2 += '%.2f '%dy2
                print('dys2=',so2)
        print(so)
        return params2


    def test(self, use_lmfit=T):
        #phh = [1.0, 0.75, 2.21]
        pho = [7.0, 2.20, 1.70]
        #poo = [2.0, 1.0, 2.65]
        #params = np.array([phh,pho,poo])
        params = np.array([pho])
        print(' Parameters to be referenced:')
        print( params )
        a, b, c = [ params[:,i] for i in range(3) ]

        iint = 1 # 2
        N, xs = initb(self.mols, iint)
        ys = fmorse.morse_g(xs, a, b, c, N) #+ 0.01*np.random.random(len(xs))
        print('is there NaN? ', np.any(np.array(ys)==np.nan))
        print('xs[0]=',xs[0][ xs[0]<12.0 ])
        print('ys=',ys[:5])
        #regress
        n1 = 9 # use 9 data for training

        self.N = N
        self.n1, self.n2 = n1, self.mols.nm-n1
        self.xs1 = xs[:n1]
        self.xs2 = xs[n1:]
        self.ys1, self.ys2 = ys[:n1], ys[n1:]
        if use_lmfit:
            params2 = self.regressor_lm(itest=T, iprt=F)
        else:
            params2 = self.regressor()
        return params2 #print(' Regressed arameters')
        #print(params2)



def ecol(x, *c):

    # unpack 1D array `x into 2D data
    #x = x0[0]
    #c = cs0[0]
    n = len(c)
    assert x.shape[1] == n + n*(n-1)/2
    cs = []
    for i in range(n):
        for j in range(i,n):
            cs.append(c[i]*c[j])
    es = np.array([ np.dot(cs, xi) for xi in x ])
    return es


def ecol_model(nsel, els=[1,6,7], qs=[0.09, 0.12, -0.26]):
    _els = []; _qs = []
    for i,ni in enumerate(nsel):
        _els += [els[i],]*ni
        _qs += [qs[i],]*ni
    na = len(_qs)
    nel = len(els)
    _els, _qs = np.array(_els), np.array(_qs)
    ias = np.arange(na)
    e = 0; rm1s = []
    for i in range(nel):
        for j in range(i,nel):
            #eij = 0
            rm1 = 0
            zi, zj = els[i], els[j]
            for ia in ias[zi==_els]:
                for ja in ias[zj==_els]:
                    if zi!=zj or (zi==zj and ia > ja):
                        rij = 1.0
                        rm1 += 1/rij
                        e += _qs[ia]*_qs[ja]/rij
            #es.append(e) #ij)
            rm1s.append(rm1)
    return rm1s, e

def test_ecol():
    nes = [ [2,0,1], [2,1,1], [4,2,0], [2,1,2], [2,0,0]]
    rm1s, es = [],[]
    for nes_i in nes:
        o1, o2 = test_ecol(nes_i)
        rm1s.append(o1)
        es.append(o2)
    rm1s = np.array(rm1s)
    es = np.array(es) + 0.0001*np.random.rand(len(es))
    # H, C, O
    param0 = [0.06, 0.15, -0.1]
    param, _ = curve_fit(fit_ecol, rm1s, es, p0=param0)
    return param



def regress(mols, n1s, iint, pn=None, ys=None, iprt=F):
    """
    pn: property name
    """
    if isinstance(n1s,(int,np.int)):
        n1s = [n1s]

    errs = [] # 4 entries for each `n1, i.e., mae,rmse for training & test
    for n1 in n1s:
        idx = np.arange(mols.nm)
        n2 = mols.nm - n1
        idx1 = idx[:n1]; idx2 = idx[n1:]
        ys0 = mols.props[pn] if ys is None else ys
        ys1, ys2 = ys0[idx1], ys0[idx2]
        ms1 = mols._slice(idx1); ms2 = mols._slice(idx2)
        n, rs1 = init_b(ms1, iint)
        #print('rs1[0]=',rs1[0])
        _n, rs2 = init_b(ms2, iint)
        assert n == _n
        print('num bond types =', n)
        params = Parameters()
        for i in range(n):
            params.add('a_%i'%(i+1), value=10., min=0.0, max=100) # set init values here
            params.add('b_%i'%(i+1), value=1.0, min=0.0, max=100)
            params.add('c_%i'%(i+1), value=2.0, min=0.0, max=15.)

        result = minimize(objective, params, args=(rs1,ys1,n))
        dys1 = objective(result.params, rs1, ys1, n)
        mae1, rmse1 = np.mean(np.abs(dys1)), np.sqrt(np.mean(dys1**2))
        if iprt:
            so1 = ''
            for dy1 in dys1: so1 += '%.2f '%dy1
            print('dys1=',so1)

        dys2 = objective(result.params, rs2, ys2, n)
        mae2, rmse2 = np.mean(np.abs(dys2)), np.sqrt(np.mean(dys2**2))
        if iprt:
            so2 = ''
            for dy2 in dys2: so2 += '%.2f '%dy2
            print('dys2=',so2)
        errs.append( [mae1, rmse1, mae2, rmse2] )
    for i,n1 in enumerate(n1s):
        e1,e2,e3,e4 = errs[i]
        print(' %5d  %9.4f %9.4f  %9.4f %9.4f'%(n1,e1,e2,e3,e4))

    return rs2, ys2, n, result.params

