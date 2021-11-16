#!/usr/bin/env python

"""
representation will be generated for calc k
"""

import os, sys, ase
import cml.distance as cmld
import cml.kernels as cmlk
import numpy as np
import scipy.spatial.distance as ssd
from scipy.interpolate import CubicSpline
from aqml.cheminfo.rw.xyz import *
from cml.representations import generate_slatm
from math import ceil
import aqml.cheminfo.molecule.core as cmc
import itertools as itl

np.set_printoptions(precision=4)


T,F = True,False


class slatm(object):

    def __init__(self, obj, subjobs='a', navks=[-1,-1], nblksmax=[-1,-1], saveblk=T, \
              wxo=F, ck=F, wd=None, label='', param={}, itarget=F, use_chunks=[F], \
              icg=F, izeff=F, forbid_w=F, cab=F, verbose=F):
        """
        vars
        =============================
        navks -- [navk1,navk2], specifies the size of each sub-block of kernel
                 matrix for training and test set, respectively. I.e., navk1 by navk1
                 for k_11 (in between training set) and navk1 x navk2 for k_21 (between
                 train & test). The number of blocks is [nblk1,nblk2], which is equal to
                 [ceil(nm1/navk1), ceil(nm2/navk2)], for train & test set, respectively.
        nblksmax -- constraint on the value of `nblk1 & `nblk2, such that a subset of
                    calculations instead of all, can be carried out seperately.
        wxo -- write x only? T/F
        forbid_w -- forbid writing x or k files? T/F

        cab: calculate similarity (distance) between dissimilar atom
             types (type=element). Must be either T or F
        icg: use connectivity graph (including vdw bonds, conj bonds)
             By conj bonds, we mean that all conjugated atoms are
             `connected'. I.e., suppose atoms idx=[1,2,3,4]
             are conjugated, then cg[a,b] = 1 for any `a and `b in `idx
             even though g[a,b] = 0, where g is the molecular graph, where
             cg stands for the ``conjugated graph''.
        izeff: use effective nuclear charge (Z*) instead of Z as prefactor
               in SLATM 2- and 3-body
        """
        self.cab = cab
        self.izeff = izeff
        self.icg = icg
        self.verbose = verbose
        self.zmax = max(obj.zs)
        if wd is None:
            self.wd = os.environ['PWD']
        else:
            _wd = wd[:-1] if wd[-1] == '/' else wd
            self.wd = _wd

        self.obj = obj
        self.wxo = wxo # write x only?
        self.navks = navks
        self.forbid_w = forbid_w

        self.saveblk = saveblk # save subblocks of k? should be used together with `navks

        n1 = len(obj.nas1)
        n2 = len(obj.nas2)
        self.n1, self.n2 = n1, n2
        self.ias2 = np.cumsum(obj.nas)
        self.ias1 = np.concatenate( ([0], self.ias2[:-1]) )
        ims1 = np.arange(n1)
        ims2 = np.arange(n1,n1+n2)

        _param= {'local':True, 'nbody':3, 'dgrids': [0.04,0.04],  'widths':[0.05,0.05],\
                   'rcut':4.8, 'alchemy':False, 'iBoA':True, 'rpower2':6, 'coeffs':[1.], \
                   'isqrt_grid': T, 'rpower3': 3, 'ws':[1.,1.,1.], 'pbc':'000', \
                   'kernel':'gaussian', 'saves':[F,F,F], 'reuses':[F,F,F]}
        for key in list(param.keys()):
            if _param[key] != param[key]:
                _param[key] = param[key]


        saves = _param['saves']
        for i,key in enumerate(['savex','saved','savek']):
            _param[key] = saves[i]

        reuses = _param['reuses']
        for i,key in enumerate(['reusex','reused','reusek']):
            _param[key] = reuses[i]

        self.label = label if label == '' else label+'_'
        self.param = _param

        self.coeffs = _param['coeffs']
        ncoeff = len(self.coeffs)
        self.ncoeff = ncoeff
        keys = ['local','nbody','dgrids','widths','rcut','alchemy','iBoA', \
                'rpower2','rpower3', 'ws','kernel','pbc']
        local,nbody,dgrids,widths,rcut,alchemy,iBoA,rpower2,rpower3,ws,kernel,pbc = \
                [ _param[key] for key in keys ]
        self.pbc = pbc
        if _param['kernel'] in ['linear']:
            assert not local, '#ERROR: for linear kernel, consider using global repr only!'
        self.kernel = kernel
        self.srep = 'aslatm' if local else 'slatm'
        self.local = local
        self.use_chunks = use_chunks

        if self.kernel in ['gaussian','g',]:
            self.dfunc = cmld.l2_distance
            self.kfunc = cmlk.get_local_kernels_gaussian if local else qk.gaussian_kernels
            self._coeffs = np.array(self.coeffs)/np.sqrt(2.0*np.log(2.0))
        elif self.kernel in ['l','laplacian']:
            self.dfunc = cmld.manhattan_distance
            self.kfunc = cmlk.get_local_kernels_laplacian if local else qk.laplacian_kernels
            self._coeffs = np.array(self.coeffs)/np.log(2.0)
        else:
            raise Exception('#ERROR: not supported!')

        if ck:
            nav1, nav2 = n1, n2
            self.nav1, self.nav2 = nav1, nav2
            if use_chunks[0]:
                nav1, nav2 = use_chunks[1:]
                nblk1 = int(ceil(self.n1/nav1))
                nblk2 = int(ceil(self.n2/nav2))
                if self.verbose:
                    print('overall blocks: nblk1,nblk2=%d,%d'%(nblk1,nblk2))
                    print('each chunk consists of: nm1,nm2=%d,%d mols'%(nav1,nav2))
                self.nav1, self.nav2 = nav1, nav2
                ks1 = np.zeros((ncoeff,n1,n1))
                ks2 = np.zeros((ncoeff,n2,n1))

                nblk1max, nblk2max = nblksmax
                if nblk1max <= 0: nblk1max = nblk1
                if nblk2max <= 0: nblk2max = nblk2
                # get all jobs
                _jobsa = []
                for i in range(nblk1):
                    _jobsa.append( '11ii%03d%03d'%(i+1,i+1) )
                    for j in range(i+1,nblk1):
                        _jobsa.append( '11ij%03d%03d'%(i+1,j+1))
                    for j in range(nblk2):
                        _jobsa.append( '12ij%03d%03d'%(i+1,j+1))
                _jobsdic = {}
                for _job in _jobsa:
                    key = _job[:4]
                    if key in _jobsdic.keys():
                        _jobsdic[key] += [_job]
                    else:
                        _jobsdic[key] = [_job]
                if self.verbose:
                    print('_jobsdic=',_jobsdic)

                wko = F # write kernel file only
                if isinstance(subjobs, str):
                    jobs = _jobsa # subjobs in ['a','all']
                elif isinstance(subjobs,(tuple,list)):
                    if subjobs[0] in ['a','all']:
                        jobs = _jobsa
                    else:
                        assert subjobs[0][0] in ['0','1'], \
                            "#ERROR: `subjobs should be like ['11ii',...]"
                        wko = T
                        jobs = []
                        for subjob in subjobs:
                            if len(subjob) == 4:
                                jobs += _jobsdic[subjob]
                            else:
                                jobs += [subjob]
                else:
                    raise Exception('invalid input for `subjobs')

                # kernel(train_i,train_i)
                for i in range(nblk1):
                    if i >= nblk1max: continue
                    jobid = '11ii%03d%03d'%(i+1,i+1)
                    if jobid in jobs:
                        if verbose:
                            print(' --- jobid=',jobid)
                            print('\nkerenl(train_%d,train_%d)'%(i+1,i+1))
                        idx, k11 = self.get_kernel_block(ims1, ims1, i, i)#'ii')
                        if not wko:
                            ib1, ie1 = idx
                            ks1[:, ib1:ie1, ib1:ie1] = k11
                # kernel(train_i,train_j)
                for i in range(nblk1):
                    for j in range(i+1,nblk1):
                        if i >= nblk1max or j>=nblk1max: continue
                        jobid = '11ij%03d%03d'%(i+1,j+1)
                        if jobid in jobs:
                            if self.verbose:
                                print(' --- jobid=',jobid)
                                print('\nkerenl(train_%d,train_%d)'%(i+1,j+1))
                            idx, k21 = self.get_kernel_block(ims1, ims1, i, j)#'ji')
                            if not wko:
                                ib1, ie1, ib2, ie2 = idx
                                ks1[:, ib2:ie2, ib1:ie1] = k21
                                for ic in range(ncoeff):
                                    ks1[ic, ib1:ie1, ib2:ie2] = k21[ic].T
                # kernel(train, test)
                for i in range(nblk1):
                    for j in range(nblk2):
                        if i>=nblk1max or j>=nblk2max: continue
                        jobid = '12ij%03d%03d'%(i+1,j+1)
                        if jobid in jobs:
                            if self.verbose:
                                print(' --- jobid=',jobid)
                                print('\nkerenl(test_%d,train_%d)'%(i+1,j+1))
                            idx, k21 = self.get_kernel_block(ims1, ims2, i, j)#'ji')
                            if not wko:
                                ib1, ie1, ib2, ie2 = idx
                                ks2[:, ib2-n1:ie2-n1, ib1:ie1] = k21
            else:
                _, ks1 = self.get_kernel_block(ims1, ims1, 0, 0)#'ii')#'11')
                _, ks2 = self.get_kernel_block(ims1, ims2, 0, 0)#'ji')#'12')
            self.mk1, self.mk2 = ks1, ks2

    def get_x_block(self):
        """ get blocks of x """
        self._ims_blk = self._ims1
        self._label_x = self._label_x1
        self._aux_x = self._aux_x1
        self.x1, self.zs1 = self.get_x() #self._ims1)
        self.x2 = None
        if self.dist in ['ij','ji']:
            self._ims_blk = self._ims2
            self._label_x = self._label_x2
            self._aux_x = self._aux_x2
            self.x2, self.zs2 = self.get_x() #x_aux, x_label, _ims2)
        if not hasattr(self,'dsmax'):
            _label = 'c11'
            self.get_dsmax(_label)


    def get_kernel_block(self, ims1, ims2, i, j): # blocks):
        """
        kernel(block_i, block_j)

        Two indicators are needed to distinguish 3 unique cases:
        dist_type     db_idx
        ii            11           k(train_i, train_i)
        ij            11           k(train_i, train_j)
        ij            21           k(test_i, train_j)
        """
        nm1, nm2 = len(ims1), len(ims2)
        nav1 = self.nav1
        if (nm1==nm2) and np.all(ims1==ims2):
            dist, dbi, nav2 = 'ii', '11', self.nav1
        else:
            dist, dbi, nav2 = 'ij', '21', self.nav2
        self.dist = dist
        self.dbi = dbi # database idx, 1 or training set, 2 for test set
        ib1 = ims1[0] + i*nav1; ie1 = ims1[0] + (i+1)*nav1
        if ie1 > ims1[-1]+1: ie1 = ims1[-1]+1
        ib2 = ims2[0] + j*nav2; ie2 = ims2[0] + (j+1)*nav2
        if ie2 > ims2[-1]+1: ie2 = ims2[-1]+1
        _ims1 = np.arange(ib1,ie1)
        _ims2 = np.arange(ib2,ie2)
        idx = [ib1,ie1,ib2,ie2]
        #print(' ib1,ie1,ib2,ie2 = ', ib1,ie1,ib2,ie2 )
        #print(' _ims1 = ', _ims1)
        #print(' _ims2 = ', _ims2)

        # combination of sets: '11' -- training set, '12' training&test set
        if dbi == '11':
            if i == j:
                _ims2 = []
                idx = [ib1,ie1]
            else:
                self.dist = 'ij'

        if dbi == '11':
            _navk1, _navk2 = [self.navks[0],]*2
        else:
            _navk1, _navk2 = self.navks
        self._navk1, self._navk2 = _navk1, _navk2

        self._ims1, self._ims2 = _ims1, _ims2
        self._ims = np.concatenate((_ims1,_ims2))
        self._nas1, self._nas2 = self.obj.nas[_ims1], self.obj.nas[_ims2]
        zs1 = []; zs2 = []
        for im in _ims1: zs1 += list( self.obj.zs[self.ias1[im]:self.ias2[im]] )
        for im in _ims2: zs2 += list( self.obj.zs[self.ias1[im]:self.ias2[im]] )
        self._zs1 = np.array(zs1, dtype=int)
        self._zs2 = np.array(zs2, dtype=int)
        self._zsu = np.unique(zs1+zs2).astype(np.int)

        self._aux_x1 = ''
        self._label_x1 = '_c1_b%03d'%(i+1)
        #self.x1 = self.get_x(x_aux, x_label, _ims1)
        #self.x2 = None
        #print('distance type = ', self.dist)
        if self.dist in ['ij','ji']:
            self._aux_x2 = '' if dbi[0] == '1' else self.label
            self._label_x2 = '_c%s_b%03d'%(dbi[0],j+1)
            #self.x2 = self.get_x(x_aux, x_label, _ims2)
        _label_k = '_c%s_blk_%03d_%03d'%(self.dbi,i+1,j+1)
        k = None
        _label_db = '' if dbi == '11' else self.label
        if not self.wxo:
            self.fk = self.wd + '/%sk_%s_%s%s.npz'%(_label_db, self.kernel, self.srep, _label_k)
            if self.verbose:
                print('        kernel file: ', self.fk )
            if not os.path.exists(self.fk):
                self.get_x_block()
            k = self.get_kernel() #k_label)
        # release memory
        self.x1 = None
        self.x2 = None
        return idx, k


    def get_slatm_mbtypes(self):
        """ get slatm many-body types"""
        zsmax = np.unique(self.obj.zs)
        nzsmax = np.max(self.obj.nzs, axis=0)
        #print('zsmax=',zsmax, ' nzsmax=',nzsmax)
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
                        nzsi = [ (zj == np.array(tasi)).sum() for zj in zsmax ]
                        #print('nzsi=',nzsi, 'nzsmax=',nzsmax)
                        if np.all(nzsi <= nzsmax):
                            bots.append( tasi )
        self.mbtypes = boas + bops + bots
        nsx = np.array([len(mb) for mb in [boas,bops,bots]],np.int)
        ins2 = np.cumsum(nsx)
        self.ins1 = np.array([0,ins2[0],ins2[1]],np.int)
        self.ins2 = ins2
        self.nsx = nsx

    def get_x(self): #, _aux, _label, _ims): #, widths=[0.05,0.05], dgrids=[0.04,0.04], \
              #rcut=4.8, rpower2=6.0, rpower3=3.0, savex=False, reusex=False):
        """ generate (a)SLATM representation """

        widths, dgrids, rcut, rpower2, rpower3, savex, reusex = [ self.param[key] for key in \
                          ['widths','dgrids','rcut','rpower2','rpower3','savex','reusex'] ]
        fx = self.wd + '/%sx_%s%s.npz'%(self._aux_x, self.srep, self._label_x )
        if not hasattr(self, 'mbtypes'):
            if self.verbose:
                print('## enumerate all possible many-body terms...')
            self.get_slatm_mbtypes()
            if self.verbose: print('   done')
        #print('mbtypes = ', self.mbtypes)

        # training mols
        x1 = []; zs1 = []
        if os.path.exists(fx) and reusex:
            if self.verbose:  print('found repr file: %s, now loading...'%fx)
            x1 = np.load(fx)['x']
            if self.verbose: print('   x.shape=',x1.shape)
        else:
            if self.verbose:
                print('calc representation from scratch...')
                print('            fx=', fx)
            for i in self._ims_blk:
                ib, ie = self.ias1[i], self.ias2[i]
                _zs = self.obj.zs[ib:ie]
                zs1 += list(_zs)
                _coords = self.obj.coords[ib:ie]
                g_dic = None
                if self.icg:
                    rawm = cmc.RawMol([_zs,_coords], scale=1.0)
                    g_dic = rawm.get_slatm_nbrs()
                #print('g_dic = ', g_dic)
                xi = generate_slatm(_coords, _zs, self.mbtypes, cg=g_dic, izeff=self.izeff, unit_cell=None,
                                    local=self.local, sigmas=widths, dgrids=dgrids,
                                    rcut=rcut, alchemy=False, pbc=self.pbc, rpower=rpower2)
                x1.append(xi)
            x1 = np.concatenate(x1) if self.local else np.array(x1)
            ########### IMPORTANT! Multiply `X by dgrid
            _factor = 1.
             # Note that when cab=F, multiply by _factor or not has no impact on d(i,j)
            if self.param['isqrt_grid']:
              if self.kernel in ['g','gaussian']:    #################### You may comment this line out
                _factor = 1./np.sqrt(dgrids[0])      #################### You may comment this line out
            n1 = self.nsx[0] # note that there are `n1 unique Z's
            x1[:,n1:] *= _factor
            if savex:
                if self.forbid_w:
                    raise Exception("#ERROR: if it's safe to write x, reset forbid_w=T")
                np.savez(fx, x=x1)
        #print('++zs1=',zs1)
        return x1, np.array(zs1)

    def get_idx1(self, im, ia):
        """ get idx of atom in `zs """
        return self.ias1[im]+ia

    def get_idx2(self, im, ia):
        """ get idx of atom in `zs """
        return self.ias2[im]+ia

    def get_ds(self):
        """ calc (a)SLATM distance between atoms/molecules """
        ds1 = qd.l2_distance(self.x1, self.x1)
        ds2 = qd.l2_distance(self.x2, self.x1)
        return ds1, ds2

    def get_dsmax(self, _label=None, itarget=F, saved=F, reused=F, cdmax_easy=T):
        """calc `dmax between aSLATM of two atoms
        """
        saved, reused = [ self.param[key] for key in ['saved','reused'] ]
        label = self.label if _label is None else _label
        fdmax = self.wd+'/%s_dsmax_%s_%s.txt'%(label,self.kernel, self.srep)
        if os.path.exists(fdmax) and reused:
            dsmax = np.loadtxt(fdmax)
            dso = dsmax
            if self.verbose:
                print('   found dsmax in file %s'%fdmax)
                print('       dsmax=', ' '.join(['%.8f'%di for di in dsmax]))
        else:
            if (not self.use_chunks[0]) and itarget: # use aslatm of atoms in target to calc dmax
                x = self.x2
                zs = self._zs2
            else:
                x = self.x1
                zs = self._zs1
            _zsu = np.unique(zs)
            if self.local:
                assert len(x)==len(zs), '#ERROR: `x and `zs shape mismatch'
                #zmax = np.max(_zsu)
                Nz = len(_zsu)
                if Nz == 1:
                    ds = self.dfunc(x,x)
                    dsmax = np.array([np.max(ds)]*Nz)
                else: # Nz == 2:
                    if self.cab:
                        if cdmax_easy:
                            # by printing out (zi,zj,dmax_i), one finds that the largest
                            # dmax_i corresponds to the dist between the envs associated
                            # with the largetest two different Z's.
                            if self.verbose:
                                print('  calc dmax as max(Z_%s,Z_%s)'%(_zsu[-2],_zsu[-1]))
                            filt1 = (zs == _zsu[-2]); filt2 = (zs == _zsu[-1])
                            ds = self.dfunc(x[filt1,:],x[filt2,:])
                        else:
                            ds = []
                            for i in range(Nz):
                                # `i starts from 1 instead of 0 (i.e., 'H' atom) due to that
                                # d(H,X) << d(X,X'), where X stands for any heavy atom
                                for j in range(i,Nz):
                                    filt1 = (zs == _zsu[i]); filt2 = (zs == _zsu[j])
                                    _ds = self.dfunc(x[filt1],x[filt2])
                                    dmax_i = np.max(_ds)
                                    #print( 'zi,zj,dmax=', _zsu[i],_zsu[j], dmax_i)
                                    ds.append( dmax_i )
                        dsmax = np.array([np.max(ds)]*Nz)
                    else:
                        # one dmax per atom type
                        dsmax = []
                        for i in range(Nz):
                            # `i starts from 1 instead of 0 (i.e., 'H' atom) due to that
                            # d(H,X) << d(X,X'), where X stands for any heavy atom
                            #dmax_i = 0.
                            #for j in range(i,Nz):
                            filt1 = (zs == _zsu[i])
                            #    filt2 = (zs == _zsu[j])
                            _ds = self.dfunc(x[filt1],x[filt1]) #2])
                            dmax_i = np.max(_ds) #max(dmax_i, np.max(_ds))
                            dsmax.append(dmax_i)
                        dsmax = np.array(dsmax)
                dso = np.ones(self.zmax) * 1.e-9 # in Fortran, first entry: 1
                for i,zi in enumerate(_zsu):
                    dso[zi-1] = dsmax[i]
            else:
                dmax = np.max( self.dfunc(x1, x1) )
                dsmax = np.array([dmax])
                dso = dsmax
            if saved: np.savetxt(fdmax, dso, '%.8f')
            if self.verbose:
                print('     _zsu = ', ' '.join(['%d'%zi for zi in _zsu ]))
                print('     calculated dsmax=', ' '.join(['%.8f'%di for di in dsmax]))
        self.dsmax = dso

    def c_imap(self, nas):
        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0],ias2[:-1]))
        return ias1,ias2

    def c_kernel(self, zs1, zs2, x1, x2, navk1, navk2, nas1, nas2, sigmas):
        nm1, nm2 = len(nas1), len(nas2) #x2.shape[0],x1.shape[0]
        if navk1 <= 0: navk1 = nm1
        if navk2 <= 0: navk2 = nm2
        nk1, nk2 = int(ceil(nm1/navk1)), int(ceil(nm2/navk2))
        if self.verbose:
            print('       nm1=%d, nm2=%d, navk1=%d, navk2=%d'%(nm1,nm2,navk1,navk2) )
            print('          kernel calculation to be divided into %d x %d subblocks'%(nk1,nk2))
            print('          each subblock consists of %d x %d elements, '%(navk1,navk2))
        k = np.zeros((self.ncoeff,nm1,nm2))
        iasb, iase = self.c_imap(nas1)
        jasb, jase = self.c_imap(nas2)
        for i in range(nk1):
            for j in range(nk2):
                imb,ime = i*navk1, (i+1)*navk1
                if ime > nm1: ime = nm1
                jmb,jme = j*navk2, (j+1)*navk2
                if jme > nm2: jme = nm2
                iab,iae = iasb[imb],iase[ime-1] # `ime/jme should be always used like imb:ime; otherwise, ime-1
                jab,jae = jasb[jmb],jase[jme-1]
                if self.verbose:
                    print('                    now subblock: ', i,j)
                fij = self.fk[:-4] + '_sub_%03d_%03d.npz'%(i+1,j+1)
                if os.path.exists(fij):
                    k[:, imb:ime, jmb:jme] = np.load(fij)['k']
                else:
                    #cabs = np.ones((iae-iab,jae-jab)).astype(np.bool)
                    #if self.cab:
                    #    cabs = ( np.array([zs[iab:iae]],dtype=int) == zs[jab:jae] )
                    ktmp = self.kfunc(x1[iab:iae],x2[jab:jae], zs1[iab:iae],zs2[jab:jae],\
                                      nas1[imb:ime],nas2[jmb:jme], self.cab, self.zmax, sigmas)
                    k[:, imb:ime, jmb:jme] = ktmp
                    if self.saveblk:
                        np.savez(fij,k=ktmp)
        return k

    def get_kernel(self): #, _label):
        """ molecular kernel """
        savek, reusek = [ self.param[key] for key in ['savek','reusek'] ]
        #fk = self.wd + '/%sk_%s_%s%s.npz'%(self.label, self.kernel, self.srep, _label )
        #self.fk = fk
        fk = self.fk
        if self.verbose:
            print('iexist, reusek = ', os.path.isfile(fk),reusek)
        if os.path.isfile(fk) and reusek:
            if self.verbose:
                print('     found kernel file %s, read kerenl from it...'%fk)
            _dt = np.load(fk)
            k = _dt['k']
            if self.verbose:
                print('     read kernel size of k: ', k.shape)
        else:
            x1, x2 = self.x1, self.x2
            zs1 = self.zs1
            if self.kernel in ['gaussian','g', 'laplacian','l']:
                sigmas = self.dsmax * self._coeffs[..., np.newaxis]
                if self.local:
                    if self.dist == 'ii':
                        k = self.c_kernel(zs1, zs1, x1, x1, self._navk1, self._navk1, self._nas1, self._nas1, sigmas)
                    else:
                        zs2 = self.zs2
                        #print('shp1,shp2 = ', x2.shape,x1.shape, _nas2.sum(), _nas1.sum())
                        k = self.c_kernel(zs2, zs1, x2, x1, self._navk2, self._navk1, self._nas2, self._nas1, sigmas)
                else:
                    if self.dist == 'ii':
                        k = np.array([ self.kfunc(x1, x1, sigma) for sigma in sigmas ])
                    else:
                        k = np.array([ self.kfunc(x2, x1, sigma) for sigma in sigmas ])
            elif self.kernel in ['linear',]: # global  repr
                if self.dist == 'ii':
                    k = qk.linear_kernel(x1, x1)
                else:
                    k = qk.linear_kernel(x2, x1)
            else:
                raise '#ERROR: not implemented yet'
            if savek:
                if self.forbid_w:
                    raise Exception("#ERROR: if it's safe to write k, reset forbid_w=T")
                np.savez(fk, k=k)
        return k


