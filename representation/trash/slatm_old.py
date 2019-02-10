
import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np
import time,os,sys
import qml
from qml.fslatm import fget_sbot, fget_sbop
from qml.fslatm import fget_sbot_local, fget_sbop_local #, fget_sboa_local
#from qml.fslatm_moment import fget_sbot_local_moment, fget_sbop_local_moment
from qml.math import cho_solve
#from qml.kernels import gaussian_kernel, get_local_kernels_gaussian
from qml.wrappers import get_atomic_kernels_gaussian
from qml.representations import *
import qml.distance as qd
import qml.kernels as qk

import io2
import io2.xyz as ux
import io2.gaussian as ug
import deepdish as dd
import os,sys
import ase.data as ad

import cheminfo.graph as cg

import collections
import ase.io as aio

class SLATM(object):

    def __init__(self, wds_, format, regexp='', properties='AE', yfiles=None, M='slatm', \
                irepr=True, local=True, ow=False, nproc=1, istart=0, \
                cm_sorting='row-norm', \
                params = { 'central_cutoff': 4.0, 'central_decay': 0.5, \
                           'interaction_cutoff': 5.0, 'interaction_decay':1.0 }, \
                slatm_params = { 'nbody':3, 'dgrids': [0.03,0.03],  'sigmas':[0.05,0.05],\
                                 'rcut':4.8, 'alchemy':False, 'iBoA':True, 'rpower2':6, \
                                 'rpower3': 3, 'isf':0, 'kernel':'g', 'intc':1, 'moment':0, \
                                 'isqrt_dgrid':False, 'ws':[1.,1.,1.] }, \
                yfilter=None, use_conf=True, cell=None, pbc='000', unit='kcal', \
                iBE=False, icomb=0, n1train=None, n2test=None):
        """
        Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.
        Both global (``local=False``) and local (``local=True``) SLATM are available.

        A version that works for periodic boundary conditions will be released soon.

        NOTE: You will need to run the ``get_slatm_mbtypes()`` function to get the ``mbtypes`` input (or generate it manually).

        :param local: Generate a local representation. Defaulted to False (i.e., global representation); otherwise, atomic version.
        :type local: bool
        :param sigmas: Controlling the width of Gaussian smearing function for 2- and 3-body parts, defaulted to [0.05,0.05], usually these do not need to be adjusted.
        :type sigmas: list
        :param dgrids: The interval between two sampled internuclear distances and angles, defaulted to [0.03,0.03], no need for change, compromised for speed and accuracy.
        :type dgrids: list
        :param rcut: Cut-off radius, defaulted to 4.8 Angstrom.
        :type rcut: float
        :param alchemy: Swith to use the alchemy version of SLATM. (default=False)
        :type alchemy: bool
        :param pbc: defaulted to '000', meaning it's a molecule; the three digits in the string corresponds to x,y,z direction
        :type pbc: string
        :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
        :type rpower: float
        :return: 1D SLATM representation
        :rtype: numpy array
        """

        self.unit =unit
        self.nproc = nproc
        self.format = format
        self.regexp = regexp
        self.use_conf = use_conf

        self.atomic_properties = ['NMR', 'CLS','POPULATION', ]

        typeP = type(properties)
        # `scs: scale the kernel or not? (if u wanna to predict the HOMO
        # of large system after training on small
        scs = []
        if typeP is str:
            prop = properties.upper()
            scs = [False,]
            if local and prop in ['HOMO','LUMO','GAP','E1st']: scs = [True, ]
            properties = [prop, ]
            self.nprop = 1
        elif typeP is list:
            properties = [ prop.upper() for prop in properties ]
            scs = []
            self.nprop = len(properties)
            for prop in properties:
                if local and prop in ['HOMO','LUMO','GAP','E1ST']:
                    scs.append(True)
                else:
                    scs.append(False)
            #assert len( set(scs) ) == 1, '#ERROR: cannot specify '
        else:
            raise '#ERROR: `typeP not supported'
        self.scs = scs
        self.properties = properties

        # u may use filter for y values when considering atomic properties.
        # For instance, experimental NMR shifts for certain atoms may be
        # uncertain/undefined, though all coordinates are well-defined; under such
        # circumstance these y values have to be removed by using a `yfilter
        if yfilter is not None:
            assert np.all( [ prop_i in self.atomic_properties for prop_i in properties ] )
            assert type(yfilter) is float

        self.yfiles = yfiles
        self.yfilter = yfilter

        if type(wds_) is str:
            self.wds = [ wds_, ]
        elif type(wds_) is list:
            self.wds = wds_

        # the first `wd in `wds_ must be a directory since
        # later the h5 file will be written to this directory
        assert os.path.isdir(wds_[0]), '#ERROR: not a dir?'

        # update general params
        params0 = { 'central_cutoff': 4.0, 'central_decay': 0.5, \
                   'interaction_cutoff': 5.0, 'interaction_decay':1.0 }
        for key in params.keys():
            val = params[key]
            if val != params0[key]:
                params0[key] = val
        keys = ['central_cutoff','central_decay','interaction_cutoff','interaction_decay']
        sorting = cm_sorting
        cc, cd, ico, idc = [ params0[key] for key in keys ]
        self.cc = cc
        self.cd = cd
        self.ico = ico
        self.idc = idc

        # update params for SLATM
        slatm_params0 = { 'nbody':3, 'dgrids': [0.04,0.04],  'sigmas':[0.05,0.05],\
                          'rcut':4.8, 'alchemy':False, 'iBoA':True, 'rpower2':6, \
                          'rpower3': 3, 'isf':0, 'moment':0, 'ws':[1.,1./2,1./3], \
                          'kernel':'g', 'intc':1, 'isqrt_dgrid':False}
        for key in slatm_params.keys():
            val = slatm_params[key]
            if val != slatm_params0[key]:
                slatm_params0[key] = val
        keys = ['nbody','dgrids','sigmas','rcut','alchemy','iBoA', \
                'rpower2','rpower3', 'ws','isf','intc','kernel','moment','isqrt_dgrid']
        nbody,dgrids,sigmas,rcut,alchemy,iBoA,rpower2,rpower3,ws,isf,intc,kernel,moment,isqrt_dgrid = \
               [ slatm_params0[key] for key in keys ]

        self.icomb = icomb
        self.iBE = iBE
        if iBE: iBoA = False # to learn binding energy, only 2- and 3-body terms matter

        self.isf = isf
        self.ws = ws
#        if not local:
#           assert np.all( np.array(ws) == [1., 1., 1./3] )
        self.kernel = kernel
        self.intc = intc
        self.isqrt_dgrid = isqrt_dgrid
        self.nbody = nbody

        c1 = M; c2 = ''
        #print ' -- dgrids = ', dgrids
        if M == 'slatm':
          c1 += '_rc%.1f'%rcut
          c1 += '_nbody%d'%nbody
          c1 += '_dgrid%.2f_sg%.2f'%(dgrids[0],sigmas[0])
          c1 += '_ws_%.1f_%.1f_%.1f'%(ws[0],ws[1],ws[2])
          c1 += '_gw%.2f'%sigmas[0] # gaussian width centered on Atom
          c1 += '_intc_%d_rps_%d_%d'%(intc,rpower2,rpower3)
          c1 += '_isf_%d'%isf
          if isf in [2,3]:
            c1 += '_cc%.1f_cd%.1f_ico%.1f_idc%.1f'%(cc,cd,ico,idc)
          c2 = '_local' if local else '_global'
          if moment != 0: c2 += '_moment'
          h5f = self.wds[0] + '/%s%s'%(c1,c2)
        elif M == 'cm':
          c1 += '_cc%.1f_cd%.1f_ico%.1f_idc%.1f'%(cc,cd,ico,idc)
          if sorting == 'row-norm':
            c1 += '__sortByNorm'
          elif sorting == 'distance':
            c1 += '__sortByDist'
          c2 = '_local' if local else '_global'
          h5f = self.wds[0] + '/%s%s'%(c1,c2)
        elif M == 'bob':
          c1 += '_rc%.1f'%rcut
          c2 += '_global'
          h5f = self.wds[0] + '/%s%s'%(c1,c2)
        else:
          raise '#ERROR: `M not accepted'
        #print ' -- h5f = ', h5f
        if iBE: h5f += '_icomb%d'%icomb
        if alchemy: h5f += '_alchemical'
        h5f += '.h5'
        self.h5f = h5f

        self.fs = []
        for wd in self.wds:
            if os.path.isdir(wd):
                FO = io2.Folder(wd, self.format, regexp=self.regexp, use_conf=self.use_conf)
                self.fs += FO.fs
            elif os.path.isfile(wd):
                self.fs += [wd ]
            else:
                print ' -- wd = ', wd
                raise '#ERROR: ???'
            #print ' -- fs = ', self.fs
        nm = len(self.fs) # number of total molecules
        self.nm = nm

        self.local = local
        self.moment = moment
        if os.path.exists(h5f) and (not ow):
            print ' -- reading h5 file'
            dic  = dd.io.load(h5f)
            keys = dic.keys()
            X = dic['X']
            self.X = X
            self.dic = dic

            props_todo = []
            for prop in properties: #['AE',]:
                if prop not in keys:
                    props_todo.append( prop )
            if len(props_todo) > 0:
                self.get_raw_data( props_todo )

            ys = [ self.dic[propi] for propi in properties ]
            self.ys = ys

            nas  = dic['nas']
            self.nas = nas
            self.nm = len(nas)
            self.nzs = dic['nzs']
            zsr = dic['zs']
            self.zsr = zsr
            self.nhass = dic['nhass']; nm = nas.shape[0]
            self.zsu = np.unique(zsr)
            self.coords = dic['coords']
        else:
            self.dgrids = dgrids
            self.cell = cell
            self.sigmas = sigmas
            self.rcut = rcut
            self.alchemy = alchemy
            self.pbc = pbc
            self.rpower2 = rpower2
            self.rpower3 = rpower3

            self.dic = {} # dictionary of vars regarding `X and `Y to be stored in h5 file
            self.get_raw_data( properties )
            coords = self.coords
            dic = self.dic
            ss = self.ss

            ys = [ self.dic[propi] for propi in properties ]
            self.ys = ys

            zs = ss.zs
            zsr = ss.zsr # zs_ravel
            zsu = ss.zsu # zs_unique
            nas = ss.nas
            nhass = ss.nhass
            nzs = ss.nzs

            self.zs = zs
            self.zsr = np.array(zsr, np.int)
            self.zsu = zsu
            self.nhass = np.array(nhass, np.int)
            self.nas = np.array(nas, np.int)
            #self.nes = np.array(nes)
            self.ias2 = np.cumsum(nas)
            self.ias1 = np.concatenate(([0,],self.ias2[:-1]))

            M_u = M.lower()
            if M_u != 'slatm':
                mols = []
                for idf,fi in enumerate(self.fs):
                    if fi[-3:] == 'xyz':
                        mol = qml.Compound(xyz=fi)
                    elif fi[-3:] == 'out':
                        fiu = fi[:-3]+'xyz'
                        aio.write(fiu, ss.objs[idf].atoms)
                        mol = qml.Compound(xyz=fiu)
                    mols.append(mol)
                msize = max(mol.nuclear_charges.size for mol in mols) + 1
                asymbs = [ ad.chemical_symbols[zk] for zk in zsu ]
                asize = dict( zip( asymbs, np.max(nzs,axis=0) ) ); #print ' -- asize = ', asize
                X = []
                if M_u == 'cm':
                    if local:
                        lmsize = 200
                        # Generate coulomb matrix representation, sorted by distance, with soft cutoffs
                        for i, mol in enumerate(mols):
                            mol.generate_atomic_coulomb_matrix(size = lmsize, sorting = sorting,
                                 central_cutoff = cc, central_decay = cd,
                                 interaction_cutoff = ico, interaction_decay = idc)

                        for mol in mols: X += list(mol.representation)
                        X = np.array(X)
                    else:
                        # Generate coulomb matrix representation, sorted by row-norm
                        for i, mol in enumerate(mols):
                            mol.generate_coulomb_matrix(size = msize, sorting = "row-norm")
                        X = np.asarray([mol.representation for mol in mols])
                    self.X = X
                elif M_u == 'bob':
                    if local:
                        raise '#ERROR: ____'
                    else:
                        for i, mol in enumerate(mols):
                            mol.generate_bob(asize)
                        X = np.asarray([mol.representation for mol in mols])
                        self.X = X
            else:
                self.get_slatm_mbtypes(iBoA)
                if irepr:
                    X = []
                    for i in range(self.nm):
                        #print ' -- Mid #%06d'%(i+1)
                        xi = self.generate_slatm(i)
                        #print ' -- x2n = ', self.X2N
                        if local:
                            for xij in xi:
                                X.append(xij)
                        else:
                            X.append(xi)
                    X = np.array( X )
                    self.X = X
                else:
                    # work with distance matrix only
                    # u must have specified `n1train and `n2test, i.e., maximal
                    # training set size & fixed test size
                    assert n1train != None and n2test != None
                    self.get_slatm_atomic_distance(n1train, n2test)

            self.nas = np.array(nas,np.int)
            self.nhass = np.array(nhass,np.int)
            self.nzs = np.array(nzs,np.int)
            self.zsr = np.array(zsr,np.int)

            # filter y values
            #ys1 = self.ys[0]
            #if yfilter is not None:
            #    ys1u = ys1[ ys1 < yfilter ]
            #    ys = [ ys1u, ]
            #    self.ys = ys

            if irepr:
                #assert not np.all(Y == 0.0), '#ERROR: Y not assigned?'
                dic2 = {'X':X, 'nas':self.nas, 'nzs':self.nzs, \
                        'zs':np.array(self.zsr, np.int), 'nhass':self.nhass, \
                        'coords': np.array(self.coords)} #, 'nes':self.nes} )
            else:
                dic2 = {'nas1':self.nas1,'nas2':self.nas2,'ds1':self.ds1,'ds2':self.ds2, \
                        'nas':self.nas, 'nzs':self.nzs, 'zs':self.zsr, 'nhass':self.nhass, \
                        'coords': self.coords} #, 'nes':self.nes} )
            for key_i in dic2.keys(): dic[key_i] = dic2[key_i]
            dd.io.save( h5f, dic )

        # for FORCE prediction
        self.visited_coordinates = []


    def get_raw_data(self, props_todo):
        nm = self.nm
        if isinstance(self.yfiles, str):
            yfiles = [self.yfiles]
        else:
            yfiles = self.yfiles
        igot_ys = False
        if yfiles is not None:
            for yfile in yfiles:
                assert os.path.exists(yfiles), '#ERROR: Y file does not exist??'
                prop_name = yfiles.split('/')[-1][:-4].split('_')[0] # e.g., yfiles = '24/E.dat', '24/E_bsse.dat'
                assert prop_name in props_todo, '#ERROR: `yfiles format should be sth like AE.dat, HOMO.dat'
                Yi = np.loadtxt(yfiles)
                self.dic[prop_name] = Yi
            igot_ys = True

        # get `zs and others to generate rpst
        if self.format == 'out': # Gaussian output file
            if not igot_ys:
                ss = ug.GRs( self.fs, properties=props_todo, unit=self.unit, nproc=self.nproc) #, istart=istart )
                for propi in props_todo:
                    if propi in ['force','FORCE']:
                        self.dic[propi] = np.concatenate( ss.dic[ propi ], axis=0)
                    else:
                        self.dic[propi] = np.array( ss.dic[ propi ] )
        elif self.format == 'xyz':
            ss = ux.XYZFiles( self.fs, nproc=self.nproc )
            assert igot_ys
        else:
            raise '#ERROR: no such `format'
        ss.get_statistics()
        self.ss = ss
        self.coords = np.concatenate( ss.coords, axis=0 )

        # check if the number of entries from `yfiles is consistent
        if yfiles is not None:
            nat = np.sum( ss.nas )
            for prop_name in props_todo:
                if prop_name in self.atomic_properties:
                    # atomic property
                    print 'nat, len(Y) = ', nat, len(Y)
                    assert nat == len(self.dic[prop_name]), '#ERROR: number of entries in `Y_raw inconsistent!'
                else:
                    # for global properties
                    assert nm == len(self.dic[prop_name]), '#ERROR: number of entries in `Y_raw inconsistent!'


    def get_dmax_local(self):
        """calc `dmax between aSLATM of two atoms"""
        fdmax = self.wds[0] + '/dmax.txt'
        X = self.X
        if os.path.exists(fdmax):
            dmax = eval( file(fdmax).readlines()[0] )
        else:
            _zsu = list(self.zsu); _zsu.sort(); Nz = len(_zsu)
            if Nz == 2:
              filt1 = (self.zsr == _zsu[0]); filt2 = (self.zsr == _zsu[1])
              ds = qd.l2_distance(X[filt1,:],X[filt2,:]) # ssd.pdist(x1, metric='euclidean')
              dmax_i = np.max(ds); dsmax.append( dmax_i )
              #print ' -- zi, zj, dmax = ', _zsu[0], _zsu[1],dmax_i
            else:
              for i in range(1,Nz-1):
                for j in range(i+1,Nz):
                    filt1 = (self.zsr == _zsu[i]); filt2 = (self.zsr == _zsu[j])
                    ds = qd.l2_distance(X[filt1,:],X[filt2,:]) # ssd.pdist(x1, metric='euclidean')
                    dmax_i = np.max(ds); dsmax.append( dmax_i )
                    #print ' -- zi, zj, dmax = ', _zsu[i], _zsu[j],dmax_i
            dmax = max(dsmax) #"""
            open(self.wd[0]+'/sigma.txt', 'w').write('%.8f'%dmax)
        self.dmax = dmax


    def save_k_local(self, coeffs=[1.0], nq=1):
        """molecular kernel k1s based on aSLATM
        note:
        Both `nblk and `iblk are integers. nblk starts from 1 while
        `ir and `ic start from 0
        """
        X = self.X
        if not hasattr(self,dmax):
            self.get_dmax_local()
        sgs = np.array(coeffs)*self.dmax/np.sqrt(2.0*np.log(2.0))
        namon = self.nm-nq
        imb1 = 0; ime1 = namon-1
        imb2 = namon; ime2 = namon+nq-1
        ias2 = np.cumsum(self.nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))
        iab1 = ias1[imb1]; iae1 = ias2[ime1] # idx_atom_beginning (iab)
        iab2 = ias1[imb2]; iae2 = ias2[ime2]
        X1 = X[iab1:iae1]; X2 = X[iab2:iae2]
        nas1 = self.nas[imb1:ime1+1]
        nas2 = self.nas[imb2:ime2+1]
        k1s = qk.get_local_kernels_gaussian(X1, X1, nas1, nas1, sgs)
        #for i in range(nsg):
        #    k1s[i] = k1s[i] - np.eye(namon)*lm0
        for K1 in K1s: assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"
        k2s = qk.get_local_kernels_gaussian(X2, X1, nas2, nas1, sgs)
        h5f = self.h5f[:-3]+'_K_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs]))
        dd.io.save(h5f, {'k1s':k1s, 'k2s': k2s})


    def save_k1_local(self, coeffs=[1.0], nq=1, nblk=1, ir=0, ic=0):
        """molecular kernel k1s based on aSLATM
        note:
        Both `nblk and `iblk are integers. nblk starts from 1 while
        `ir and `ic start from 0
        """
        X = self.X
        if not hasattr(self,dmax):
            self.get_dmax_local()
        sgs = np.array(coeffs)*self.dmax/np.sqrt(2.0*np.log(2.0))
        namon = self.nm-nq
        nav = namon/nblk
        navf = namon*1.0/nblk
        if nav != navf: nav += 1
        assert ir <= nblk-1 and ic <= nblk-1
        imb1 = ir*nav; ime1 = (ir+1)*nav-1 # index_molecule_beginning (imb), ..
        imb2 = ic*nav; ime2 = (ic+1)*nav-1 # index_molecule_beginning (imb), ..
        if ime1 > namon-1: ime1 = namon-1
        if ime2 > namon-1: ime2 = namon-1

        ias2 = np.cumsum(self.nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        iab1 = ias1[imb1]; iae1 = ias2[ime1] # idx_atom_beginning (iab)
        iab2 = ias1[imb2]; iae2 = ias2[ime2]
        X1 = X[iab1:iae1]; X2 = X[iab2:iae2]

        nas1 = self.nas[imb1:ime1+1]
        nas2 = self.nas[imb2:ime2+1]
        k1s = qk.get_local_kernels_gaussian(X1, X2, nas1, nas2, sgs)
        for K1 in K1s: assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"
        xtra = ''
        if nblk > 1:
            xtra = '_nblk%d_ir%d_ic%d'%(nblk,ir,ic)
        h5fk1 = self.h5f[:-3]+'_K1_sgs_%s%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs]), xtra)
        dd.io.save(h5fk1, {'k1s': k1s})


    def save_k2_local(self, coeffs=[1.0], nq=1):
        """molecular kernel k2s
        """
        X = self.X
        if not hasattr(self,dmax):
            self.get_dmax_local()
        sgs = np.array(coeffs)*self.dmax/np.sqrt(2.0*np.log(2.0))
        namon = self.nm-nq
        imb1 = 0; ime1 = namon-1
        imb2 = namon; ime2 = namon+nq-1

        ias2 = np.cumsum(self.nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        iab1 = ias1[imb1]; iae1 = ias2[ime1] # idx_atom_beginning (iab)
        iab2 = ias1[imb2]; iae2 = ias2[ime2]
        X1 = X[iab1:iae1]; X2 = X[iab2:iae2]

        nas1 = self.nas[imb1:ime1+1]
        nas2 = self.nas[imb2:ime2+1]
        k2s = qk.get_local_kernels_gaussian(X2, X1, nas2, nas1, sgs)
        for k2 in k2s: assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"
        h5fk2 = self.h5f[:-3]+'_K2_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs]))
        dd.io.save(h5fk2, {'k2s': k2s})


    def merge_k(self, lm0, coeffs=[1.0], nq=1, nblk=1):
        nsg = len(coeffs)
        if nblk == 1:
            fk = self.h5f[:-3]+'_K_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs]))
            d0 = dd.io.load(fk)
            if 'K11' in d0.keys():
                k1s = d0['K11']; k2s = d0['K12']
            elif 'k1s' in d0.keys():
                k1s = d0['k1s']; k2s = d0['k2s']
            else:
                raise '#ERROR: neither `K11 nor `k1s is in h5 file??'
            for i in range(nsg):
                k1s[i] = k1s[i] - np.eye(namon)*lm0
            self.k1s = k1s
            self,k2s = k2s
        else:
            namon = self.nm-nq
            nav = namon/nblk
            navf = namon*1.0/nblk
            if nav != navf: nav += 1
            xtra = '_nblk%d'%nblk
            fk = self.h5f[:-3] +  xtra
            ssg = '_K1_sgs_%s'%('_'.join(['%.2f'%coeff for coeff in coeffs]))
            # now k1s
            k1s = np.zeros((nsg,namon,namon))
            for ir in range(nblk):
                for ic in range(ir,nblk):
                    fk1 = fk + ssg + '_ir%d_ic%d'%(ir,ic) + '.h5'
                    print 'fk1 = ', fk1
                    assert os.path.exists(fk1)
                    d0 = dd.io.load(fk1)
                    _k1s = d0['k1s']
                    imb1 = ir*nav; ime1 = (ir+1)*nav-1 # index_molecule_beginning (imb), ..
                    imb2 = ic*nav; ime2 = (ic+1)*nav-1 # index_molecule_beginning (imb), ..
                    if ime1 > namon-1: ime1 = namon-1
                    if ime2 > namon-1: ime2 = namon-1
                    for isg in range(nsg):
                        if ir == ic:
                            k1s[isg, imb1:ime1+1, imb2:ime2+1] = _k1s[isg].triu()
                        else:
                            k1s[isg, imb1:ime1+1, imb2:ime2+1] = _k1s[isg]
            k1s_u = np.zeros((nsg,namon,namon))
            for isg in range(nsg):
                k1 = k1s[isg]
                k1s_u[isg,:,:] = np.maximum(k1,k1.T)
            k1s = k1s_u
            self.k1s = k1s
            # now k2s
            h5fk2 = self.h5f[:-3]+'_K2_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs]))
            d0 = dd.io.load(h5fk2)
            k2s = d0['k2s']
            self.k2s = k2s


    def ML_local(self, N1s=None, aml=False, nt=None, usebl=True, coeffs=[1.0,], \
                 isign=False, pf=1.0, iprt=False, seed=1, namin=1, namax=10000, \
                 nheavDisp=[], idsOnly=[], idsOut=[], iprt0=False, llambda=1e-4 ):
        """
        train the local ML model
        notes:
        ----------------------------------------
        namax -- amons with Num_heavy_Atom_max
        namin -- amons with Num_heavy_Atom_min
        isign -- print out signed error? (when `nt = 1)
        """

        rand = ( not aml )
        self.update_N1s(N1s, nt, rand=rand, idsOnly=idsOnly, idsOut=idsOut, seed=seed, \
                        nheavDisp=nheavDisp, namin=namin, namax=namax)
        N1s = self.N1s
        #print ' -- N1s, nis = ', N1s, self.nis
        nm = self.nm
        if not rand: nis = self.nis
        tidxs = self.tidxs
        nhass = self.nhass
        nas   = self.nas
        tidxs = self.tidxs
        nzs   = self.nzs
        zsr   = self.zsr
        ias1  = self.ias1
        ias2  = self.ias2

        scs = self.scs
        #print ' -- N1s = ', N1s

        if pf != 1.0: # power_factor
            # 2.0/v # v = 2.26 (for m.p.), 3.30 (for b.p.)
            val = 0.0
            for q in range(nm):
                nzs[q] = nzs[q]/(nas[q]**pf + val)
                #nzs[q] = nzs[q]/(nes[q]**pf + val)

        Y0 = np.array( self.dic[self.properties[0]] )
        Y = Y0[tidxs]
        Xu = []
        for i in range(self.nm):
            ib = ias1[i]; ie = ias2[i]
            Xu += list( self.X[ib:ie] )
        X = np.array(Xu)

        #obsolete = """
        dsmax = []
        #print ' nat, zsu, nzs = ', len(self.zsr), self.zsu, [ (self.zsu[i] == self.zsr).sum() for i in range(len(self.zsu)) ]
        _zsu = list(self.zsu); _zsu.sort(); Nz = len(_zsu)
        if Nz == 2:
          filt1 = (self.zsr == _zsu[0]); filt2 = (self.zsr == _zsu[1])
          ds = qd.l2_distance(X[filt1,:],X[filt2,:]) # ssd.pdist(x1, metric='euclidean')
          dmax_i = np.max(ds); dsmax.append( dmax_i )
          #print ' -- zi, zj, dmax = ', _zsu[0], _zsu[1],dmax_i
        else:
          for i in range(1,Nz-1):
            for j in range(i+1,Nz):
                filt1 = (self.zsr == _zsu[i]); filt2 = (self.zsr == _zsu[j])
                ds = qd.l2_distance(X[filt1,:],X[filt2,:]) # ssd.pdist(x1, metric='euclidean')
                dmax_i = np.max(ds); dsmax.append( dmax_i )
                #print ' -- zi, zj, dmax = ', _zsu[i], _zsu[j],dmax_i
        dmax = max(dsmax) #"""
        #dmax = 18.6293688909

        # update `ias1, `ias2, MUST-DO !!
        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        self.maes = []
        self.rmses = []
        self.errors = []
        self.N1s = N1s
        #print ' @@'

        nsg = len(coeffs)
        s0 = '           '
        for ix in range(nsg):
            s0 += '|     sigma = %4.2f    '%coeffs[ix]
        print s0

        for j,N1 in enumerate(N1s):
            if N1 == 0: continue
            ntarget = nt if aml else nm - N1
            X1 = X[ :ias2[N1-1] ]
            it2 = nm # the ending Idx of Target(s)

            # Note that from now on, we fix the number of test examples, regardless of random sampling or not
            it1 = nm - ntarget # the beginning Idx of Target(s)
            #print ' -- it1, it2 = ', it1, it2
            #print ' -- ias1[it1], ias2[it2-1] = ', ias1[it1], ias2[it2-1]
            X2 = X[ias1[it1]:ias2[it2-1]]
            Y1 = Y[:N1]
            Y2 = Y[it1:it2]
            nas1 = nas[:N1]
            nas2 = nas[it1:it2]
            #nes1 = nes[:N1]
            #nes2 = nes[N1:nm]
            nzs1 = nzs[:N1]
            nzs2 = nzs[it1:it2]

            if usebl:
                esb = np.linalg.lstsq(nzs1,Y1)[0]
            else:
                esb = np.zeros((nzs1.shape[1],1))
            #print ' -- Y1 = ', Y1.shape
            Y1 = Y1 - np.squeeze( np.dot(nzs1,esb) )
            #print ' -- o2  Y1 = ', Y1
            Y2 = Y2 - np.squeeze( np.dot(nzs2,esb) )
            #if len(Y2) <= 2: print ' -- Y2 = ', Y2, '  @'
            Y1 = np.squeeze(Y1); Y2 = np.squeeze(Y2)
            if Y1.shape is (): Y1 = np.array([Y1,])

            sgs = np.array(coeffs)*dmax/np.sqrt(2.0*np.log(2.0))

            #print ' sigma = ', sigma
            #print ' calculating K1 ... '
            K1s = qk.get_local_kernels_gaussian(X1, X1, nas1, nas1, sgs)
            #print ' K1 is obtained!'
            pn = 1 ##################################
            if scs[0]: # scale the kernel
                diag1s = [ K1.diagonal() for K1 in K1s ]
                K1s = [ ( K1s[ix]/np.sqrt( diag1s[ix][:,np.newaxis]*diag1s[ix][np.newaxis,:] ) )**pn \
                              for ix in range(nsg) ]
            #print ' -- K1 = ', K1
            for K1 in K1s:
                assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"

            if iprt:
                print ' -- esb = '
                print esb
                print ' -- Y2 - baseline = '
                print Y2
            if pf != 1.0:
                nas_x, nas_y = np.meshgrid(nas1,nas1)
                scale_factors = (nas_x**pf + val)*(nas_y**pf + val)
                for ix in range(nsg):
                    K1i = K1s[ix]
                    K1s[ix] = K1i/scale_factors

            # Solve alpha
            for i in range(nsg):
                K1i = K1s[i]
                K1i[np.diag_indices_from(K1i)] += llambda
                K1s[i] = K1i

            self.K1s = K1s
            if iprt: print ' -- now get alphas ', time.ctime()
            #print ' - Y1 = ', Y1
            #print ' calc alpha ... '
            #print ' ***** ', K1.shape, Y1.shape
            if len(Y1) == 1:
                alphas = [ np.array([ [ Y1[0]/K1[0,0], ] ]) for K1 in K1s ]
            else:
                alphas = [ np.array([ cho_solve(K1,Y1) ]).T for K1 in K1s ]
            #print ' alpha done!'

            # Calculate prediction kernel
            if iprt: print ' -- now get prediction kernel'
            #print ' shape of X2, nas2 = ', X2.shape, nas2.shape
            #print ' shape of X1, nas1 = ', X1.shape, nas1.shape
            K21s = qk.get_local_kernels_gaussian(X2, X1, nas2, nas1, sgs)
            #print ' __ K21.shape = ', K21.shape
            #print ' kernel done'
            if scs[0]:
                K2s = qk.get_local_kernels_gaussian(X2, X2, nas2, nas2, sgs)
                for ix in range(nsg):
                    diag2 = K2s[ix].diagonal()
                    K21_ix = ( K21s[ix]/np.sqrt( diag2[np.newaxis,:]*diag1[np.newaxis,:] ) )**pn
                    K2s[ix] = K21_ix
            #print ' ___ K21 = ', K21
            #print ' __ K21.shape = ', K21.shape

            if pf != 1.0:
                nas_x, nas_y = np.meshgrid(nas1,nas2)
                scale_factors = (nas_x**pf + val)*(nas_y**pf + val)
                for i in range(nsg):
                    K21i = K21s[i]/scale_factors
                    K21s[i] = K21i

            self.K21s = K21s
            dd.io.save(self.h5f[:-3]+'_K_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs])), \
                                        {'K11':np.array(K1s), 'K12':np.array(K21s)})
            #print ' starting prediction ... '
            y2s_est = [ np.squeeze( np.dot(K21s[ix], alphas[ix] ) ) for ix in range(nsg) ]
            y2_dft = np.squeeze(Y2)
            errors_ = [ y2est - y2_dft for y2est in y2s_est ]
            maes_ = [ np.sum(np.abs(error))/ntarget for error in errors_ ]
            rmses_ = [ np.sqrt( np.sum( error**2 )/ntarget ) for error in errors_ ]
            self.maes.append( maes_ )
            self.rmses.append( rmses_ )
            self.errors.append( errors_ )
            if nt == 1 and isign: maes_ = errors_
            if aml:
                s = ' %4d %4d '%(nis[j],N1)
                for ix in range(nsg): s += ' | %9.4f %9.4f'%(maes_[ix], rmses_[ix])
                print s
                if iprt0:
                    for ii in range(nsg):
                        for jj in range(ntarget):
                            print ' %9.4f %9.4f %9.4f'%( y2_dft[jj],y2s_est[ii][jj], errors_[ii][jj])
            else:
                s = ' %4d '%N1
                for ix in range(nsg):
                    s += ' |  %9.4f %9.4f  @@'%(maes_[ix], rmses_[ix])
                print s


    def ML_local_2(self, nblk, lm0, N1s=None, aml=False, nq=None, coeffs=[1.0,], \
                   usebl=True, namin=1, namax=10000, llambda=1e-4, \
                   isign=False, pf=1.0, iprt=False, seed=1, idsOnly=[], \
                   idsOut=[], nheavDisp=[], iprt0=False):
        """
        train the local ML model
        notes:
        ----------------------------------------
        namax -- amons with Num_heavy_Atom_max
        namin -- amons with Num_heavy_Atom_min
        isign -- print out signed error? (when `nq = 1)
        """

        rand = ( not aml )
        self.update_N1s(N1s, nq, rand=rand, idsOnly=idsOnly, idsOut=idsOut, seed=seed, \
                        nheavDisp=nheavDisp, namin=namin, namax=namax)
        N1s = self.N1s
        #print ' -- N1s, nis = ', N1s, self.nis
        nm = self.nm
        if not rand: nis = self.nis
        tidxs = self.tidxs
        nhass = self.nhass
        nas   = self.nas
        tidxs = self.tidxs
        nzs   = self.nzs
        zsr   = self.zsr
        ias1  = self.ias1
        ias2  = self.ias2

        scs = self.scs
        #print ' -- N1s = ', N1s

        if pf != 1.0: # power_factor
            # 2.0/v # v = 2.26 (for m.p.), 3.30 (for b.p.)
            val = 0.0
            for q in range(nm):
                nzs[q] = nzs[q]/(nas[q]**pf + val)
                #nzs[q] = nzs[q]/(nes[q]**pf + val)

        Y0 = np.array( self.dic[self.properties[0]] )
        Y = Y0[tidxs]
        Xu = []
        for i in range(self.nm):
            ib = ias1[i]; ie = ias2[i]
            Xu += list( self.X[ib:ie] )
        X = np.array(Xu)

        self.get_dmax_local()
        dmax = self.dmax

        # update `ias1, `ias2, MUST-DO !!
        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        self.maes = []
        self.rmses = []
        self.errors = []
        self.N1s = N1s
        #print ' @@'


        nsg = len(coeffs)
        s0 = '           '
        for ix in range(nsg):
            s0 += '|     sigma = %4.2f    '%coeffs[ix]
        print s0

        # retrieve kernel
        self.merge_k(lm0, coeffs=coeffs, nq=nq, nblk=nblk)
        k1s = self.k1s
        k2s = self.k2s

        for j,N1 in enumerate(N1s):
            if N1 == 0: continue
            ntarget = nq if aml else nm - N1
            X1 = X[ :ias2[N1-1] ]
            it2 = nm # the ending Idx of Target(s)

            # Note that from now on, we fix the number of test examples, regardless of random sampling or not
            it1 = nm - ntarget # the beginning Idx of Target(s)
            #print ' -- it1, it2 = ', it1, it2
            #print ' -- ias1[it1], ias2[it2-1] = ', ias1[it1], ias2[it2-1]
            X2 = X[ias1[it1]:ias2[it2-1]]
            Y1 = Y[:N1]
            Y2 = Y[it1:it2]
            nas1 = nas[:N1]
            nas2 = nas[it1:it2]
            #nes1 = nes[:N1]
            #nes2 = nes[N1:nm]
            nzs1 = nzs[:N1]
            nzs2 = nzs[it1:it2]

            if usebl:
                esb = np.linalg.lstsq(nzs1,Y1)[0]
            else:
                esb = np.zeros((nzs1.shape[1],1))
            #print ' -- Y1 = ', Y1.shape
            Y1 = Y1 - np.squeeze( np.dot(nzs1,esb) )
            #print ' -- o2  Y1 = ', Y1
            Y2 = Y2 - np.squeeze( np.dot(nzs2,esb) )
            #if len(Y2) <= 2: print ' -- Y2 = ', Y2, '  @'
            Y1 = np.squeeze(Y1); Y2 = np.squeeze(Y2)
            if Y1.shape is (): Y1 = np.array([Y1,])

            sgs = np.array(coeffs)*dmax/np.sqrt(2.0*np.log(2.0))

            #print ' sigma = ', sigma
            #print ' calculating K1 ... '
            K1s = [ k1s[i,:N1, :N1] for i in range(nsg) ]
            #print ' K1 is obtained!'
            pn = 1 ##################################
            if scs[0]: # scale the kernel
                diag1s = [ K1.diagonal() for K1 in K1s ]
                K1s = [ ( K1s[ix]/np.sqrt( diag1s[ix][:,np.newaxis]*diag1s[ix][np.newaxis,:] ) )**pn \
                              for ix in range(nsg) ]

            if iprt:
                print ' -- esb = '
                print esb
                print ' -- Y2 - baseline = '
                print Y2
            if pf != 1.0:
                nas_x, nas_y = np.meshgrid(nas1,nas1)
                scale_factors = (nas_x**pf + val)*(nas_y**pf + val)
                for ix in range(nsg):
                    K1i = K1s[ix]
                    K1s[ix] = K1i/scale_factors

            # Solve alpha
            for i in range(nsg):
                K1i = K1s[i]
                K1i[np.diag_indices_from(K1i)] += llambda
                K1s[i] = K1i

            self.K1s = K1s
            if iprt: print ' -- now get alphas ', time.ctime()
            #print ' - Y1 = ', Y1
            #print ' calc alpha ... '
            #print ' ***** ', K1.shape, Y1.shape
            if len(Y1) == 1:
                alphas = [ np.array([ [ Y1[0]/K1[0,0], ] ]) for K1 in K1s ]
            else:
                alphas = [ np.array([ cho_solve(K1,Y1) ]).T for K1 in K1s ]
            #print ' alpha done!'

            # Calculate prediction kernel
            if iprt: print ' -- now get prediction kernel'
            #print ' shape of X2, nas2 = ', X2.shape, nas2.shape
            #print ' shape of X1, nas1 = ', X1.shape, nas1.shape
            K21s = k2s #qk.get_local_kernels_gaussian(X2, X1, nas2, nas1, sgs)
            #print ' __ K21.shape = ', K21.shape
            #print ' kernel done'
            if scs[0]:
                K2s = qk.get_local_kernels_gaussian(X2, X2, nas2, nas2, sgs)
                for ix in range(nsg):
                    diag2 = K2s[ix].diagonal()
                    K21_ix = ( K21s[ix]/np.sqrt( diag2[np.newaxis,:]*diag1[np.newaxis,:] ) )**pn
                    K2s[ix] = K21_ix
            #print ' ___ K21 = ', K21
            #print ' __ K21.shape = ', K21.shape

            if pf != 1.0:
                nas_x, nas_y = np.meshgrid(nas1,nas2)
                scale_factors = (nas_x**pf + val)*(nas_y**pf + val)
                for i in range(nsg):
                    K21i = K21s[i]/scale_factors
                    K21s[i] = K21i

            self.K21s = K21s
#            dd.io.save(self.h5f[:-3]+'_K_sgs_%s.h5'%('_'.join(['%.2f'%coeff for coeff in coeffs])), \
#                                        {'K11':np.array(K1s), 'K12':np.array(K21s)})
            #print ' starting prediction ... '
            y2s_est = [ np.squeeze( np.dot(K21s[ix], alphas[ix] ) ) for ix in range(nsg) ]
            y2_dft = np.squeeze(Y2)
            errors_ = [ y2est - y2_dft for y2est in y2s_est ]
            maes_ = [ np.sum(np.abs(error))/ntarget for error in errors_ ]
            rmses_ = [ np.sqrt( np.sum( error**2 )/ntarget ) for error in errors_ ]
            self.maes.append( maes_ )
            self.rmses.append( rmses_ )
            self.errors.append( errors_ )
            if nq == 1 and isign: maes_ = errors_
            if aml:
                s = ' %4d %4d '%(nis[j],N1)
                for ix in range(nsg): s += ' | %9.4f %9.4f'%(maes_[ix], rmses_[ix])
                print s
                if iprt0:
                    for ii in range(nsg):
                        for jj in range(ntarget):
                            print ' %9.4f %9.4f %9.4f'%( y2_dft[jj],y2s_est[ii][jj], errors_[ii][jj])
            else:
                s = ' %4d '%N1
                for ix in range(nsg):
                    s += ' |  %9.4f %9.4f  @@'%(maes_[ix], rmses_[ix])
                print s


    def update_N1s(self, N1s, nt, rand=False, idsOnly=[], idsOut=[], \
                   nheavDisp=[], seed=1, namin=1, namax=10000):
        """
        update training & test data

        nt      -- num_target molecule
        idsOnly -- Idxs of amons to be used for training
        idsOut  -- Idxs of amons not to be used for training
        """

        n1_In = len(idsOnly)
        n1_Out = len(idsOut)
        iok1 = ( n1_In > 0 )
        iok2 = ( n1_Out > 0 )
        assert not (iok1 and iok2), '#ERROR: '

        nm = self.nm
        tidxs = range(nm) # idxs of the whole dataset
        self.tidxs = tidxs

        nhass = self.nhass
        zsr = self.zsr

        nas = self.nas
        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        typeN = type(N1s)
        sort_ias = False
        if not rand:
            if typeN is int:
                N1s_0 = N1s
                assert N1s in [0, -1], '#ERROR: `N1s not accepted'
                if iok1:
                  tidxs = idsOnly + ims[self.nm-nt:]
                if iok2:
                  assert set(idsOut).issubset( set(self.tidxs) )
                  tidxs = list( set(self.tidxs).difference( set(idsOut) ) )

                nas = self.nas[tidxs]
                nhass = self.nhass[tidxs]
                nm = len(tidxs)
                ng = nm - nt
                namax0 = nas[:ng].max() # assume that amons are all ordered by `N_I
                if namax > namax0:
                    namax = namax0
                nis0 = range(namin, namax+1)
                nis = []; N1s = []
                for ni in nis0:
                    N1 = np.logical_and(nhass >= namin, nhass <= ni).sum()
                    if N1 not in N1s and N1 > 0: # it's possible that there r no amons corresp. to some `N_I
                        if len(nheavDisp) > 0:
                            if ni in nheavDisp:
                                N1s.append( N1 )
                                nis.append( ni )
                        else:
                            N1s.append( N1 )
                            nis.append( ni )
                if N1s_0 == -1: # print the E corresponding to the largest N1 __only__; otherwise, print all E vs. N
                  N1s = [ng, ]
                  nis = [ namax, ]
                #print ' __ N1s, nis = ', N1s, nis
                self.nis = nis
                self.N1s = N1s
                self.nm = nm
                sort_ias = True
            else:
                #self.nis = nis
                self.N1s = N1s
        else:
            assert typeN is list, '#ERROR: `N1s is not a list?'
            self.N1s = N1s
            np.random.seed(seed) # fix the random sequence
            tidxs = np.random.permutation(nm)
            nas = self.nas[tidxs]
            sort_ias = True

        nzs = self.nzs[tidxs]

        if sort_ias:
          ias2u = []; ias1u = []
          zsr_u = []
          for i in tidxs:
              ib = ias1[i]; ie = ias2[i]
              ias1u.append( ib ); ias2u.append( ie )
              zsr_u += list( zsr[ib:ie] )
          zsr = np.array(zsr_u )
          ias2 = ias2u
          ias1 = ias1u

        self.nhass = nhass
        self.nas = nas
        self.tidxs = tidxs
        self.zsr = zsr
        self.nzs = nzs
        self.ias1 = ias1
        self.ias2 = ias2

    def ML_atp(self, N1s=0, nt=1, zsnmr=[1,6,7], coeff=1.0, iprt0=True, iprt=False, \
               idsOnly=[], idsOut=[], kernel='g', llambda=1e-4, rand=False, seed=1, \
               namin=1, namax=10000):
        """
        a local ML model for atomic properties (e.g., NMR shift, CLS, charges)

        nt      -- num_target molecule
        idsOnly -- Idxs of amons to be used for training
        """

        self.update_N1s(N1s, nt, rand=rand, idsOnly=idsOnly, idsOut=idsOut, seed=seed, \
                        namin=namin, namax=namax)
        N1s = self.N1s
        if not rand: nis = self.nis
        tidxs = self.tidxs
        nm = self.nm
        ng = nm - nt

        X = self.X
        = self.properties[0].upper()
        Y0 = [ self.dic[propi.upper()] for propi in self.properties ]
        nys = [ len(ysi) for ysi in Y0 ]
        assert np.unique(nys).shape[0] == 1
        for propi in self.properties:
            assert propi.upper() in ['NMR', 'CLS','POPULATION']
        Y = np.array(Y0).T
        self.Y = Y
        zsr = self.zsr
        nhass = self.nhass
        nas = self.nas
        ias2 = self.ias2
        ias1 = self.ias1
        nzs = self.nzs
        Xu = []; Yu = []
        for i in tidxs:
            ib = ias1[i]; ie = ias2[i]
            Xu += list( self.X[ib:ie] )
            Yu += list( self.Y[ib:ie] )
        X = np.array(Xu)
        #print ' -- X = ', np.max(X.ravel())
        #print ' --X[0] = ', [ xx for xx in X[0] ]
        assert not np.any(np.isnan(X)), '#ERROR: `X contains np.nan'
        Y = np.array(Yu)
        #print ' tidxs = ', tidxs

        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        self.maes = []
        self.rmses = []
        self.errors = []
        self.N1s = N1s
        self.ntargets = []
        spr = ','.join( self.properties )
        for z_i in zsnmr:
          if iprt0: print '\n atomic properties of %s: %s'%( ad.chemical_symbols[ z_i ], spr )
          maes_i = []
          rmses_i = []
          errors_i = []
          ntarget = []
          sll = '' # strings of last line to be printed
          fk = 'K_%.2f_na_%d_%d_'%(coeff,namin,namax) + self.h5f
          ks = []
          for i0 in range(2):
            for j,N1 in enumerate(N1s):
              ical = False # continue
              if i0 == 0:
                if N1 == N1s[-1]:
                  ical = True
                else:
                  continue
              else:
                if N1 != N1s[-1]: ical = True
                assert os.path.exists(fk), '#ERROR: no K file?'
                if kdic is None:
                  kdic = dd.io.load(fk); _K1 = kdic['K1']; _K2 = kdic['K2']

              if N1 == 0: continue
              #ntarget = nt if not rand else nm - N1
              ids1 = np.arange( ias2[N1-1] )
              zsr1 = zsr[ ids1 ]
              ids1u = ids1[ zsr1 == z_i ]
              X1 = X[ ids1u ]
              Y1 = Y[ ids1u ]
              ntrain = len(Y1)
              if ntrain <= 1: continue
              ## center the data
              #yc = Y1.sum()/ntrain
              #Y1 -= yc
              it2 = nm
              it1 = ng # the beginning Idx of Target
              ids2 = np.arange(ias1[it1], ias2[it2-1])
              zsr2 = zsr[ ids2 ]
              ids2u = ids2[ zsr2 == z_i ]
              ntarget = len(ids2u)
              X2 = X[ ids2u ]
              Y2 = Y[ ids2u ] #- yc

              if not ical:
                if kernel == 'g':
                  ds = qd.l2_distance(X1,X1) # ssd.pdist(x1, metric='euclidean')
                  dmax = max(ds.ravel())
                  sigma = coeff*dmax/np.sqrt(2.0*np.log(2.0))
                  K1 = qk.gaussian_kernel(X1, X1, sigma)
                  K2 = qk.gaussian_kernel(X2, X1, sigma)
                elif kernel == 'l':
                  ds = qd.manhattan_distance(X1,X1)
                  dmax = max(ds.ravel())
                  sigma = coeff*dmax/np.log(2.0)
                  K1 = qk.laplacian_kernel(X1, X1, sigma)
                  K2 = qk.laplacian_kernel(X2, X1, sigma)
              else:
                K1 = _K1[ids1u,:][:,ids1u]
                K2 = _K2[:,ids2u]
              #K1 = (K1 + K1.T)/2 #
              #print ' -- size(K1) = ', K1.shape
              #print ' -- sigma = ', sigma
              #print ' -- K1 = ', K1[:12,:12]
              assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"

              # Solve alpha
              K1[np.diag_indices_from(K1)] += llambda
              if iprt: print ' -- now get alphas ', time.ctime()
              #alpha = np.array([ cho_solve(K1,np.squeeze(Y1)) ]).T
              alpha = np.linalg.solve(K1,Y1).T

              y2est = np.dot(K2, alpha)

              if iprt:
                  print ' -- Y2 = '
                  print Y2

              y2 = Y2
              if iprt: print ' --Y2_est = ', y2est
              if iprt: print ' -- Y2 = ', y2
              error = y2est - y2; #print ' -- error = ', error

              >>mae = np.sum(np.abs(error))/ntarget
              rmse = np.sqrt( np.sum( error**2 )/ntarget )
              if not rand:
                  if iprt0: print ' %4d %4d %6d %6d %12.6f %12.6f'%(nis[j], N1, ntrain, ntarget, mae, rmse)
                  if iprt:
                      for l in range(len(Y2)): print ' %8.2f %8.2f'%(y2est[l],y2[l])
              else:
                  if iprt0: print ' %4d %6d %6d %9.4f %9.4f'%(N1, ntrain, ntarget, mae, rmse)
              maes_i.append( mae )
              rmses_i.append( rmse )
              errors_i.append( error )

          self.maes.append( maes_i )
          self.rmses.append( rmses_i )
          self.errors.append( errors_i )
          self.ntargets.append( ntarget )

        #mae_av = np.sum(np.array(self.

    def ML_force(self, iat, it, iaxis, N1s=0, nt=1, coeff=1.0, iprt0=True, iprt=False, \
               idsOnly=[], idsOut=[], kernel='g', llambda=1e-4, rand=False, \
               seed=1, namin=1, namax=10000):
        """
        a local ML model for force prediction

        one atom in one target molecule per time !!
        The idxs of these atoms are stored in `iast

        nt      -- num_target molecule

        """

        self.update_N1s(N1s, nt, rand=rand, idsOnly=idsOnly, idsOut=idsOut, seed=seed, \
                        namin=namin, namax=namax)
        N1s = self.N1s
        #print ' -- N1s = ', N1s
        if not rand: nis = self.nis
        tidxs = self.tidxs
        nm = self.nm
        ng = nm - nt

        X = self.X
        propi = self.properties[0]
        Y0 = self.dic[ propi ]; Y = []

        if propi in ['FORCE','force']:
            for Yi in Y0:
                Y += [list(Yi)] # = np.concatenate(Y0, axis=0)
        else:
            raise '#ERROR:'
        #print ' Y = ', np.array(Y)
        Y = np.array(Y)
        self.Y = Y[:,iaxis]
        zsr = self.zsr
        nhass = self.nhass
        nas = self.nas
        ias2 = self.ias2
        ias1 = self.ias1
        nzs = self.nzs
        Xu = []; Yu = []
        for i in tidxs:
            ib = ias1[i]; ie = ias2[i]
            Xu += list( self.X[ib:ie] )
            Yu += list( self.Y[ib:ie] )
        X = np.array(Xu)
        #print ' -- Y.shape = ', np.array(self.Y).shape # np.max(X.ravel())
        #print ' --X[0] = ', [ xx for xx in X[0] ]
        assert not np.any(np.isnan(X)), '#ERROR: `X contains np.nan'
        Y = np.array(Yu)
        #print ' tidxs = ', tidxs

        ias2 = np.cumsum(nas)
        ias1 = np.concatenate(([0,],ias2[:-1]))

        self.maes = []
        self.rmses = []
        self.errors = []
        self.N1s = N1s
        self.ntargets = []

        #print ' -- coords: ', np.array(self.coords).shape
        #print ' -- iat, it, idx = ', iat, it, iat + ias1[ng+it]
        coords_iat = self.coords[ iat+ias1[ng+it] ]
        print ' -- coords_iat = ', coords_iat
        #visited_coords = np.array(self.visited_coordinates)
        print ' -- visited_coords = ', self.visited_coordinates
        #if len(visited_coords) > 0:
        #    if np.any( ssd.cdist([coords_iat],  np.array(visited_coords))[0] <= 1e-4 ):
        #        return
        #    else:
        #        self.visited_coordinates.append( coords_iat )
        #else:
        #    self.visited_coordinates.append( coords_iat )

        maes_i = []
        rmses_i = []
        errors_i = []
        ntarget = []
        #print ' -- N1s = ', N1s
        for j,N1 in enumerate(N1s):
            if N1 == 0:
                print ' ** zero training instances?'
                continue
            #print ' -- n1 = ', N1
            #ntarget = nt if not rand else nm - N1
            ids = np.arange( ias2[N1-1] )
            dsi = ssd.cdist( [coords_iat], self.coords[ids] )[0]
            #print ' dsi = ', dsi
            idsu = ids[ dsi <= 0.0001 ]; print ' -- idsu_traing = ', idsu

            X1 = X[ idsu ]
            Y1 = Y[ idsu ]
            ntrain = len(Y1)
            if ntrain <= 1: continue
            ## center the data
            #yc = Y1.sum()/ntrain
            #Y1 -= yc
            it2 = nm
            it1 = ng # the beginning Idx of Target
            ids = np.arange(ias1[it1],ias2[it2-1])
            dsi = ssd.cdist( [coords_iat], self.coords[ids] )[0]
            idsu = ids[ dsi <= 0.0001 ]; #print ' -- idsu_test = ', idsu

            ntarget = len(idsu)
            X2 = X[ idsu ]
            #print ' Y.shape = ', np.array(Y).shape
            Y2 = Y[ idsu ] #- yc

            if kernel == 'g':
                ds = qd.l2_distance(X1,X1) # ssd.pdist(x1, metric='euclidean')
                dmax = max(ds.ravel())
                sigma = coeff*dmax/np.sqrt(2.0*np.log(2.0))
                K1 = qk.gaussian_kernel(X1, X1, sigma)
            elif kernel == 'l':
                ds = qd.manhattan_distance(X1,X1)
                dmax = max(ds.ravel())
                sigma = coeff*dmax/np.log(2.0)
                K1 = qk.laplacian_kernel(X1, X1, sigma)
            #K1 = (K1 + K1.T)/2 #
            #print ' -- size(K1) = ', K1.shape
            #print ' -- sigma = ', sigma
            #print ' -- K1 = ', K1[:12,:12]
            assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"

            # Solve alpha
            K1[np.diag_indices_from(K1)] += llambda
            if iprt: print ' -- now get alphas ', time.ctime()
            #alpha = np.array([ cho_solve(K1,np.squeeze(Y1)) ]).T
            alpha = np.linalg.solve(K1,np.squeeze(Y1)).T

            # Calculate prediction kernel
            if iprt: print ' -- now get prediction kernel'
            if kernel == 'g':
                K2 = qk.gaussian_kernel(X2, X1, sigma)
            elif kernel == 'l':
                K2 = qk.laplacian_kernel(X2, X1, sigma)
            Y2_est = np.dot(K2, alpha)

            if iprt:
                print ' -- Y2 = '
                print Y2

            y2est = np.squeeze(Y2_est.T)
            y2 = np.squeeze(Y2)
            if iprt: print ' --Y2_est = ', y2est
            if iprt: print ' -- Y2 = ', y2
            error = y2est - y2; #print ' -- error = ', error
            mae = np.sum(np.abs(error))/ntarget
            rmse = np.sqrt( np.sum( error**2 )/ntarget )
            if not rand:
                if iprt0: print ' %4d %4d %6d %6d %9.4f %9.4f'%(nis[j], N1, ntrain, ntarget, mae, rmse)
                if iprt:
                    for l in range(len(Y2)): print ' %8.2f %8.2f'%(y2est[l],y2[l])
            else:
                if iprt0: print ' %4d %6d %6d %9.4f %9.4f'%(N1, ntrain, ntarget, mae, rmse)
            maes_i.append( mae )
            rmses_i.append( rmse )
            errors_i.append( error )

        self.maes.append( maes_i )
        self.rmses.append( rmses_i )
        self.errors.append( errors_i )
        self.ntargets.append( ntarget )

    def ML_global(self, N1s, nt=None, usebl=True, coeff=1.0, iprt=False, seed=1, \
                  rand=False, idsOnly=[], kernel='g', llambda = 1e-4, idsOut=[], \
                  namin=1, namax=10000):
        """
        train the local ML model
        """
        nm = self.nm
        self.update_N1s(N1s, nt, rand=rand, idsOnly=idsOnly, idsOut=idsOut, seed=seed, \
                        namin=namin, namax=namax)
        N1s = self.N1s
        tidxs = self.tidxs

        Y0 = np.array( self.dic[self.properties[0] ] ); #print ' __ Y0 = ', Y0

        nas = self.nas[tidxs]
        #nes = self.nes[tidxs]
        nzs = self.nzs[tidxs]
        Y = Y0[tidxs]
        X = self.X[tidxs]
        self.maes = []
        self.rmses = []
        self.N1s = N1s

        if nt != None:
            X2 = X[nm-nt:]; nzs2 = nzs[nm-nt:];
            Y2 = Y[nm-nt:]; ntarget = nt

        for N1 in N1s:
            # fix test set, i.e., `ntarget is fixed if `nt != None
            if nt is None: # treat all samples untrained as test
                nt = nm - N1
                X2 = X[nm-nt:]; nzs2 = nzs[nm-nt:];
                Y2 = Y[nm-nt:]; ntarget = nt

            X1 = X[:N1]
            Y1 = Y[:N1]
            nzs1 = nzs[:N1]
            #llambda = 10**(-4) if kernel == 'g' else 10**(-8)
            if kernel == 'g':
                ds = qd.l2_distance(X1,X1) # ssd.pdist(x1, metric='euclidean')
                dmax = max(ds.ravel())
                sigma = coeff*dmax/np.sqrt(2.0*np.log(2.0))
                K1 = qk.gaussian_kernel(X1, X1, sigma)
            elif kernel == 'l':
                ds = qd.manhattan_distance(X1,X1)
                dmax = max(ds.ravel())
                sigma = coeff*dmax/np.log(2.0)
                K1 = qk.laplacian_kernel(X1, X1, sigma)
            self.K1 = K1
            assert np.allclose(K1, K1.T), "Error in local Gaussian kernel symmetry"
            if usebl:
                esb = np.linalg.lstsq(nzs1,Y1)[0]
            else:
                esb = np.zeros(nzs1.shape[1])
            #print ' -- dmax = ', dmax
            Y1 = Y1 - np.dot(nzs1,esb)
            Y2 = Y2 - np.dot(nzs2,esb)
            if iprt:
                print ' -- esb = '
                print esb
                print ' -- Y2 - baseline = '
                print Y2

            # Solve alpha
            K1[np.diag_indices_from(K1)] += llambda
            if iprt: print ' -- now get alphas ', time.ctime()
            alpha = np.array([ cho_solve(K1,np.squeeze(Y1)) ]).T

            # Calculate prediction kernel
            if iprt: print ' -- now get prediction kernel'
            if kernel == 'g':
                K2 = qk.gaussian_kernel(X2, X1, sigma)
            elif kernel == 'l':
                K2 = qk.laplacian_kernel(X2, X1, sigma)
            Y2_est = np.dot(K2, alpha)
            error = np.squeeze(Y2_est) - np.squeeze(Y2)
            mae = np.sum(np.abs(error))/ntarget
            rmse = np.sqrt( np.sum( error**2 )/ntarget )
            self.maes.append( mae )
            self.rmses.append( rmse )
            print ' %4d %9.4f %9.4f'%(N1, mae, rmse)

    def get_slatm_mbtypes(self, iBoA=True):
        """
        Get the list of minimal types of many-body terms in a dataset. This resulting list
        is necessary as input in the ``generate_slatm_representation()`` function.

        :param nuclear_charges: A list of the nuclear charges for each compound in the dataset.
        :type nuclear_charges: list of numpy arrays
        :param pbc: periodic boundary condition along x,y,z direction, defaulted to '000', i.e., molecule
        :type pbc: string
        :return: A list containing the types of many-body terms.
        :rtype: list
        """
        zsu = np.array( self.zsu )
        nass = []
        for i in range(self.nm):
            zsi = np.array(self.zs[i], np.int)
            nass.append( [ (zi == zsi).sum() for zi in zsu ] )
        nzmax = np.max(np.array(nass), axis=0)
        nzmax_u = []
        if self.pbc != '000':
            # the PBC will introduce new many-body terms, so set
            # nzmax to 3 if it's less than 3
            for nzi in nzmax:
                if nzi <= 2:
                    nzi = 3
                nzmax_u.append(nzi)
            nzmax = nzmax_u
        boas = [ [zi,] for zi in zsu ] if iBoA else []
        bops = [ [zi,zi] for zi in zsu ]
        for ci in itl.combinations(zsu,2): bops += [ list(ci), ]
        bots = []
        for i in zsu:
            for bop in bops:
                j,k = bop
                tas = [ [i,j,k], [i,k,j], [j,i,k] ]
                for tasi in tas:
                    if (tasi not in bots) and (tasi[::-1] not in bots):
                        nzsi = [ (zj == tasi).sum() for zj in zsu ]
                        if np.all(nzsi <= nzmax):
                            bots.append( tasi )
        if self.nbody == 2:
            mbtypes = boas + bops
        elif self.nbody == 3:
            mbtypes = boas + bops + bots
        else:
            raise '#ERROR: at most 3-body'
        self.mbtypes = mbtypes

    def generate_slatm(self, i):
        c = self.cell
        iprt = False
        if c is None:
            c = np.array([[1,0,0],[0,1,0],[0,0,1]])
        if self.pbc != '000':
            # print(' -- handling systems with periodic boundary condition')
            assert c != None, 'ERROR: Please specify unit cell for SLATM'
            # =======================================================================
            # PBC may introduce new many-body terms, so at the stage of get statistics
            # info from db, we've already considered this point by letting maximal number
            # of nuclear charges being 3.
            # =======================================================================
        zs = np.array(self.zs[i])
        na = len(zs)
        ib = self.ias1[i]; ie = self.ias2[i]
        coords = self.coords[ib:ie]
        dsr = ssd.pdist(coords)
        ds = ssd.squareform( dsr )
        d0 = 0.45
        assert dsr.min() >= d0, '#ERROR: atoms are too close'
        #print ' -- min(ds) = ', np.min(dsr)
        obj = [ zs, coords, c, ds ]
        ntypes = np.array([ len(mbtype) for mbtype in self.mbtypes ])
        n1 = (ntypes == 1 ).sum()
        zsu = self.zsu

        ias = np.arange(na)
        yfilter = self.yfilter
        if yfilter is not None:
            filt = ( self.ys[0][ib:ie] < yfilter )
            ias = ias[ filt ]
        #print ' -- ib,ie, ys = ', ib,ie,self.ys

        if self.iBE:
            cgo = cg.graph(zs, coords)
            cgo.perceive_connectivity()
            cliques = cg.find_cliques(cgo.g)

        if self.local:
            mbs = []
            X2Ns = []
            #print ' -- mbtypes = ', self.mbtypes; sys.exit(2)
            for ia in ias: #range(na):
                #print ' - ia = ', ia+1

                if self.iBE:
                    for cliques_i in cliques:
                        if ia in cliques_i:
                            ias_c = np.array(cliques_i, np.int)
                            break
                    ias_other = np.setdiff1d(ias, ias_c)
                    if self.icomb == 1:
                        # get the subg consisting of node `ia
                        ias_U = ias_c
                    elif self.icomb == 2:
                        # get the subg where `ia is not a node
                        ias_U = np.concatenate(([ia,], ias_other))
                    else:
                        raise '#ERROR: invalid `icomb'
                    obj = [ zs[ias_U], coords[ias_U], c, ds[ias_U,:][:,ias_U] ]

                n2 = 0; n3 = 0
                mbs_ia = np.zeros(0)
                icount = 0
                for mbtype in self.mbtypes:
                    #print '  |__ mbtype = ', mbtype
                    if len(mbtype) == 1:
                        #print '   |'
                        #print '   |__ now 1-body'
                        if self.isf in [0,1]:
                            mbsi = self.get_boa(mbtype[0], [zs[ia],])
                        elif self.isf in [2,3]:
                            mbsi = fget_sboa_local( mbtype[0], zs, zsu, ia, ds, self.cc, self.cd )

                        mbsi = np.array(mbsi)*self.ws[0]
                        if self.alchemy:
                            n1 = 1
                            n1_0 = mbs_ia.shape[0]
                            if n1_0 == 0:
                                mbs_ia = np.concatenate( (mbs_ia, [mbsi.sum()]), axis=0 )
                            elif n1_0 == 1:
                                mbs_ia += mbsi.sum()
                            else:
                                raise '#ERROR'
                        else:
                            if self.isf in [0,1]:
                                mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                            else: # in this case, `mbsi is a vector
                                if mbs_ia.shape[0] == 0:
                                    mbs_ia = mbsi
                                else:
                                    mbs_ia += mbsi
                    elif len(mbtype) == 2:
                        #print '   |'
                        #print '   |__ now 2-body'
                        mbsi = self.get_sbop(mbtype, obj, local=self.local, ia=ia, \
                                    sigma=self.sigmas[0], dgrid=self.dgrids[0], \
                                    rcut=self.rcut, pbc=self.pbc, rpower2=self.rpower2, \
                                    isf=self.isf, moment=self.moment, \
                                    isqrt_dgrid=self.isqrt_dgrid, \
                                    cc=self.cc, cd=self.cd, ico=self.ico, \
                                    idc=self.idc)
                        mbsi *= self.ws[1] #0.5 # only for the two-body parts, local rpst
                        if self.alchemy:
                            n2 = len(mbsi)
                            n2_0 = mbs_ia.shape[0]
                            if n2_0 == n1:
                                mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                            elif n2_0 == n1 + n2:
                                t = mbs_ia[n1:n1+n2] + mbsi
                                mbs_ia[n1:n1+n2] = t
                            else:
                                raise '#ERROR'
                        else:
                            n2 += len(mbsi)
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                    else: # len(mbtype) == 3:
                        #print '   |__ now 3-body'
                        mbsi = self.get_sbot(mbtype, obj, local=self.local, ia=ia, \
                                        sigma=self.sigmas[1], dgrid=self.dgrids[1], \
                                        rcut=self.rcut, rpower3=self.rpower3, isf=self.isf, \
                                        intc=self.intc, isqrt_dgrid=self.isqrt_dgrid, \
                                        pbc=self.pbc, moment=self.moment, \
                                        cc=self.cc, cd=self.cd, ico=self.ico, \
                                        idc=self.idc)
                        mbsi *= self.ws[2]
                        if self.alchemy:
                            n3 = len(mbsi)
                            n3_0 = mbs_ia.shape[0]
                            if n3_0 == n1 + n2:
                                mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                            elif n3_0 == n1 + n2 + n3:
                                t = mbs_ia[n1+n2:n1+n2+n3] + mbsi
                                mbs_ia[n1+n2:n1+n2+n3] = t
                            else:
                                raise '#ERROR'
                        else:
                            n3 += len(mbsi)
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )

                if self.moment == 0:
                    mbs.append( mbs_ia )
                else:
                    mbs.append( [ sum(mbs_ia), ] )
                X2N = [n1,n2,n3];
                if X2N not in X2Ns:
                    X2Ns.append(X2N)
            assert len(X2Ns) == 1, '#ERROR: multiple `X2N ???'
            self.X2N = X2N
            #print ' -- x2n = ', X2N; print ' -- dgrids = ', self.dgrids; sys.exit(2)
        else:
            n1 = 0; n2 = 0; n3 = 0
            mbs = np.zeros(0)
            for mbtype in self.mbtypes:
                if len(mbtype) == 1:
                    mbsi = np.array( self.get_boa(mbtype[0], zs) ).astype(np.float)
                    mbsi *= self.ws[0]
                    if self.alchemy:
                        n1 = 1
                        n1_0 = mbs.shape[0]
                        if n1_0 == 0:
                            mbs = np.concatenate( (mbs, [sum(mbsi)] ), axis=0 )
                        elif n1_0 == 1:
                            mbs += sum(mbsi )
                        else:
                            raise '#ERROR'
                    else:
                        n1 += len(mbsi)
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                elif len(mbtype) == 2:
                    mbsi = self.get_sbop(mbtype, obj, sigma=self.sigmas[0], \
                                    dgrid=self.dgrids[0], rcut=self.rcut, \
                                    rpower2=self.rpower2, isf=self.isf, \
                                    isqrt_dgrid=self.isqrt_dgrid)
                    mbsi *= self.ws[1]
                    if self.alchemy:
                        n2 = len(mbsi)
                        n2_0 = mbs.shape[0]
                        if n2_0 == n1:
                            mbs = np.concatenate( (mbs, mbsi), axis=0 )
                        elif n2_0 == n1 + n2:
                            t = mbs[n1:n1+n2] + mbsi
                            mbs[n1:n1+n2] = t
                        else:
                            raise '#ERROR'
                    else:
                        n2 += len(mbsi)
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                else: # len(mbtype) == 3:
                    mbsi = self.get_sbot(mbtype, obj, sigma=self.sigmas[1], \
                            rpower3=self.rpower3, dgrid=self.dgrids[1], \
                            rcut=self.rcut, isf=self.isf, isqrt_dgrid=self.isqrt_dgrid, \
                            intc=self.intc)
                    mbsi *= self.ws[2]
                    if self.alchemy:
                        n3 = len(mbsi)
                        n3_0 = mbs.shape[0]
                        if n3_0 == n1 + n2:
                            mbs = np.concatenate( (mbs, mbsi), axis=0 )
                        elif n3_0 == n1 + n2 + n3:
                            t = mbs[n1+n2:n1+n2+n3] + mbsi
                            mbs[n1+n2:n1+n2+n3] = t
                        else:
                            raise '#ERROR'
                    else:
                        n3 += len(mbsi)
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
            X2N = [n1,n2,n3]
            #print ' -- x2n = ', X2N
            self.x2n = X2N
        return mbs


    def get_slatm_atomic_distance(self, n1, n2):
        """
        To save memory & speedup ML for gigantic systems,
        we can calculate & store kernel matrix element directly
        without writing repr to a h5 file

        var's
        ================
        n1    -- training set size, within the range [1, self.nm - n2]
        n2    -- test set size
        """

#        df = self.wds[0] + '/distance.h5'
#        if os.path.exists( df ):
#            dic = dd.io.load( df )
#            nas1 = dic['nas1']
#            nas2 = dic['nas2']
#            ds1 = dic['ds1']
#            ds2 = dic['ds2']
        if True: #else:
            n1t = self.nm - n2

            ds11 = []; nas1 = []
            ds12 = []
            ds13 = []
            for i in range(n1):
                print ' -- i = ', i+1
                fxi = self.wds[0] + '/%04d.h5'%(i+1)
                if os.path.exists(fxi):
                    xi = dd.io.load(fxi)['x']
                else:
                    xi = np.array( self.generate_slatm(i) )
                    dd.io.save(fxi, {'x':xi})
                nai = xi.shape[0]
                nas1.append( nai )
                ds11.append( qd.l2_distance(xi,xi) )

                for j in range(i+1,n1):
                    print '   |_ j = ', j+1
                    fxj = self.wds[0] + '/%04d.h5'%(j+1)
                    if os.path.exists(fxj):
                        xj = dd.io.load(fxj)['x']
                    else:
                        xj = np.array( self.generate_slatm(j) )
                        dd.io.save(fxj, {'x':xj})
                    ds12.append( qd.l2_distance(xi,xj) )

                nas2 = []
                for k in range(n1t,self.nm):
                    print '   |_ k = ', k+1
                    fxk = self.wds[0] + '/%04d.h5'%(k+1)
                    if os.path.exists(fxk):
                        xk = dd.io.load(fxk)['x']
                    else:
                        xk = np.array( self.generate_slatm(k) )
                        dd.io.save(fxk, {'x':xk})
                    nas2.append( xk.shape[0] )
                    ds13.append( qd.l2_distance(xi,xk) )

            nas1 = np.array(nas1); nas2 = np.array(nas2)
            nat1 = nas1.sum(); nat2 = nas2.sum()
            ias2 = np.cumsum(nas1); ias1 = np.zeros(n1).astype(np.int); ias1[1:] = ias2[:-1]
            kas2 = np.cumsum(nas2); kas1 = np.zeros(n2).astype(np.int); kas1[1:] = kas2[:-1]
            ds1 = np.empty((nat1,nat1))
            ds2 = np.empty((nat1,nat2))
            icnt = 0; kcnt = 0
            for i in range(n1):
                ds1[ias1[i]:ias2[i], ias1[i]:ias2[i]] = ds11[i]
                for j in range(i+1,n1):
                    ds1[ias1[i]:ias2[i], ias1[j]:ias2[j]] = ds12[icnt]
                    ds1[ias1[j]:ias2[j], ias1[i]:ias2[i]] = ds12[icnt].T
                    icnt += 1
                for k in range(n1t,self.nm):
                    ds2[ias1[i]:ias2[i], kas1[k-n1t]:kas2[k-n1t]] = ds13[kcnt]
                    kcnt += 1
        #    dic = {'nas1':nas1, 'nas2':nas2, 'ds1':ds1, 'ds2':ds2}
        #    dd.io.save(df, dic)

        self.ds1 = ds1
        self.ds2 = ds2
        self.nas1 = nas1
        self.nas2 = nas2


    def get_slatm_kernel(self, iap, coeff):
        """

        var's
        ================================
        coeff -- to be multiplied to `dmax so as to obtain the optimal `sigma.
                 It can be set to either a float value or a list of values
        iap   -- is atomic property
        """

        ds1 = self.ds1
        ds2 = self.ds2
        nas1 = self.nas1
        nas2 = self.nas2

        dmax = np.max(ds1)
        # now calculate kernel matrix
        if self.kernel in ['g','gaussian','G']:
            sigma = dmax/np.sqrt(2.0*np.log(2.0))
        elif self.kernel in ['l','laplacian','L']:
            sigma = dmax/np.log(2.0)

        if type(coeff) in [int, float]:
            sigmas = np.array([coeff*sigma,])
        else:
            sigmas = np.array([ coeff_i * sigma for coeff_i in coeff ])

        if iap: # Is Atomic Property?
            K1 = np.exp( - ds1*ds1/(2.*sigmas[0]**2) )
            K2 = np.exp( - ds2*ds2/(2.*sigmas[0]**2) )
        else:
            K1 = qk.fget_kernel_from_dist(ds1, nas1, nas1, sigmas)[0]
            K2 = qk.fget_kernel_from_dist(ds2, nas1, nas2, sigmas)[0]
        self.K1 = K1
        self.K2 = K2


    def get_boa(self, z1, zs):
        t = z1*np.array( [(np.array(zs) == z1).sum(), ])
        #t = -0.5*z1**2.4*np.array( [(zs_ == z1).sum(), ])
        return t

    def get_sbop(self, mbtype, obj, local=False, ia=None, normalize=True, sigma=0.05, \
                 rcut=4.8, dgrid=0.03, isqrt_dgrid=False, pbc='000', rpower2=6, isf=0, moment=0, \
                 cc=None, cd=None, ico=None, idc=None):
        """
        two-body terms

        :param obj: molecule object, consisting of two parts: [ zs, coords ]
        :type obj: list
        :param Rref: reference R, has to be optimized for vdW only interactions;
                     defaulted to 0 for covalent bonds
        :type Rref: float
        """

        z1, z2 = mbtype
        zs, coords, c, ds = obj
        if local:
            assert ia != None, '#ERROR: plz specify `za and `ia '
        if pbc != '000':
            if rcut < 9.0: raise '#ERROR: rcut too small for systems with pbc'
            assert local, '#ERROR: for periodic system, plz use atomic rpst'
            zs, coords = self.update_m(obj[:3], ia, rcut=rcut, pbc=pbc)
            ds = ssd.squareform( ssd.pdist(coords) )
            # after update of `m, the query atom `ia will become the first atom
            ia = 0

        # bop potential distribution
        r0 = 0.1
        nx = int((rcut - r0)/dgrid) + 1
        coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

        dgrid_u = dgrid
        if isqrt_dgrid: dgrid_u = np.sqrt(dgrid)
        if local:
            v2l = fget_sbop_local if moment == 0 else fget_sbop_local_moment
            #ys = v2l(coords, zs, ia, z1, z2, rcut, nx, dgrid_u, sigma, coeff, rpower2, isf) #, cc, cd, ico, idc)
            ys = v2l(coords, zs, ia, z1, z2, rcut, nx, dgrid_u, sigma, coeff, rpower2) #, isf) #, cc, cd, ico, idc)
        else:
            #ys = fget_sbop(coords, zs, z1, z2, rcut, nx, dgrid_u, sigma, coeff, rpower2) #, isf) #, Rref)
            ys = fget_sbop(coords, zs, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower2) #, isf) #, Rref)
        #print ' -------- bop '
        if np.any( ys > 1000. ):
            print '  |_ ia = ', ia, ' mbtype = ', mbtype, ', np.max(ys) = ', np.max(ys)

        return ys

    def get_sbot(self, mbtype, obj, local=False, ia=None, normalize=True, sigma=0.05, \
                 rcut=4.8, dgrid=0.0262, isqrt_dgrid=False, rpower3=3, isf=0, pbc='000', moment=0, \
                 intc=3, cc=None, cd=None, ico=None, idc=None):

        """
        sigma -- standard deviation of gaussian distribution centered on a specific angle
                defaults to 0.05 (rad), approximately 3 degree
        dgrid    -- step of angle grid
                defaults to 0.0262 (rad), approximately 1.5 degree
        """
        z1, z2, z3 = mbtype
        zs, coords, c, ds = obj
        if local:
            assert ia != None, '#ERROR: plz specify `za and `ia '
        if pbc != '000':
            assert local, '#ERROR: for periodic system, plz use atomic rpst'
            zs, coords = self.update_m(obj[:3], ia, rcut=rcut, pbc=pbc)
            # after update of `m, the query atom `ia will become the first atom
            ds = ssd.squareform( ssd.pdist(coords) )
            ia = 0

        # for a normalized gaussian distribution, u should multiply this coeff
        coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0
        # Setup grid in Python
        d2r = np.pi/180 # degree to rad
        a0 = -20.0*d2r
        a1 = np.pi + 20.0*d2r
        nx = int((a1-a0)/dgrid) + 1

        dgrid_u = dgrid
        if isqrt_dgrid: dgrid_u = np.sqrt(dgrid)
        if local:
            v2l = fget_sbot_local if moment == 0 else fget_sbot_local_moment
            #ys = v2l(coords, zs, ia, z1, z2, z3, rcut, nx, dgrid_u, sigma, coeff, rpower3, isf, intc) #, cc, cd, ico, idc)
            ys = v2l(coords, zs, ia, z1, z2, z3, rcut, nx, dgrid_u, sigma, coeff, rpower3)
        else:
            #ys = fget_sbot(coords, zs, z1, z2, z3, rcut, nx, dgrid_u, sigma, coeff, rpower3, isf, intc)
            ys = fget_sbot(coords, zs, z1, z2, z3, rcut, nx, dgrid_u, sigma, coeff, rpower3) #, isf, intc)
        #print ' -------- bot '
        if np.any( ys > 1000. ):
            print '  |_ ia = ', ia, ' mbtype = ', mbtype, ', np.max(ys) = ', np.max(ys)
        return ys

    def update_m(self, obj, ia, rcut=9.0, pbc=None):
        """
        retrieve local structure around atom `ia
        for periodic systems (or very large system)
        """
        zs, coords, c = obj
        v1, v2, v3 = c
        vs = ssd.norm(c, axis=0)
        nns = []
        for i,vi in enumerate(vs):
            n1_doulbe = rcut/li
            n1 = int(n1_doulbe)
            if n1 - n1_doulbe == 0:
                n1s = range(-n1, n1+1) if pbc[i] else [0,]
            elif n1 == 0:
                n1s = [-1,0,1] if pbc[i] else [0,]
            else:
                n1s = range(-n1-1, n1+2) if pbc[i] else [0,]
            nns.append(n1s)

        n1s,n2s,n3s = nns
        n123s_ = np.array( list( itl.product(n1s,n2s,n3s) ) )
        n123s = []
        for n123 in n123s_:
            n123u = list(n123)
            if n123u != [0,0,0]: n123s.append(n123u)
        nau = len(n123s)
        n123s = np.array(n123s, np.float)
        na = len(zs)
        cia = coords[ia]
        if na == 1:
            ds = np.array([[0.]])
        else:
            ds = ssd.squareform( ssd.pdist(coords) )

        zs_u = []; coords_u = []
        zs_u.append( zs[ia] ); coords_u.append( coords[ia] )
        for i in range(na) :
            di = ds[i,ia]
            if (di > 0) and (di <= rcut):
                zs_u.append(zs[i]); coords_u.append(coords[ia])
    # add new coords by translation
                ts = np.zeros((nau,3))
                for iau in range(nau):
                    ts[iau] = np.dot(n123s[iau],c)

                coords_iu = coords[i] + ts #np.dot(n123s, c)
                dsi = ssd.norm(coords_iu - cia, axis=1);
                filt = np.logical_and(dsi > 0, dsi <= rcut); nx = filt.sum()
                zs_u += [zs[i],]*nx
                coords_u += [ list( coords_iu[filt,:] ), ]
        obj_u = [zs_u, coords_u]
        return obj_u

