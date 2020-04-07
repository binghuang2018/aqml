#!/usr/bin/env python

import numpy as np
from pyscf import lib, gto, scf, dft, cc, ao2mo
from aqml.cheminfo.core import *
#from aqml.cheminfo.molecule.core import *
from aqml.cheminfo.lo.rotate import *
from ase import Atoms
import ase.io as aio
import os, sys, io2
import scipy.spatial.distance as ssd
#import torch
from functools import reduce

home = os.environ['HOME']
np.set_printoptions(precision=3,suppress=True)

T, F = True, False

UN = io2.Units()
h2e = UN.h2e
h2kc = UN.h2kc

class density_matrix(object):

    def __init__(self, m, ihao=T, output=None, basis='sto-3g', \
                 meth='b3lyp', spin=0, verbose=3, iprt=False):
        """
        get dm1
        """
        self.meth = meth
        self.iprt = iprt
        self.zs = m.zs
        self.coords = m.coords
        self.ds = ssd.squareform( ssd.pdist(coords) )

        na = len(zs)
        self.na = na
        obj = config_adapted_hao(zs, coords, basis=basis, ihao=ihao)
        self.obj = obj
        obj.run()
        self.nbf = obj.nbf
        B = obj.t
        self.B = B #print ' * B = ', B
        self.aoidxs = obj.aoidxs

        # pyscf run
        m = obj.mol
        m.verbose = verbose
        if output != None: m.output = output
        self.m = m
        s = m.intor_symmetric('int1e_ovlp')
        s1 = reduce(np.dot, (B.T,s,B))
        self.s = s
        self.s1 = s1
        #print ' * s2 = ', s2
        self.units = ['', 'Debye', 'kcal/mol', 'eV', 'eV']
        self.prop_names = ['n','dip','e','homo','lumo']

        self.idft = F
        if meth in ['b3lyp']:
            self.scf = dft.RKS
            self.idft = T
        else:
            self.scf = scf.RHF

    def localize_dm(self, _dm, rc_dm=4.2):
        """ set block of dm corresponding to d(I-J) > rc_dm """
        ds = self.ds #ssd.squareform( ssd.pdist(self.coords) )
        dm = _dm.copy()
        for i in range(self.na):
            for j in range(i):
                if ds[i,j] >= rc_dm:
                    iaos, jaos = self.aoidxs[i], self.aoidxs[j]
                    ib, ie = iaos[0], iaos[-1]+1
                    jb, je = jaos[0], jaos[-1]+1
                    dm[ib:ie,jb:je] = 0.; dm[jb:je,ib:ie] = 0. # Note that `dm[iaos][:,jaos] = 0.0` does not work!!!
        return dm

    def feed_new_dm(self, _dm, hao=F):
        """
        hao: T indicates the input _dm uses `hybridized AOs as basis
        """
        if hao:
            dm = reduce(np.dot, (self.B, _dm, self.B.T))
        else:
            dm = _dm
        mf = self.scf(self.m)
        if self.idft: mf.xc = self.meth
        e = mf.energy_tot(dm=dm)
        n = np.einsum('ij,ji', self.s, dm);
        dip = np.linalg.norm( mf.dip_moment(dm=dm) )
        #else:
        #    h1e = mf.get_hcore()
        #    vhf = mf.get_veff(dm)
        #    e = np.einsum('ij,ji', h1e, dm) + np.einsum('ij,ji', vhf, dm) * .5 + mf.energy_nuc()
        nocc = int(np.sum(self.zs)/2) # for Restricted case only!!
        fock = mf.get_fock(dm=dm)
        es_mo = mf.eig(fock, self.s)[0] # HCS = ES
        homo = es_mo[nocc-1]; lumo = es_mo[nocc]
        #homo, lumo = [0., 0.]
        return np.array([n,dip,e*h2kc,homo*h2e,lumo*h2e])

    def calc_properties(self):
        """
        calculate molecular properties after scf
        These properties will be used as reference values for test
        (for larger molecules of course)
        """
        mf = self.mf
        occ = mf.mo_occ
        nocc = (occ>0).sum()
        self.nocc = nocc
        er = mf.e_tot
        e_mos = mf.mo_energy
        dip_r = np.linalg.norm( mf.dip_moment() )
        dm = mf.make_rdm1(mf.mo_coeff)
        ner = np.trace( np.dot(dm,self.s))
        props_r = np.array([ ner, dip_r, er*h2kc, e_mos[nocc-1]*h2e, e_mos[nocc]*h2e ])
        #e = mf.energy_tot(dm=dm1r)
        #if self.meth in ['hf']:
        #    _h1e = mf.get_hcore()
        #    _vhf = mf.get_veff(m, dm1r)
        #    h1e = reduce(np.dot, (B.T, _h1e, B))
        #    vhf = reduce(np.dot, (B.T, _vhf, B))
        #    e = ( np.einsum('ij,ji', h1e, dm1) + np.einsum('ij,ji', vhf, dm1) * .5 + mf.energy_nuc() )
        #    print ' ** er, e = %.6f, %.6f'%(er*h2kc,e*h2kc)
        #print '  csc=', reduce(np.dot, (C.T,s1,C))
        self.props_r = props_r

    def get_diff(self, dm=None, props_r=None, hao=F, rc_dm=4.2):
        # test if E is local w.r.t dm1r
        if dm is None:
            _dm = self.mf.make_rdm1()
            print(' * test on the accuracy of localized rdm1 determined by a cutoff ')
            dm_new = self.localize_dm(_dm, rc_dm=rc_dm)
        else:
            dm_new = dm
        props = self.feed_new_dm(dm_new, hao=hao)

        if props_r is None:
            if not hasattr(self,'props_r'): self.calc_properties()
        deltas = props - self.props_r
        for i,key in enumerate(self.prop_names):
            v1,v2 = self.props_r[i], self.props[i]; dv = v2-v1
            print(' ** %s=%.2f, %s_u=%.2f, delta_%s=%.2f [%s]'%(key,v1,key,v2,key,dv,self.units[i]))
        self.props = props

    def calc_ca_dm(self, idx=None, idx2=None, iprt=F):
        """
        Calculate configuration-adapted density matrix
        Pyscf is used as the solver to generate training data

        ****
          Note that in pyscf, orbital \chi_k = \sum_i \phi_i * A_{ik}
          thus, \rho = \sum_k \chi_k^{*} \chi_k
                     = \sum_{k,i,j} A_{ik}^* S_{ij} A_{jk}
                     = \sum_{i,j} (A.T * S * A)_kk

                     = \sum_{k,i,j} A_{jk} * (A'^T)_{ki} S_{ij}
                     = \sum_{i,j} (\sum_k A_{jk} * (A'^T)_{ki}) S_{ij}
                     = \sum_{i,j} D_{ji}*S_{ij}
          where D = A * A.T (summed over all occupied orbitals, indexed by `k)
        """
        meth = self.meth
        m = self.m
        B = self.B
        s = self.s
        s1 = self.s1
        if meth in ['hf','b3lyp']:
            if iprt: print(' * scf starting...')
            mf = self.scf(m)
            if self.idft: mf.xc = meth
            er = mf.kernel()
            self.mf = mf
            if iprt: print(' * scf done')
            A = mf.mo_coeff
            dm0 = reduce(np.dot, (A, np.diag(mf.mo_occ), A.T))
            dm0u = mf.make_rdm1()
            self.dm0 = dm0u
            assert np.all(np.abs(dm0u-dm0)<=1e-6)
            C = np.linalg.solve(B, A)
            dm1 = reduce( np.dot, (C, np.diag(mf.mo_occ), C.T) )
            if idx != None:
                if idx2==None:
                    print(' dm1 = ', dm1[idx][:,idx])
                else:
                    print(' dm1 = ', dm1[idx][:,idx2])
            self.dm1 = dm1

        elif self.meth in ['ccsd',]:
            print(' * scf starting...')
            mf = self.scf(m).run()
            print(' * scf done')
            A = mf.mo_coeff
            nmo = A.shape[1]
            if self.iprt: print(' -- nmo = ', nmo)

            if self.iprt: print(' * cc starting...')
            cc2 = cc.CCSD(mf)
            if nmo>60: cc2.direct=True
            cc2.kernel()
            if self.iprt: print(' * cc done')
            if self.iprt: print(' * generating rdm1...')
            dm1r = cc2.make_rdm1()
            if self.iprt: print(' * rdm1 done')
            if self.iprt: print(' * generating rdm2...')
            dm2r = cc2.make_rdm2()
            if self.iprt: print(' * rdm2 done')

            # CCSD energy based on density matrices
            if self.iprt: print(' * calc reference h1')
            h1 = np.einsum('pi,pq,qj->ij', A.conj(), mf.get_hcore(), A)
            E1 = np.einsum('pq,qp', h1, dm1r); #print 'h1r=', E1
            if self.iprt: print(' * done')
            if self.iprt: print(' * calc reference g2')
            eri = ao2mo.kernel(m, A, compact=False).reshape([nmo]*4)
            # Note dm2 is transposed to simplify its contraction to integrals
            t = np.einsum('pqrs,pqrs', eri, dm2r) * .5; #print 'h2r=',t
            if self.iprt: print(' * done')
            E1+= t
            E1+= m.energy_nuc()
            #print('E1 = %s, reference %s' % (E1, cc2.e_tot))

            # When plotting CCSD density on the grids, CCSD density matrices need to be
            # first transformed to AO basis.
            #dm1_ao = np.einsum('pi,ij,qj->pq', A, dm1r, A.conj())
            C = np.linalg.solve(B, A)
            # \gamma_1 = \sum_{ij} D_ij * \chi_i * \chi_j
            #          = \sum_{pq} \sum_{ij} D_ij \phi_p * A_{pi} * \phi_q *  A_{qj}
            #          = \sum_{pq} \sum_{ij} A_{pi} * D_ij * A_{qj} * S_{pq}
            #          = \sum_{pq} D'_{pq} S_{pq}
            if self.iprt: print(' * calc h1 in HAO')
            dm1_hao = np.einsum('pi,ij,qj->pq', C, dm1r, C.conj()) # D' = dm1_hao
            # \gamma_2 = \sum_{ijkl} D_{ijkl} \chi_i * \chi_j * \chi_k * \chi_l
            #          = \sum_{pqrs} \sum_{ijkl} D_{ijkl} \phi_p*A_pi * \phi_q*A_qj * \phi_r*A_rk * \phi_s*A_sl
            #          = \sum_{pqrs} \sum_{ijkl} A_pi * A_qj * D_{ijkl} * A_rk * A_sl * <pq|rs>
            h1 = np.einsum('pi,pq,qj->ij', B.conj(), mf.get_hcore(), B)
            E2 = np.einsum('pq,qp', h1, dm1_hao); #print ' h1=', E2
            if self.iprt: print(' * done')
            if self.iprt: print(' * calc g2 in HAO')

            #t1 = np.tensordot(C,C)
            #t2 = np.tensordot(C.conj(),C.conj())
            #dm2_hao = np.einsum('pqij,ijkl,klrs->pqrs', t1, dm2r, t2)
            dm2_hao = lib.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, dm2r, C.conj(), C.conj() )
            eri = ao2mo.kernel(m, B, compact=False).reshape([nmo]*4)
            # Note dm2 is transposed to simplify its contraction to integrals
            t = np.einsum('pqrs,pqrs', eri, dm2_hao) * .5; #print ' h2=', t
            if self.iprt: print(' * done')
            E2 += t
            E2+= m.energy_nuc()
            assert np.abs(E2-E1) <= 1e-9
            print(('E2 = %s, reference %s' % (E2, cc2.e_tot)))
            self.dm1 = dm1_hao
            self.dm2 = dm2_hao
        else:
            raise '#unknown `meth'

    def decompose_dm(self, naomax, im, rc_dm=4.8):
        """ convert DM matrix to vectors, each one represents a block of DM,
        which is more convinient to deal with in KRR

        Assume the same basis was used for atoms of the same kind.
        """
        _vs = []
        ns = []
        labels = []
        for i in range(self.na):
            for j in range(self.na):
                if self.ds[i,j] <= rc_dm:
                    zi, zj = self.zs[i], self.zs[j]
                    iaos, jaos = self.aoidxs[i], self.aoidxs[j]
                    nr, nc = len(iaos), len(jaos)
                    ns.append([nr,nc,nr*nc])
                    ib, ie = iaos[0], iaos[-1]+1
                    jb, je = jaos[0], jaos[-1]+1
                    v = self.dm1[ib:ie,jb:je]
                    _vs.append( v.ravel() )
                    labels.append( [im, i, j, zi, zj] )
        nlb = len(labels)
        labels = np.array(labels, np.int)
        ns = np.array(ns, np.int)
        vs  = np.zeros((nlb,naomax**2))
        for iv,v in enumerate(_vs):
            vs[iv,:ns[iv,2]] = v
        return labels, ns, vs

    def reconstruct_dm(self, labels, vs):
        """ from ML pieces of DM blocks, i.e., Y vectors """
        dm = np.zeros((self.nbf, self.nbf))
        for iv,v in enumerate(vs):
            i,j,zi,zj = labels[iv, 1:5]
            if np.all(v==0): print('i,j,zi,zj = ', i,j,zi,zj)
            iaos, jaos = self.aoidxs[i], self.aoidxs[j]
            nr, nc = len(iaos), len(jaos)
            ib, ie = iaos[0], iaos[-1]+1
            jb, je = jaos[0], jaos[-1]+1
            dm[ib:ie,jb:je] = v[:nr*nc].reshape([nr,nc])
        return dm


class yobj(object):
    def __init__(self, labels, ns, props, ys):
        self.ys = np.array(ys) # density matrix
        self.props = np.array(props)
        self.ns = np.array(ns,np.int)
        self.labels = np.array(labels,np.int)


class YData(object):
    def __init__(self, nas, zs, coords, rc_dm=4.2, basis='sto-3g',\
                 meth='b3lyp', spin=0, verbose=3, iprt=F):
        self.rc_dm = rc_dm
        self.basis = basis
        self.meth = meth
        self.spin = spin
        self.verbose = verbose
        self.iprt = iprt
        nbfs = []
        zsu = np.unique(zs)
        nm = len(nas)
        for zi in zsu:
            atm = gto.M(atom='%s 0 0 0'%chemical_symbols[zi], basis=basis, spin=zi%2)
            nbfs.append(atm.nao)
        naomax = np.max(nbfs)
        ias2 = np.cumsum(nas)
        ias1 = np.array([0]+list(ias2[:-1]),np.int)
        ys=[]; ns=[]; props=[]; labels=[]
        for im in range(nm):
            ib, ie = ias1[im], ias2[im]
            obj = density_matrix(zs[ib:ie], coords[ib:ie], basis=basis, meth=meth, \
                                 spin=spin, verbose=verbose, iprt=iprt)
            obj.calc_ca_dm()
            obj.calc_properties()
            _labels, _ns, _ys = obj.decompose_dm(naomax, im, rc_dm=rc_dm)
            labels += list(_labels)
            props.append( obj.props_r )
            ns += list(_ns)
            ys += list(_ys)
        self.yobj = yobj(labels, ns, props, ys)



if __name__ == '__main__':
  import stropr as so

  args = sys.argv[1:]

  cnt = 0
  keys = ['-iprt']; iprt, cnt = so.haskey(args, keys, cnt)

  s0 = """
  x = 1.09/np.sqrt(3.)
  c1,s1 = np.cos(np.pi/3), np.sin(np.pi/3)
  zs = [6, 1, 1, 1, 1]
  coords = np.array( [[ 0,  0,  0],
                      [ -x,  -x,  -x],
                      [ x, x, -x],
                      [ x,  -x, x],
                      [ -x, x, x] ])
  m = Atoms(zs,coords)
  """

  #s0 = """
  _fns = ['ch4.xyz',] # ['ch4','c2h6','c3h8','c4h10']
  #_fns = [ 'test/'+fi+'.xyz' for fi in ['c12h14', 'c12h26']]#'c04h06','c06h08','c08h10', 'c12h14','c16h18'] ]
  #_fns = [ 'test/'+fi+'.xyz' for fi in ['c12h26', ]]#'c04h06','c06h08','c08h10', 'c12h14','c16h18'] ]

  fns = _fns #[2:4];
  idx = None # range(51,54)  #None #range(5) # None # [42] #range(35,40) #[0,1,2,3,4] # 41]
  idx2 =  None #range(51,54) #None # range(10,15) # None #[41] #range(30,35) #range(5,10) #range(35,40)+[49] #
  rcs = [3.2, 4.2] # 4.8] # [ rc_x, rc_dm ]
  rc_x, rc_dm = rcs

  basis='sto-3g'; meth='b3lyp'
  spin=0; a=0.; verbose=3

  #fns = ['%s/Dropbox/QMC-Anouar/bio/g09_opt/b3lyp_def2tzvp/sent-to-Anouar/bio_geom/frag_01.xyz'%home]
  #for fn in fns:
  for fn in fns:

    print('\n molecule=', fn)
    print(' [rc_x, rc_dm] = ', rc_x, rc_dm)

    m = aio.read(fn)
    a = 0.0
    v = [1.,1.,1.]
    if a != 0.0:
        m.rotate(v, a*np.pi/180.) # rotate `a degree around vector `v
    zs, coords = m.numbers, m.positions
    fno = fn[:-4] + '.out'
    obj = density_matrix(zs, coords, output=None, basis=basis, meth=meth, \
                         spin=spin, verbose=verbose, iprt=iprt)
    obj.calc_ca_dm(idx=idx, idx2=idx2)
    print(obj.dm1)
    continue #sys.exit(2)
    hao=F; dm1_hao = None #obj.dm1 # np.around(obj.dm1, decimals=4)
    obj.calc_properties()
    props = []
    for rc_dm in np.linspace(2.0,6.0,41): #[2,3,4,5,6.0]:
      obj.get_diff(dm=dm1_hao, hao=hao, rc_dm=rc_dm)
      props.append(obj.props)
    if meth not in ['hf','b3lyp']:
        dms = obj.dm1, obj.dm2
        #np.savez('/data/bio/01_vdz.npz', dm1=dms[0],dm2=dms[1])
    else:
        dms = obj.dm1


