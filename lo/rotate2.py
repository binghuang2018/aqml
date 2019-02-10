#!/usr/bin/env python

from time import time
from functools import reduce
import numpy as np
from pyscf import gto, scf
from cheminfo.core import *
import cheminfo.graph as cg
from cheminfo.molecule.elements import *
from cheminfo.molecule.core import *
from pyscf.lo.orth import pre_orth_ao_atm_scf
from pyscf.tools import molden

class atoms(object):
    """
    get HF orbitals for free atoms
    """
    def __init__(self, zs, basis):
        return


lmap = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4}
mmap = {'':0, \
        'x':0, 'y':1, 'z':2, \
        'xy':0, 'yz':1, 'z^2':2, 'xz':3, 'x2-y2':4 }
nmap = {1:1, 6:2, 7:2, 8:2, 9:2, 15:3,16:3,17:3} # for MINBASIS

class pyscf_object(object):

    def __init__(self, symbs, coords, basis, spin=0):
        self.spin = spin
        str_m = '' #'O 0 0 0; H 0 0 1; H 0 1 0'
        for i, si in enumerate(symbs):
            x, y, z = coords[i]
            str_m += '%s %.8f %.8f %.8f; '%(si, x, y, z)
        str_m = str_m[:-2]
        mol = gto.M(atom=str_m, basis=basis, verbose=0, spin=spin)#, symmetry=False)
        self.mol = mol
        self.na = len(symbs)
        self.nao = mol.nao
        self.zs = [ chemical_symbols.index(symbs[0]) ] + [1,]*(self.na-1)

    def get_ao_map(self):
        ids = self.mol.offset_ao_by_atom()[:, 2:4]
        ibs, ies = ids[:,0], ids[:,1]
        self.ibs = ibs
        self.ies = ies
        idx = [ np.arange(ibs[i],ies[i]) for i in range(self.na) ]
        self.idx = idx
        imap = np.zeros(self.nao).astype(np.int)
        for i in range(self.na):
            imap[ibs[i]:ies[i]] = i
        self.imap = imap
        labels = self.mol.ao_labels()
        nlms = []
        for _lb in labels:
            lb = _lb.strip().split()[-1]
            n = int(lb[0])
            l = lmap[lb[1]]
            m = mmap[lb[2:]]
            nlms.append( [n,l,m] )
        self.nlms = np.array(nlms, np.int)

    def get_hao(self):
        """
        C: central atom
        L: ligands (i.e., hydrogen atoms)
        """
        mol = self.mol
        zs = mol.atom_charges()
        nh = (zs[1:]==1).sum()

        c = pre_orth_ao_atm_scf(mol) #
        _s = mol.intor_symmetric('int1e_ovlp')
        s = reduce( np.dot, (c.conjugate().T, _s, c) )
        s1 = s[-nh:,:-nh]
        ## Recommended approach: SVD, which is more elegent!!
        ## use svd instead! Now each col is a HAO!!
        u,d,vh = np.linalg.svd(s1, full_matrices=False, compute_uv=True)
        a1 = np.dot(vh.T, u.T) # transformation matrix, { \psi } = { \phi }*A

        n1 = nh
        n2 = self.nao - nh
        t = np.eye(n2)
        t[:,:nh] = a1
        for i in range(n2-nh):
            print('i=',i+nh)
            ti = t[:,i+nh]
            for j in range(i+nh):
                cj = np.dot(t[:,j],t[:,i+nh])
                ti -= cj*t[:,j]
            t[:,i+nh] = ti/np.linalg.norm(ti)

        # check orthogonality
        for i in range(n2):
            ti = t[:,i]
            assert np.abs(np.dot(ti,ti)-1.0)<=1e-6
            for j in range(i+1,n2):
                tj = t[:,j]
                val = np.abs(np.dot(ti,tj))
                print('val=',i,j,val)
                assert val <=1e-6

        aolbs = mol.ao_labels()
        for i in range(n2):
            csi = t[i,:]
            so = ' '.join(['%6.2f '%si for si in csi])
            print('%20s'%aolbs[i], so)
        return t


class config_adapted_hao(RawMol):

    """
    configuration adapted hybridized ANO
    """

    def __init__(self, zs, coords, basis='cc-pvdz'): # fn='test'):
        self.rcs = Elements().rcs
        self.basis = basis
        #self.fn = fn
        #assert np.sum(self.zs)%2 == 0, '#ERROR: spin polarised?'
        RawMol.__init__(self, list(zs), coords)

        spin = sum(self.zs)%2
        symbs = [ chemical_symbols[zi] for zi in self.zs ]
        OBJ = pyscf_object(symbs, coords, basis, spin=spin)
        self.mol = OBJ.mol
        self.nao = OBJ.mol.nao

        self.aoidxs = OBJ.mol.offset_ao_by_atom()[:, 2:4]
        self.T0 = pre_orth_ao_atm_scf(OBJ.mol) #
        self.T = np.eye(OBJ.mol.nao)

    def run(self):
        #
        for iac in range(self.na):
            jas = self.ias[ self.g[iac]>0 ]
            zi = self.zs[iac]
            si = chemical_symbols[zi]
            zs1 = [ zi ] + [1,]*len(jas)
            symbs1 = [ si, ] + [ 'H@2' ]*len(jas)
            basis1 = { si: self.basis, 'H@2':'sto-3g' }
            origin = self.coords[iac]
            coords1 = [ origin ]
            for j in jas:
                dest = self.coords[j]
                v = dest - origin
                d = np.sum( self.rcs[ [1,self.zs[j]] ] )
                coords1.append( origin + d * v/np.linalg.norm(v) )
            spin = sum(zs1)%2
            subm = pyscf_object(symbs1, coords1, basis1, spin=spin)
            if self.basis=='sto-3g' and zi==1:
                R = 1.0
            else:
                R = subm.get_hao()
            ib,ie = self.aoidxs[iac]
            self.T[ib:ie,ib:ie] = R
            #molden.from_mo(subm.mol, '%s_%03d.molden'%(self.fn, iac+1), subm.c2)
        self.t = np.dot(self.T0, self.T)


if __name__ == '__main__':

  from ase import Atoms
  import ase.io as aio
  import os, sys
  import scipy.spatial.distance as ssd

  home = os.environ['HOME']
  np.set_printoptions(precision=3,suppress=True)

  s0 = """
  x = 1.09/np.sqrt(3.)
  c1,s1 = np.cos(np.pi/3), np.sin(np.pi/3)
  zs = [6, 1, 1, 1, 1]
  #coords = np.array( [[ 0,  0,  0],
  #                    [ -x*c1,  x*s1,  0],
  #                    [ -x*c1, -x*s1,  0],
  #                    [ x,  0, 0],
  #                    [ 0, 0, x*2] ])

  coords = np.array( [[ 0,  0,  0],
                      [ -x,  -x,  -x],
                      [ x, x, -x],
                      [ x,  -x, x],
                      [ -x, x, x] ])
  m = Atoms(zs,coords)
  """

  #s0 = """
  fns = ['test-ch/ch4',] #'c2h6','c3h8','c4h10']
  for fn in fns:
    #fn = fns[3]
    print(' molecule=', fn)
    m = aio.read('%s.xyz'%fn)

    a = 0.0 # 36.0
    v = [1.,1.,1.]
    if a != 0.0:
        m.rotate(a, v) # rotate `a degree around vector `v
    zs, coords = m.numbers, m.positions

    na = len(zs)
    bst='sto-3g' #'cc-pvdz'
    obj = config_adapted_hao(zs, coords, basis=bst)
    obj.run()
    B = obj.t
    #print(' * B (ANO) = ')
    #print (obj.T)

    # pyscf run
    m = obj.mol
    s1 = m.intor_symmetric('int1e_ovlp')
    s2 = reduce(np.dot, (B.T,s1,B))
    #print ' * s2 = ', s2

    mf = scf.RHF(m)
    mf.run()
    occ = mf.mo_occ
    nmo = (occ>0).sum()

    e1 = mf.e_tot
    es = mf.mo_energy[nmo-1:nmo+1]

    print(' e_tot, e_homo, e_lumo = %.4f %.4f %.4f'%(e1, es[0], es[1]))

    # in pyscf, orbital \chi_k = \sum_i \phi_i * A_{ik}
    # thus, \rho = \sum_k \chi_k^{*} \chi_k
    #            = \sum_{k,i,j} A_{ik}^* S_{ij} A_{jk}
    #            = \sum_{i,j} (A.T * S * A)_kk
    #
    #            = \sum_{k,i,j} A_{jk} * (A'^T)_{ki} S_{ij}
    #            = \sum_{i,j} (\sum_k A_{jk} * (A'^T)_{ki}) S_{ij}
    #            = \sum_{i,j} D_{ji}*S_{ij}
    # where D = A * A.T (summed over all occupied orbitals, indexed by `k)
    #
    A = mf.mo_coeff
    #print(' A = ')
    #print(A)
    #occ[nmo] += 1
    #print ' occ = ', occ

    #dm1 = reduce( np.dot, (A, np.diag(occ), A.T) )
    dm1 = mf.make_rdm1(A, mo_occ=occ)
    print('dm1 = ', dm1)
    ne1 = np.trace( np.dot(dm1,s1) )
    e1_u = mf.energy_tot(dm=dm1)
    assert abs(e1_u-e1) < 1e-9
    #print(' dE = ', e1_u - e1)
    #print ' a) n = ', ne1, ', csc=', reduce(np.dot, (A.T,s1,A))

    C = np.linalg.solve(B, A)
    #print(' C = ' )
    #print(C)
    #C = np.dot(A, np.linalg.inv(B))

    dm2 = reduce( np.dot, (C, np.diag(occ), C.T) )
    print(' dm2 = ')
    print(dm2)
    ne2 = np.trace( np.dot(dm2,s2))
    msg = ' ** ne1=%.2f, ne2=%.2f'%(ne1,ne2)
    assert np.abs(ne2-ne1)<=1e-4, msg
    #print ' N_e = ', ne2, ', csc=', reduce(np.dot, (C.T,s2,C))

    #idx = [0,1,2,3,4, 11,12,13]
    #n = int(fn[1])
    #idx = [0,1,2,3,4] + [ 5*n+j for j in range(2) ]
    #idx = range(10) #+ range(30,35) #idx = [7*5+i for i in range(5)] + [48,49]
    #print ' dm2 = ', dm2[idx][:,idx]
    #print ' C = ', C[idx][:,idx]

