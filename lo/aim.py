

import cheminfo as co 
import calculators.pscf as pscf 
from pyscf.lib import param
from pyscf.data import elements
from pyscf.scf import hf
from pyscf import gto
from pyscf.scf import atom_hf
from pyscf.scf import addons
import scipy
import numpy as np 
from functools import reduce


def eig(h, s):
    '''
    Solver for generalized eigenvalue problem
    .. math:: HC = SCE
    '''
    e, c = scipy.linalg.eigh(h, s)
    idx = np.argmax(abs(c.real), axis=0)

    # identify the maximal coeff for the basis function of 
    # each MO (column of `c) and ensure it's positive!
    c[:,c[idx,np.arange(len(e))].real<0] *= -1
    return e, c


class ascf(pscf.calculator):

    """
      ascf: scf (hf/dft) solution for an atom

      Note solution is spherically averaged
    """

    def __init__(self, symb, meth='hf', basis='sto-3g', debug=False):

        self.debug = debug
        zs = [ co.chemical_symbols.index(symb) ]
        coords = [[0.,0.,0.]]
        m = co.atoms(zs, coords)
        pscf.calculator.__init__(self, m, meth=meth, \
                  basis=basis, isphav=True, charge=0)
        self.get_h() #href=href) 
        self.get_ascf()


    def get_ascf(self):
        """
          solve spherically averaged hf of atom 
        """
        mol = self.mol
        symb = mol.atom_symbol(0)
        aolm = self.aolm 
        nao = mol.nao

        f = self.fock # fock matrix
        s = self.s 

        mo_coeff = []
        mo_energy = []
        for l in range(param.L_MAX):
            degen = 2 * l + 1
            idx = np.where(aolm == l)[0]
            nao_l = len(idx)
            if nao_l > 0:
                nsh = nao_l // degen
                f_l = f[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                s_l = s[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                # Average over angular parts
                f_l = np.einsum('piqi->pq', f_l) / degen
                s_l = np.einsum('piqi->pq', s_l) / degen
                e, c = eig(f_l, s_l)
                if self.debug:
                    for i, ei in enumerate(e):
                        print(' ## l = %d  e_%d = %.9g'%(l, i, ei))
                mo_energy.append(np.repeat(e, degen))
                mo = np.zeros((nao, nsh, degen))
                for i in range(degen):
                    mo[idx[i::degen],:,i] = c
                mo_coeff.append(mo.reshape(nao, nao_l))

        self.es = np.hstack(mo_energy)
        self.cs = np.hstack(mo_coeff)
        self.occs = self.get_occ()


    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''spherically averaged fractional occupancy'''
        mol = self.mol
        symb = mol.atom_symbol(0)

        nelec_ecp = mol.atom_nelec_core(0)
        coreshl = gto.ecp.core_configuration(nelec_ecp)

        occ = []
        for l in range(param.L_MAX):
            n2occ, frac = frac_occ(symb, l)
            degen = 2 * l + 1
            idx = mol._bas[:,gto.ANG_OF] == l
            nbas_l = mol._bas[idx,gto.NCTR_OF].sum()
            if l < 4:
                n2occ -= coreshl[l]
                assert n2occ <= nbas_l

                if self.debug:
                    print(' ## l = %d  occ = %d + %.4g'%(l, n2occ, frac))

                occ_l = np.zeros(nbas_l)
                occ_l[:n2occ] = 2
                if frac > 0:
                    occ_l[n2occ] = frac
                occ.append(np.repeat(occ_l, degen))
            else:
                occ.append(np.zeros(nbas_l * degen))

        return np.hstack(occ)


def frac_occ(symb, l, atomic_configuration=elements.NRSRHF_CONFIGURATION):
    nuc = gto.charge(symb)
    if l < 4 and atomic_configuration[nuc][l] > 0:
        ne = atomic_configuration[nuc][l]
        nd = (l * 2 + 1) * 2
        ndocc = ne.__floordiv__(nd)
        frac = (float(ne) / nd - ndocc) * 2
    else:
        ndocc = frac = 0
    return ndocc, frac


class adct(object):

    """
      a dictionary of spherically averaged 
      hf/dft solution of free atoms 
    """

    def __init__(self, symbs, meth='hf', basis='sto-3g'):
        self.basis = basis
        self.atoms = {}
        for s in symbs:
            aobj = ascf(s, meth=meth, basis=basis)
            self.atoms[s] = aobj 



class aimcao(pscf.calculator):

    """ 
      atom-in-molecule configuration-adapted atomic orbital 
    """

    def __init__(self, m, meth='hf', basis='sto-3g', rcut=None):

        self.m = m 
        self.rcut = rcut # 6.0 Angstrom
        pscf.calculator.__init__(self, m, meth=meth, basis=basis)
        self.get_h(iexe=False) 

    @property 
    def dm1(self):
        return self.get_dm() #cycle=1)

    def get_dm(self): #, cycle=1):
        '''
          Get DM in configuration-adapted basis
          
        steps:
        1) Guess density matrix from superposition 
           of atomic HF density matrix
        2) build Fock matrix
        3) update DM by diagonalizing corresponding
           Fock matrix
        4) construct new DM

        Note that the initial atomic DM was obtained through solving
        the occupancy averaged RHF

        Returns:
            Density matrix, 2D ndarray
        '''
        mol = self.mol 
        atm_scf = atom_hf.get_atm_nrhf(mol)
        aoslice = mol.aoslice_by_atom()
        atm_dms = []
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb not in atm_scf:
                symb = mol.atom_pure_symbol(ia)
            if symb in atm_scf:
                e_hf, e, c, occ = atm_scf[symb]
                dm = np.dot(c*occ, c.conj().T)
            else:  # symb's basis is not specified in the input
                nao_atm = aoslice[ia,3] - aoslice[ia,2]
                dm = np.zeros((nao_atm, nao_atm))
            atm_dms.append(dm)

        dm = scipy.linalg.block_diag(*atm_dms)
        if mol.cart:
            cart2sph = mol.cart2sph_coeff(normalized='sp')
            dm = reduce(np.dot, (cart2sph, dm, cart2sph.T))

        mf = self.mf 
        #if cycle > 0:
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(self.h1e, self.s, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, self.s)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)

        return dm


    @property
    def pnao(self):
        return self.get_pnao(dm1=self.dm1)


    def get_dm_cab(self):
        """ density matrix in CAO """
        self.mf.kernel()
        dm = self.rdm1
        _, T = self.pnao
        C = np.linalg.inv(T) # np.linalg.solve(T, self.mf.mo_coeff)
        return reduce(np.dot, (C, dm, C.T))

