
import ase.data as ad
from pyscf import gto, dft, scf, cc, mp, ci
#from pyscf.geomopt import berny_solver
import aqml.cheminfo.core as cic
import os, sys, scipy
from pyscf.tools.cubegen import *
from pyscf.data import elements
import numpy as np
from functools import reduce
import pyscf

_mult = {1:2, 3:2, 4:1, 5:2, 6:3, 7:4, 8:3, 9:2, \
        11:2, 12:0, 13:2, 14:3, 15:4, 16:3, 17:2,\
        33:4, 34:3, 35:2, 53:2}

T, F = True, False

def lowdin(s_mm):
    """Return S^-1/2"""
    eig, rot_mm = np.linalg.eig(s_mm) #find eigenvalues overlap matrix
    eig = np.abs(eig)
    rot_mm = np.dot(rot_mm / np.sqrt(eig), rot_mm.T.conj()) #rotation matrix S^-1/2.
    return rot_mm



class EHT(object):

    def __init__(self, mol):

        self.mol = mol

        atm_scf = scf.atom_hf.get_atm_nrhf(self.mol)

        # GWH parameter value
        Kgwh = 1.75

        # Run atomic SCF calculations to get orbital energies,
        # coefficients and occupations
        at_e = []
        at_c = []
        at_occ = []
        for ia in range(self.mol.natm):
            symb = self.mol.atom_symbol(ia)
            if symb not in atm_scf:
                symb = self.mol.atom_pure_symbol(ia)
            e_hf, e, c, occ = atm_scf[symb]
            at_c.append(c)
            at_e.append(e)
            at_occ.append(occ)

        # Number of basis functions
        nbf = mol.nao_nr()
        # Collect AO coefficients and energies
        orb_E = np.zeros(nbf)
        orb_C = np.zeros((nbf,nbf))

        # Atomic basis info
        aoslice = mol.aoslice_by_atom()

        for ia in range(mol.natm):
            # First and last bf index
            abeg = aoslice[ia, 2]
            aend = aoslice[ia, 3]
            orb_C[abeg:aend,abeg:aend] = at_c[ia]
            orb_E[abeg:aend] = at_e[ia]

        # Overlap matrix
        S = scf.hf.get_ovlp(mol)
        # Atomic orbital overlap
        orb_S = reduce(np.dot, (orb_C.T, S, orb_C))

        # Build Huckel matrix
        orb_H = np.zeros((nbf,nbf))
        for iorb in range(nbf):
            # Diagonal is just the orbital energies
            orb_H[iorb,iorb] = orb_E[iorb]
            for jorb in range(iorb):
                # Off-diagonal is given by GWH approximation
                orb_H[iorb,jorb] = 0.5*Kgwh*orb_S[iorb,jorb]*(orb_E[iorb]+orb_E[jorb])
                orb_H[jorb,iorb] = orb_H[iorb,jorb]

        #print('orb_H=', orb_H)
        #print('orb_S=', orb_S)

        # Energies and coefficients in the minimal orbital basis
        mo_energy, atmo_C = scf.hf.eig(orb_H, orb_S)
        # and in the AO basis
        mo_coeff = orb_C.dot(atmo_C)

        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy

    def get_dm(self):
        mo_occ = scf.hf.get_occ(scf.hf.SCF(self.mol), self.mo_energy, self.mo_coeff)
        return scf.hf.make_rdm1(self.mo_coeff, mo_occ)



class calculator(object):

    def __init__(self, m, meth='b3lyp', basis='sto-3g', \
                 spin=None, charge=0, isphav=F):
        """

        :param isphav: is spherically averaged calculation? True/False
        :type isphav: bool
        """

        self.meth = meth.lower()
        self.basis = basis.lower()
        self.isphav = isphav # is spherically averaged? T/F, for free atom only
        self.m = m
        smol = '' #'O 0 0 0; H 0 0 1; H 0 1 0'
        ne = sum(m.zs) - charge
        for i in range(m.na):
            x, y, z = m.coords[i]
            si = m.symbols[i]
            smol += '%s %.8f %.8f %.8f; '%(si, x, y, z)
        _smol = smol[:-2]
        restricted = F
        if isphav:
            spin = ne % 2
            restricted = T
        else:
            if m.na == 1:
                spin = _mult[zs[0]] - 1
            else:
                if spin is None:
                    spin = ne % 2
            if spin == 0:
                restricted = T
        self.restricted = restricted
        mol = gto.M(atom=_smol, basis=basis, verbose=0, spin=spin, charge=charge)
        self.mol = mol

    @property
    def ao_labels(self):
        return self.mol.ao_labels()

    @property
    def aoidxs(self):
        return self.mol.aoslice_by_atom( self.mol.ao_loc_nr() )

    @property
    def aolm(self):
        """
          angular momentum for each AO, return as a list
        """
        if not hasattr(self, '_aolm'):
            _aolm = np.zeros(self.mol.nao, dtype=numpy.int)
            ao_loc = self.mol.ao_loc_nr()
            for i in range(self.mol.nbas):
                p0, p1 = ao_loc[i], ao_loc[i+1]
                _aolm[p0:p1] = self.mol.bas_angular(i)
            self._aolm = _aolm
        return self._aolm


    def get_h(self, iexe=True, href=None, frozen=0):
        """
            get hamitonian of the sysem, which is to be described by
            a hf/dft single slater determinant

        :param href: reference hamiltonian, could be hf/uhf/rohf/rks/uks
        :type href: str
        """
        meth = self.meth
        self.xc = None
        if href is None:
            if meth in ['eht', 'hf', 'mp2', 'cisd', 'ccsd',]:
                fun = scf.RHF if self.restricted else scf.UHF
                mf = fun(self.mol)
            elif meth in ['pbe','b3lyp','w95xb']:
                fun = dft.RKS if self.restricted else dft.UKS
                mf = fun(self.mol)
                mf.xc = meth
                self.xc = meth
            else:
                raise Exception('Unknow method: %s'%meth)
        else:
            dct = {'rohf':scf.ROHF, 'rhf':scf.RHF, 'hf':scf.RHF,\
                   'rks': dft.RKS, 'uks':dft.UKS, 'ks':dft.RKS}
            assert href in dct
            fun = dct[href]
            mf = fun(self.mol)
            if 'ks' in href: mf.xc = meth
        if iexe:
            mf.kernel()
        self.mf = mf

        h2 = None
        self.isd = T # single (slater) determinant
        # final hamiltonian
        if meth[:2] in ['mp','ci','cc']:
            self.isd = F
            if meth in ['mp2','mp3','mp4']:
                h2 = mp.MP2(self.mf) #.set(frozen=frozen)
            elif meth in ['cisd',]:
                h2 = ci.CISD(self.mf) #.set(frozen=frozen)
            elif meth in ['ccsd', 'ccsd(t)']:
                h2 = cc.CCSD(self.mf) #.set(frozen=frozen)
                h2.direct = True
            else:
                raise Exception('Todo')
            if frozen:
                h2.set(frozen=frozen)
            if iexe:
                h2.kernel()
        self.h2 = h2

    def get_ecc2(self):
        """ get ccsd(t) energy """
        mycc = self.h2.kernel()
        e3 = cc.ccsd_t.kernel(mycc, mycc.ao2mo())
        return mycc.e_tot + e3

    @property
    def s(self):
        """ overlap matrix """
        if not hasattr(self, '_ovlp'):
            if getattr(self.mol, 'pbc_intor', None):  # whether mol object is a cell
                s = self.mol.pbc_intor('int1e_ovlp', hermi=1)
            else:
                s = self.mol.intor_symmetric('int1e_ovlp')
            self._ovlp = s
        return self._ovlp

    @property
    def dm1r(self):
        """ 1st order reduced density matrix based on reference wf """
        if not hasattr(self, '_dm1r'):
            if self.meth == 'eht':
                _dm1 = EHT(self.mol).get_dm()
            else:
                _dm1 = self.mf.make_rdm1()
            dm1 = _dm1.copy()
            if not (isinstance(_dm1, np.ndarray) and _dm1.ndim == 2):
                dm1 = _dm1[0] + _dm1[1]
            self._dm1r = dm1
        return self._dm1r

    @property
    def rdm1(self):
        """ 2nd order reduced density matrix based on reference wf """
        if not hasattr(self, '_dm1'):
            if self.isd:
                _dm1 = self.dm1r
            else:
                _dm1 = self.h2.make_rdm1()
            self._dm1 = _dm1
        return self._dm1

    @property
    def rdm2(self):
        """ 2nd order reduced density matrix based on reference wf """
        if not hasattr(self, '_dm2'):
            _dm2 = self.h2.make_rdm2()
            self._dm2 = _dm2
        return self._dm2


    def get_pnao(self, dm1=None, i_sph_avg=False):
        """ pre-orthogonal NAO """
        s = self.s
        dm = dm1 if dm1 is not None else self.rdm1
        mol = self.mol
        p = reduce(np.dot, (s, dm, s))
        ao_loc = mol.ao_loc_nr()
        nao = ao_loc[-1]
        occ = np.zeros(nao)
        cao = np.zeros((nao,nao), dtype=s.dtype)
        for ia, (b0,b1,p0,p1) in enumerate(mol.aoslice_by_atom(ao_loc)):
            pA = p[p0:p1, p0:p1]
            sA = s[p0:p1, p0:p1]
            ## lowdin orthogonalize
            lA = lowdin(sA) #S^(-1/2)
            pAL = reduce(np.dot, (lA.T.conj(), pA, lA))
            sAL = reduce(np.dot, (lA.T.conj(), sA, lA))
            e, v = scipy.linalg.eigh(pAL, sAL)
            e = e[::-1]
            v = v[:,::-1]
            norms_v = reduce(np.dot, (v.T.conj(), sA, v)).diagonal()
            v /= np.sqrt(norms_v) # ensure normalization
            v = np.dot(lA, v)
            occ[p0:p1] = e
            cao[p0:p1, p0:p1] = v
        return occ, cao

    def get_nao(self):
        """ get NAO, with spherical averaging! """
        occ, cao = pyscf.lo.nao._prenao_sub(self.mol, self.p, self.s)
        return occ, cao

    def optg(self):
        stat = berny_solver.optimize(self.mf, include_ghost=False)
        return stat

    def get_reference_energy(self):
        return self.mf.kernel()

    @property
    def fock(self):
        return self.mf.get_fock()

    def get_xc_energy(self):
        e_xc = self.vhf.exc # see the source code energy_elec() in file pyscf/dft/rks.py
        return e_xc

    def int2e_sph(self, cart=False):  # pragma: no cover
        if cart:
            intor = 'int2e_cart'
        else:
            intor = 'int2e_sph'
        atm = self.mol._atm
        bas = self.mol._bas
        env = self.mol._env
        # 8-fold permutation symmetry
        _eri = gto.moleintor.getints4c(intor, atm, bas, env, aosym='s8')
        return _eri

    @property
    def eri(self):
        if not hasattr(self, '_eri'):
            self._eri = self.int2e_sph()
        return self._eri

    @property
    def h1e(self):
        """ 1-e part of Fock matrix """
        return self.mf.get_hcore(self.mol)



class io(object):

    def __init__(self, mol):
        self.mol = mol

    def orbital(self, coeffs, grids=[80,80,80], idx=None, label=None):
        """
        coeff : 2D array
                coeff[0] -- orbital 1
                coeff[1] -- orbital 2
                ...
        """
        mol = self.mol
        nx, ny, nz = grids
        cb = Cube(mol, nx, ny, nz) #, resolution)
        # Compute density on the .cube grid
        coords = cb.get_coords()
        ngrids = cb.get_ngrids()
        blksize = min(8000, ngrids)
        data = []
        nmo = len(coeffs)
        if idx is None:
            idx = [ i for i in range(nmo) ]
            print(' all orbitals are selected')
        else:
            nmo = len(idx)
            print(' selected orbital idx: ', idx)

        fmt = '%%0%dd'%( len(str(nmo)) )
        orig = cb.boxorig # cell origin
        cell = cb.box

        if label is not None:
            if '/' in label:
                lbt = label[::-1]; i0 = lbt.index('/')
                fd = lbt[i0:][::-1]
                if not os.path.exists(fd):
                    os.system('mkdir -p %s'%fd)

        for ir,i in enumerate(idx): #range(nmo):
            print(' now working on orbital: %d/%d'%(ir+1,nmo))
            orb_on_grid = np.empty(ngrids)
            for ip0, ip1 in lib.prange(0, ngrids, blksize):
                ao = numint.eval_ao(mol, coords[ip0:ip1])
                orb_on_grid[ip0:ip1] = np.dot(ao, coeffs[:,i]) # each column corresp. to MO coeffs
            orb_on_grid = orb_on_grid.reshape(cb.nx,cb.ny,cb.nz)
            data.append( orb_on_grid )

            if label is not None:
                outfile = label + '_' + fmt%(i+1) + '.cube'
                cb.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')

        if label is None:
            return orig, cell, data

