
import ase.data as ad
from ase import Atom, Atoms
from pyscf import gto, dft, scf, cc, mp
from pyscf.cc import uccsd_t
#from pyscf.geomopt import berny_solver
import cheminfo.core as mcc

from pyscf.tools.cubegen import *
import ase.io as aio
import numpy as np

_mult = {1:2, 3:2, 4:1, 5:2, 6:3, 7:4, 8:3, 9:2, \
        11:2, 12:0, 13:2, 14:3, 15:4, 16:3, 17:2,\
        33:4, 34:3, 35:2, 53:2}

class molecules(object):
    def __init__(self, fs, basis, etype='HF', spin=0, nbatch=1):
        es = []
        n = len(fs)
        for i,f in enumerate(fs):
            if (i+1)%nbatch == 0: print(' i/nt = ', i+1, n)
            m = aio.read(f)
            o = molecule(m, basis, spin)
            if etype in ['x', 'X']:
                o.get_dft_energy(xc='HF')
                es.append( o.e_xc )
            else:
                raise Exception('Todo')
        self.es = np.array(es)

class molecule(object):

    def __init__(self, obj, basis, spin=0):
        self.spin = spin
        s = '' #'O 0 0 0; H 0 0 1; H 0 1 0'
        if isinstance(obj, (list,tuple)):
            assert len(obj)==2
            zs, coords = obj
        elif isinstance(obj, Atoms):
            zs = obj.numbers; coords = obj.positions
        else:
            raise Exception('unknown object')
        for i,zi in enumerate(zs):
            x, y, z = coords[i]
            si = mcc.chemical_symbols[zi]
            s += '%s %.8f %.8f %.8f; '%(si, x, y, z)
        _s = s[:-2]
        na = len(zs)
        if na == 1:
            spin = _mult[zs[0]] - 1
        self.mol = gto.M(atom=_s, basis=basis, verbose=0, spin=spin)

    def get_dft_energy(self,xc='b3lyp',optg=False):
        if self.spin == 0:
            mf = dft.RKS(self.mol)
        else:
            mf = dft.UKS(self.mol)
        mf.xc = xc
        if optg:
            berny_solver.optimize(mf, include_ghost=False)
        e = mf.kernel()
        self.mf = mf
        vhf = mf.get_veff()
        self.e_xc = vhf.exc # see the source code energy_elec() in file pyscf/dft/rks.py
        return e

    def get_hf_energy(self):
        mf = scf.RHF(self.mol)
        e = mf.kernel()
        return e

    def get_mp2_energy(self):
        if self.spin == 0:
            mf = scf.RHF(self.mol)
        else:
            mf = scf.UHF(self.mol)
        e0 = mf.kernel()
        mp2 = mp.MP2(mf)
        ecorr = mp2.kernel()[0]
        return e0, ecorr, e0+ecorr

    def get_ccsdt_energy(self):
        #if self.spin == 0:
        mf = scf.RHF(self.mol).run()
        mycc = cc.UCCSD(mf)
        t = mycc.kernel()
        e = uccsd_t.kernel(mycc, mycc.ao2mo())
        return e


class io(object):

    def __init__(self, mol):
        self.mol = mol

    def orbital(self, coeffs, grids=[80,80,80], label=None):
        """
        coeff : 2D array
                coeff[0] -- orbital 1
                coeff[1] -- orbital 2
                ...
        """
        mol = self.mol
        nx, ny, nz = grids
        cc = Cube(mol, nx, ny, nz) #, resolution)
        # Compute density on the .cube grid
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
        blksize = min(8000, ngrids)
        data = []
        nmo = len(coeffs)
        for i in range(nmo):
            orb_on_grid = numpy.empty(ngrids)
            for ip0, ip1 in lib.prange(0, ngrids, blksize):
                ao = numint.eval_ao(mol, coords[ip0:ip1])
                orb_on_grid[ip0:ip1] = numpy.dot(ao, coeffs[:,i]) # each column corresp. to MO coeffs
            orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)
            data.append( orb_on_grid )
        fmt = '%%0%dd'%( len(str(nmo)) )
        if label is not None:
            for i in range(nmo):
                outfile = label + '_' + fmt%(i+1) + '.cube'
                cc.write(data[i], outfile, comment='Orbital value in real space (1/Bohr^3)')

        orig = cc.boxorig # cell origin
        cell = cc.box
        return orig, cell, data


