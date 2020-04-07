
import ase.data as ad
from ase import Atom, Atoms
from pyscf import gto, dft, scf, cc, mp
from pyscf.cc import uccsd_t
from pyscf.geomopt import berny_solver

from pyscf.tools.cubegen import *

class molecule(object):

    def __init__(self, m, basis, spin=0):
        self.spin = spin
        sm = '' #'O 0 0 0; H 0 0 1; H 0 1 0'
        for ai in m:
            x, y, z = ai.position
            sm += '%s %.8f %.8f %.8f; '%(ai.symbol, x, y, z)
        sm = sm[:-2]
        self.mol = gto.M(atom=sm, basis=basis, verbose=0, spin=spin)

    def get_dft_energy(self,xc='b3lyp',optg=False):
        if self.spin == 0:
            mf = dft.RKS(self.mol)
        else:
            mf = dft.UKS(self.mol)
        mf.xc = xc
        if optg:
            berny_solver.optimize(mf, include_ghost=False)
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
                orb_on_grid[ip0:ip1] = numpy.dot(ao, coeffs[i])
            orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)
            data.append( orb_on_grid )
        fmt = '%%0%dd'%( len(str(nmo)) )
        if label is not None:
            for i in range(nmo):
                outfile = label + '_' + fmt%(i+1) + '.cube'
                cc.write(data[i], outfile, comment='Orbital value in real space (1/Bohr^3)')
        return data


