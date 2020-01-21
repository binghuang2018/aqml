#!/usr/bin/env python

"""This module defines an ASE interface to MOPAC.

Set $ASE_MOPAC_COMMAND to something like::

    LD_LIBRARY_PATH=/path/to/lib/ \
    MOPAC_LICENSE=/path/to/license \
    /path/to/MOPAC2012.exe PREFIX.mop 2> /dev/null

"""

import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye
import ase.io as aio
import cheminfo.rw.xyz as rwx

def get_index(lines, pattern):
    for i, line in enumerate(lines):
        if line.find(pattern) != -1:
            return i


class MopacParser(object):

    def __init__(self, label):
        """Read the Atoms from the output file stored as list of str in lines.
        Parameters:

            lines: list of str
        """
        f = label + '.out'
        with open(f) as fid:
            lines = fid.readlines()
        self.lines = lines
        # first try to read from final point (last image)

        symbols = []
        positions = []
        i = get_index(lines, 'FINAL  POINT  AND  DERIVATIVES')
        if True: #i is None:  # XXX should we read it from the input file?
            cmd = "sed -n '/                        CARTESIAN COORDINATES/,/Empirical Formula:/p' %s"%f
            conts = os.popen(cmd).read().strip().split('\n')[2:-3]
            na = len(conts)
            for k in range(na):
                tag, symbol, px, py, pz = conts[k].strip().split()
                symbols.append(symbol)
                positions.append([ float(val) for val in [px, py, pz] ])
        else:
            #assert 0, 'Not implemented'
            print(' ** Gradients found')
            lines1 = lines[i:]
            i = get_index(lines1, 'CARTESIAN COORDINATES')
            j = i + 2
            while not lines1[j].isspace():  # continue until we hit a blank line
                l = lines1[j].split()
                symbols.append(l[1])
                positions.append([float(c) for c in l[2: 2 + 3]])
                j += 1
        self.atoms = Atoms(symbols=symbols, positions=positions)

    def center(self):
        """ centre all atoms """
        return

    @property
    def properties(self):
        if not hasattr(self, '_property'):
            self._prop = self.get_properties()
        return self._prop

    def get_properties(self, igrad=False):
        """Read the results, such as energy, forces, eigenvalues, etc.
        """
        results = {}
        lines = self.lines
        for i, line in enumerate(lines):
            if line.find('TOTAL ENERGY') != -1:
                results['e'] = results['energy'] = float(line.split()[3])
            elif line.find('FINAL HEAT OF FORMATION') != -1:
                results['h'] = results['heat_of_formation'] = float(line.split()[5]) * kcal / mol
            elif line.find('NO. OF FILLED LEVELS') != -1:
                nspins = 1
                no_occ_levels = int(line.split()[-1])
                results['no_occ_levels'] = no_occ_levels
            elif line.find('NO. OF ALPHA ELECTRON') != -1:
                nspins = 2
                no_alpha_electrons = int(line.split()[-1])
                no_beta_electrons = int(lines[i+1].split()[-1])
                results['magmom'] = abs(no_alpha_electrons -
                                        no_beta_electrons)
            elif igrad and line.find('FINAL  POINT  AND  DERIVATIVES') != -1:
                forces = [-float(line.split()[6])
                          for line in lines[i + 3:i + 3 + 3 * len(atoms)]]
                results['forces'] = np.array(
                    forces).reshape((-1, 3)) * kcal / mol
            elif line.find('EIGENVALUES') != -1:
                if line.find('ALPHA') != -1:
                    j = i + 1
                    eigs_alpha = []
                    while not lines[j].isspace():
                        eigs_alpha += [float(eps) for eps in lines[j].split()]
                        j += 1
                    results['eigs'] = eigs_alpha
                elif line.find('BETA') != -1:
                    j = i + 1
                    eigs_beta = []
                    while not lines[j].isspace():
                        eigs_beta += [float(eps) for eps in lines[j].split()]
                        j += 1
                    eigs = np.array([eigs_alpha, eigs_beta]).reshape(2, 1, -1)
                    results['eigs'] = eigs
                else:
                    eigs = []
                    j = i + 1
                    while not lines[j].isspace():
                        eigs += [float(e) for e in lines[j].split()]
                        j += 1
                    results['eigs'] = np.array(eigs).reshape(1, 1, -1)
            elif line.find('DIPOLE   ') != -1:
                results['dipole'] = np.array(
                    lines[i + 3].split()[1:1 + 3], float) * Debye
        return results

    def write(self, f, props=['e','h'], unit='eV'):
        _unit = unit.lower()
        if _unit in ['ev', 'default']:
            const = 1.0
        elif _unit in ['kcal']:
            const = 1./(kcal / mol)
        else:
            raise Exception('unknown unit!')
        pdic = dict(zip(props, [ self.properties[p]*const for p in props ]))
        rwx.write_xyz_simple(f, self.atoms, pdic)



class MOPAC(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole', 'magmom']

    # `nohup is the key to stop the program from hanging over there instead
    # of a complete exit after normal exit
    command = 'nohup mopac PREFIX.mop 2>/dev/null'

    default_parameters = dict(
        method='PM7',
        task='1SCF GRADIENTS',
        relscf=0.0001)

    methods = ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+',
               'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PMEP', 'PM7',
               'PM7-TS', 'RM1']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mopac', atoms=None, **kwargs):
        """Construct MOPAC-calculator object.

        Parameters:

        label: str
            Prefix for filenames (label.mop, label.out, ...)

        Examples:

        Use default values to do a single SCF calculation and print
        the forces (task='1SCF GRADIENTS'):

        >>> from ase.build import molecule
        >>> from ase.calculators.mopac import MOPAC
        >>> atoms = molecule('O2')
        >>> atoms.calc = MOPAC(label='O2')
        >>> atoms.get_potential_energy()
        >>> eigs = atoms.calc.get_eigenvalues()
        >>> somos = atoms.calc.get_somo_levels()
        >>> homo, lumo = atoms.calc.get_homo_lumo_levels()

        Use the internal geometry optimization of Mopac:

        >>> atoms = molecule('H2')
        >>> atoms.calc = MOPAC(label='H2', task='GRADIENTS')
        >>> atoms.get_potential_energy()

        Read in and start from output file:

        >>> atoms = MOPAC.read_atoms('H2')
        >>> atoms.calc.get_homo_lumo_levels()

        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters

        # Build string to hold .mop input file:
        s = p.method + ' ' + p.task + ' '

        if p.relscf:
            s += 'RELSCF={0} '.format(p.relscf)

        # Write charge:
        charge = atoms.get_initial_charges().sum()
        if charge != 0:
            s += 'CHARGE={0} '.format(int(round(charge)))

        magmom = int(round(abs(atoms.get_initial_magnetic_moments().sum())))
        if magmom:
            s += (['DOUBLET', 'TRIPLET', 'QUARTET', 'QUINTET'][magmom - 1] +
                  ' UHF ')

        s += '\nTitle: ASE calculation\n\n'

        # Write coordinates:
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            s += ' {0:2} {1} 1 {2} 1 {3} 1\n'.format(symbol, *xyz)

        for v, p in zip(atoms.cell, atoms.pbc):
            if p:
                s += 'Tv {0} {1} {2}\n'.format(*v)

        with open(self.label + '.mop', 'w') as f:
            f.write(s)

    def get_spin_polarized(self):
        return self.nspins == 2

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.out') as f:
            lines = f.readlines()

        self.parameters = Parameters(task='', method='')
        p = self.parameters
        parm_line = self.read_parameters_from_file(lines)
        for keyword in parm_line.split():
            if 'RELSCF' in keyword:
                p.relscf = float(keyword.split('=')[-1])
            elif keyword in self.methods:
                p.method = keyword
            else:
                p.task += keyword + ' '

        p.task.rstrip()
        self.read_results()

    def read_results(self):
        """Read the results, such as energy, forces, eigenvalues, etc.
        """
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError
        parser = MopacParser(self.label)
        self.parser = parser
        self.atoms = parser.atoms
        self.results = parser.get_properties()

    def read_parameters_from_file(self, lines):
        """Find and return the line that defines a Mopac calculation

        Parameters:

            lines: list of str
        """
        for i, line in enumerate(lines):
            if line.find('CALCULATION DONE:') != -1:
                break

        lines1 = lines[i:]
        for i, line in enumerate(lines1):
            if line.find('****') != -1:
                return lines1[i + 1]

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.results['eigs'][spin, kpt]

    def get_homo_lumo_levels(self):
        eigs = self.eigenvalues
        if self.nspins == 1:
            nocc = self.results['no_occ_levels']
            return np.array([eigs[0, 0, nocc - 1], eigs[0, 0, nocc]])
        else:
            na = self.results['no_alpha_electrons']
            nb = self.results['no_beta_electrons']
            if na == 0:
                return None, eigs[1, 0, nb - 1]
            elif nb == 0:
                return eigs[0, 0, na - 1], None
            else:
                eah, eal = eigs[0, 0, na - 1: na + 1]
                ebh, ebl = eigs[1, 0, nb - 1: nb + 1]
                return np.array([max(eah, ebh), min(eal, ebl)])

    def get_somo_levels(self):
        assert self.results['nspins'] == 2
        na, nb = self.results['no_alpha_electrons'], self.results['no_beta_electrons']
        if na == 0:
            return None, self.results['eigs'][1, 0, nb - 1]
        elif nb == 0:
            return self.results['eigs'][0, 0, na - 1], None
        else:
            return np.array([self.results['eigs'][0, 0, na - 1],
                             self.results['eigs'][1, 0, nb - 1]])

    def get_heat_of_formation(self):
        """hof: heat of formation of the final config
        as reported in the Mopac output file
        """
        return self.results['h']

    @property
    def force(self):
        assert 'force' in self.results
        return self.results['force']

    @property
    def energy(self):
        return self.results['energy']


if __name__ == "__main__":

    import sys, time
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', nargs='*', default=['PM7'], type=str, help='Keywords to be used to control MOPAC. E.g., "--task PM7 BFGS')
    parser.add_argument('--task', nargs='*', default=['BFGS', 'GNORM=0.01'], type=str, help='')
    #parser.add_argument('--iar', nargs='*', type=str, help='idx of atom to be relaxed, specified by atom type, E.g., "H"')
    parser.add_argument('ipts', nargs='*', type=str, help='Input files to be processed')

    args = parser.parse_args() # sys.argv[1:] )
    print( args.ipts )

    for f in args.ipts:
        print(' -- file : ', f)
        fn = f[:-4]; fmt = f[-3:]
        if fmt == 'out':
            obj = MopacParser(fn)
        elif fmt in ['xyz','sdf']:
            atoms = aio.read(f)
            c = MOPAC( label=f[:-4], method=' '.join(args.method), task=' '.join(args.task) )
            atoms.calc = c
            e = atoms.get_potential_energy()
            obj = c.parser
        else:
            raise Exception('format not supported')

        if os.path.exists(fn+'.xyz'):
            os.system('cp %s.xyz %s.xyz_0'%(fn,fn))
        obj.write(fn+'.xyz', props=['e','h'], unit='kcal')


