#!/usr/bin/env python

"""
This module defines an ASE interface to MOPAC.

"""
import os,sys,re
import numpy as np
import ase.io as aio
import ase
import io2

iu = io2.Units()

class mopac(object):

    def __init__(self, obj, label=None, method='PM7', task='OPT', \
                 ias_fix=[], ias_relax=[], iwrite=False):
        """
        atomic indices in either `ias_fix or `ias_relax starts from 1
        """

        self.method = method
        self.task = task
        self.iwrite = iwrite
        self.obj = obj

        typ = type(obj)
        if typ is str:
            suffix = obj[-3:]
            if obj[-3:] in ['com','gjf']:
                fmt = 'gaussian'
            elif suffix in ['out','log']:
                fmt = 'gaussian-out'
            else:
                fmt = suffix
            atoms = aio.read(obj, format=fmt)
            label = obj[:-4]
        elif typ is ase.Atoms:
            atoms = obj
            assert label != None, '#ERROR: plz specify `label'
        else:
            raise '#ERROR: invalid type of `obj'

        self.atoms = atoms
        self.label = label

        na = len(atoms)
        naf = len(ias_fix)
        nar = len(ias_relax)
        assert naf == 0 or nar == 0
        if naf == 0 and nar != 0:
            ias_fix = np.setdiff1d( list(range(1,na+1)), ias_relax )
        self.ias_fix = ias_fix


    def run(self):

        label = self.label
        task = self.task
        method = self.method
        atoms = self.atoms

        # Build string to hold .mop input file:
        s = ''
        if task in ['opt', 'OPT']:
            s += '%s BFGS'%method
        elif task in ['spe','e','SPE','E',]:
            s += '%s 1SCF'%method
        else:
            print('#ERROR: non-supported `task %s'%task); sys.exit(2)

        s += '\nASE calculation\n\n'

        # Write coordinates
        icnt = 0
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            icnt += 1
            if icnt in self.ias_fix:
                # To freeze an atom in MOPAC, put "0" after each of {x,y,z};
                # To relax an atom, put "1" instead
                s += ' {0:2} {1} 0 {2} 0 {3} 0\n'.format(symbol, *xyz)
            else:
                s += ' {0:2} {1} 1 {2} 1 {3} 1\n'.format(symbol, *xyz)

        with open(label + '.mop', 'w') as f:
            f.write(s)

        fout0 = label + '.out'; fout = label + '.omf'
        command = "/opt/mopac/MOPAC2016.exe %s.mop 2>/dev/null"%label #,fout)
        iok = os.system(command)
        if iok:
            raise '#ERROR: mopac failed !!'
        else:
            cmd = 'mv %s %s'%(fout0, fout)
            iok2 = os.system(cmd)

        self.fout = fout

        cmd = "sed -n '/                             CARTESIAN COORDINATES/,/Empirical Formula:/p' %s"%fout
        conts = os.popen(cmd).read().strip().split('\n')[2:-3]
        symbs = []; ps = []
        na = len(conts)
        atoms_U = ase.Atoms([], cell=[1,1,1])
        for k in range(na):
            tag, symb, px, py, pz = conts[k].strip().split()
            atoms_U.append(ase.Atom(symb, [px, py, pz])); symbs.append(symb)

        self.atoms = atoms_U


    def get_enthalpy(self):
        return eval(os.popen( \
             "grep 'FINAL HEAT OF FORMATION =' %s | \
              awk '{print $6}'"%(self.fout)).read())

    def get_total_energy(self, iAE=True, unit='Hartree'):
        # read H first
        cmd = "grep '          TOTAL ENERGY            =' %s | tail -n 1"%self.fout

        E0 = eval( os.popen(cmd).read().strip().split()[-2] )
        if unit in ['hartree', 'Hartree',]:
            E0 = E0/iu.h2e
        else:
            raise '#ERROR: unknown unit'
        #note = 'Id_%s H(PM7) = '%f[:-4].split('_')[-1] + '%.4f'%E + ' kcal/mol'
        energy = E0
        if self.iAE:
            energy = io2.data.get_ae(atoms, E0, 'pm7')
        self.energy = energy


    def get_mo(self, unit='Hartree'):

        # read HOMO, LUMO energies
        cmd = "grep 'HOMO LUMO ENERGIES (EV) =' %s | tail -n 1"%self.fout
        homo, lumo = np.array([ eval(ej) for ej in os.popen(cmd).read().strip().split()[5:7] ])/iu.h2e
        gap = lumo - homo
        self.homo = homo
        self.lumo = lumo
        self.gap = gap


    def write_exyz(self):
        """
        write extended xyz  file
        """
        label = self.label

        idx = int( re.findall('(\d+)', label)[0] )
        atoms = self.atoms
        ps = atoms.positions

        fo = label + '_U.XYZ'
        with open(fo,'w') as foh:
            foh.write('%d\n'%na)
            # set all properties except H to 0.0
            foh.write('gdb %d 0. 0. 0. 0. 0. %.6f %.6f %.6f 0. 0. 0. 0. %.6f 0. 0.\n'%(idx, homo, lumo, gap, E))
            for k in range(na):
                px, py, pz = ps[k]
                if k < na - 1:
                    foh.write('%s %s %s %s 0.0\n'%(symbs[k], px, py, pz))
                else:
                    foh.write('%s %s %s %s 0.0'%(symbs[k], px, py, pz))


    def write_g09(self, np=6, qcl='OPT(Loose) B3LYP/6-31G(D)'):
        """
        convert mopac output file to Gaussian input file
        """

        atoms = self.atoms
        if (np is None) or (qcl is None):
            raise '#ERROR: to write com file, plz specify `np and `qcl'
        s = ''
        note = 'Id_%s'%self.label.split('_')[-1]
        s += '%%nproc=%d\n# %s\n\n%s\n\n0 1\n'%(np, qcl, note)
        icnt = 0
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            icnt += 1
            # To freeze atom in G09, put "-1" right after the atomic symbol;
            # To relax, put "0" instead
            if icnt in self.ias_fix:
                s += '{0:2} -1 {1} {2} {3}\n'.format(symbol, *xyz)
            else:
                s += '{0:2}  0 {1} {2} {3}\n'.format(symbol, *xyz)
        s += '\n'
        iok = os.system("echo '%s' > %s"%(s, self.label +'_U.com'))


    def write_xyz(self):
        aio.write(self.label+'_U.xyz', self.atoms, format='xyz')


    def write_sdf(self):
        import aqml.cheminfo.OEChem as cio
        assert type(self.obj) is str and self.obj[-3:].lower() in ['sdf', 'mol',]
        s = cio.StringM(self.obj)
        s.coords = self.atoms.positions
        cio.write_sdf(s, self.label + '_U.sdf')


if __name__ == "__main__":
    args = sys.argv[1:]
    idx = 0

    nproc = 1
    if '-np' in args:
        nproc = int(args[ args.index('-np') + 1 ])
        idx += 2

    qcl = None
    if '-qcl' in args:
        qcl = args[ args.index('-qcl') + 1 ]
        idx += 2

    fs = args[idx:]
    for f in fs:
        obj = mopac(f, label=None, method='PM7', task='OPT', \
                    ias_fix=[], ias_relax=[])
        obj.run()
        obj.write_xyz()

