
import os, sys
#import subprocess
#from subprocess import Popen as run

#import chemcoord as cc

import aqml.cheminfo.rw.xyz as crx

import numpy as np
import re


defaults = {}
defaults['molpro_exe'] = '/apps/molpro/molpro2018/molprop_2018_0_linux_x86_64_i8/bin/molpro'
defaults['task'] = '' #single point calc
defaults['charge'] = 0
defaults['multiplicity'] = 1
defaults['forces'] = False
defaults['wfn_symmetry'] = 1
defaults['title'] = ''
defaults['etol'] = 1e-6  # Convergence criterium for the energy
defaults['gtol'] = 6e-4  # Convergence criterium for the gradient
defaults['max_iter'] = 100
defaults['coord_fmt'] = '.6f'

defaults['num_procs'] = 1
defaults['num_threads'] = 1
defaults['mem_per_proc'] = '100,M'


def calculate(obj, hamiltonian, basis, molpro_exe=None,
              charge=defaults['charge'],
              forces=defaults['forces'],
              title=defaults['title'],
              multiplicity=defaults['multiplicity'],
              wfn_symmetry=defaults['wfn_symmetry'],
              num_procs=None, mem_per_proc=None, 
              task=defaults['task']):
    """Calculate the energy of a molecule using Molpro.

    Args:
        obj (str): 
            it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
            But 'RASSCF' and 'CASPT2' not yet implemented.
        basis (str): {basis}
        molpro_exe (str): {molpro_exe}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}
        num_procs (int): {num_procs}
        mem_per_proc (str): {mem_per_proc}

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    if molpro_exe is None:
        molpro_exe = defaults['molpro_exe']
    if num_procs is None:
        num_procs = defaults['num_procs']
    if mem_per_proc is None:
        mem_per_proc = defaults['mem_per_proc']

    assert os.path.isfile(obj) and obj[-3:] == 'xyz'

    atoms = crx.read_xyz_simple(obj)
    symbs, coords = atoms
    na = len(symbs)
    s = '%d\n\n'%na
    for i in range(na): #.join( file(obj).readlines() )
        xi,yi,zi = coords[i]
        s += '%2s %12.8f %12.8f %12.8f\n'%(symbs[i],xi,yi,zi)
    fn = obj[:-4]
    title = fn

    input_str = generate_inp(
        s=s,
        hamiltonian=hamiltonian, basis=basis, charge=charge,
        forces=forces,
        title=title, multiplicity=multiplicity,
        wfn_symmetry=wfn_symmetry)

    ipf = fn + '.inp'
    opf = fn + '.out'
    dirname = os.path.dirname(ipf)
    #if dirname != '':
    #    os.makedirs(dirname, exist_ok=True)
    if not os.path.isfile(ipf):
        with open(ipf, 'w') as f:
            f.write(input_str)

    if not os.path.isfile(opf):
        iok = os.system('%s -d /scratch/$USER -n %d %s'%(molpro_exe, num_procs, ipf))
        assert not iok

    return parse_output(na, atoms, opf)


def parse_output(na, atoms, opf):
    """Parse a molpro output file.

    Args:
        opf (str):

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """

    cmdout1 = lambda cmd: os.popen(cmd).read().strip()
    cmdout = lambda cmd: os.popen(cmd).read().strip().split('\n')

    output = {}

    cmd = "grep -E '^\s*!' %s | awk '{print $1}' | uniq"%opf
    hs = cmdout(cmd) # Halmitonian (strs)
    h = hs[-1][1:] # hs[-1] may be, e.g., '!MP2'
    if h[-2:] in ['KS','HF']:
        e = cmdout1("grep -E '^\s*!(RHF|UHF|RKS)\s\s*STATE 1.1 Energy' %s | tail -1 | awk '{print $NF}'"%opf)
    else:
        cmd = "grep -E '^\s*!%s total energy' %s | tail -1 | awk '{print $NF}'"%(h,opf); #print cmd
        es = cmdout1(cmd); #print es
        e = es[-1]
    print ' Get %s total energy: %s'%(h, e)
    output['energy'] = float(e)

    # get gradient (default unit: Hartree/Bohr)
    sgrad = cmdout("grep -A%d '%s GRADIENT FOR STATE 1.1' %s | tail -%d"%(na+3, h, opf, na))
    #print sgrad
    output['gradient'] = np.array([ si.strip().split()[1:] for si in sgrad ], np.float) # /b2a
    
    # get geometry
    #atoms = 
    output['symbols'] = np.array(atoms[0])
    output['coords'] = np.array(atoms[1])

    return output


def generate_inp(s, hamiltonian, basis,
                        charge=defaults['charge'],
                        forces=defaults['forces'],
                        title=defaults['title'],
                        multiplicity=defaults['multiplicity'],
                        wfn_symmetry=defaults['wfn_symmetry'],
                        mem_per_proc=None,
                        task=defaults['task']):
    """Generate a molpro input file.

    Args:
        atoms:
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}
        mem_per_proc (str): {mem_per_proc}


    Returns:
        str : Molpro input.
    """

    if mem_per_proc is None:
        mem_per_proc = defaults['mem_per_proc']
    
    sf = 'forces' if forces else ''

    out = """\
*** %s
memory,%s

basis={%s} ! basis

geomtype=xyz
geometry = {
%s
}

%s !hamiltonian
%s

%s
---
"""%(title, mem_per_proc, basis, s, hamiltonian, sf, task)

    return out

