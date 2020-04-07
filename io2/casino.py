#!/usr/bin/env python

import aqml.cheminfo as co
import aqml.cheminfo.core as cc
import os
import numpy as np

T, F = True, False


def get_up_down(*path):
    """Get up and down electron numbers from ORCA output file.
    Number of up&down electrons required in CASINO input.
    """

    regexp1 = re.compile('Multiplicity           Mult            ....\s+(?P<mult>\d+)')
    regexp2 = re.compile('Number of Electrons    NEL             ....\s+(?P<elec>\d+)')
    with open(os.path.join(*path, 'mol.out'), 'r') as orca_out:
        for line in orca_out:
            m1 = re.search(regexp1, line)
            if m1:
                mult = int(m1.group('mult'))
            m2 = re.search(regexp2, line)
            if m2:
                elec = int(m2.group('elec'))
    neu = (elec + mult - 1)//2
    ned = (elec - mult + 1)//2
    return neu, ned

def get_nelec(*path):
    """Get total electron numbers from ORCA output file.
    """
    regexp = re.compile('Number of Electrons    NEL             ....\s+(?P<elec>\d+)')
    with open(os.path.join(*path, 'mol.out'), 'r') as orca_out:
        for line in orca_out:
            m = re.search(regexp, line)
            if m:
                elec = int(m.group('elec'))
    return elec


def get_irrep(*path):
    """The symmetry of the initial guess is 1-Ag
    """
    table = {
        'Ag': 0, 'B1g': 1, 'B2g': 2, 'B3g': 3, 'Au': 4, 'B1u': 5, 'B2u': 6, 'B3u': 7,
        'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3,
        'A': 0, 'B': 1,
    }
    irrep = 0
    regexp = re.compile('The symmetry of the initial guess is \d-(?P<symmetry>\w+)')
    with open(os.path.join(*path, 'mol.out'), 'r') as orca_out:
        for line in orca_out:
            m = re.search(regexp, line)
            if m:
                symmetry = m.group('symmetry')
                irrep = table[symmetry]
    return irrep

def pp_basis(basis):
    "check if basis is pseudopotential basis"
    return basis in ('aug-cc-pVDZ-CDF', 'aug-cc-pVTZ-CDF', 'aug-cc-pVQZ-CDF', 'aug-cc-pV5Z-CDF')

def write_orca_input(mol, method='MP2-CASSCF(N,M)', basis='cc-pvtz', nproc=1):
    if 'casscf' in method.lower():
        irrep = get_irrep(file_res_hf)

        if method in ('MP2', 'OO-RI-MP2'):
            moinp = 'mol.mp2nat'
        else:
            moinp = 'mol.qro'

        with open('casscf.inp', 'w') as f:
            offset = method.find('(') + 1
            nel, norb = map(int, method[offset:-1].split(','))
            f.write(open('orca_casscf.tmpl').read().format(
                basis=basis,
                molecule=mol,
                moinp=moinp,
                irrep=irrep,
                charge=charge,
                multiplicity=multiplicity,
                nel=nel,
                norb=norb,
                procs='! PAL{}'.format(nproc),
            ))

        postprocess = """
                  cd "$(dirname "{output}")"
                  $(which orca_2mkl) casscf -molden &&
                  ln -s casscf.molden.input mol.molden.input"""
        if pp_basis(basis):
            print('molden2qmc.py 3 "{input}" "{output}" --pseudoatoms all')
        else:
            print('molden2qmc.py 3 "{input}" "{output}"')


class cats(object):

    dct_a = {'C':[4,2], 'N':[5,2], 'O':[5,3], 'F':[5,4], 'H':[1,0] }

    def __init__(self, _obj, ipsp=F, ispin=F, order_trunk=3, mult=None, \
            order_jastrow = {'trun':3, 'mu':8, 'chi':8, 'en':4, 'ee':4}, \
                 order_bf = {'mu':9, 'eta':9, 'en':3, 'ee':3} ):
        self.ipsp = ipsp
        if isinstance(_obj, str):
            if _obj in co.chemical_symbols:
                label = _obj
                z1 = co.chemical_symbols.index(_obj)
                obj = cc.molecules( cc.catoms([1], [z1], [[0.,0.,0.]]) )
            else:
                assert os.path.exists(_obj)
                label = _obj[:-4]
                obj = cc.molecules(_obj)
        self.label = label
        na = np.sum(obj.nas) #len(np.unique(zs)) # num atom type
        zsu = obj.zsu
        nat = len(zsu)
        self.nat = nat  # number of atom types
        nzs = obj.nzs[0]
        self.nzs = nzs
        ias = np.arange(na)
        if na == 1:
            neu, ned = self.dct_a[ obj.symbols[0] ]
        else:
            tne = np.sum(obj.zs)
            if ispin or ( tne % 2 != 0 ):
                assert mult
                neu = (elec + mult - 1)//2
                ned = (elec - mult + 1)//2
            else:
                ispin = F
                neu, ned = tne/2, tne/2
        self.ispin = ispin
        self.neu = neu
        self.ned = ned
        self.spin_dep = 1 if neu != ned else 0 ##
        self.type_eN = 0 if ipsp else 1 #
        lats = []
        for iat in range(nat):
            nai = nzs[iat]
            atsi = ias[ obj.zsu[iat] == obj.zs ] + 1
            latsi = '   '.join( [ '%d'%ai for ai in atsi ] )
            lats.append( latsi )
        self.lats = lats
        self.order_jastrow = order_jastrow
        self.order_bf = order_bf
        self.order_trunk = order_trunk

    def write(self, wd='.', bf=F):
        opf = wd+'/'+self.label
        print(' output file prefix: ', opf)
        generate_inputs(self, opf, bf=bf)


def generate_jastraw(a):
    """
    vars
    -----------------------\
    expansion_order: 4-8,
    """

    sj = """ START HEADER
 No title given.
 END HEADER

 START VERSION
   1
 END VERSION

 START JASTROW
 Title
  None
 Truncation order C
   {order_jastrow[trun]}
 START U TERM
 Number of sets
   1
 START SET 1
 Spherical harmonic l,m
   0 0
 Expansion order N_u
   {order_jastrow[mu]}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   3.d0                1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET 1
 END U TERM
 START CHI TERM
 Number of sets
   {nat}""".format(order_jastrow=a.order_jastrow, spin_dep=a.spin_dep, nat=a.nat)

    for iat in range(a.nat):
        sj += """
 START SET {iset}
 Spherical harmonic l,m
   0 0
 Number of atoms in set
   {nat}
 Labels of the atoms in this set
   {latsi}
 Impose electron-nucleus cusp (0=NO; 1=YES)
   0
 Expansion order N_chi
   {order_jastrow[chi]}
 Spin dep (0->u=d; 1->u/=d)
   {spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   3.d0                 1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET {iset}""".format(iset=iat+1, nat=a.nzs[iat], latsi=a.lats[iat], order_jastrow=a.order_jastrow, spin_dep=a.spin_dep)

    sj += """
 END CHI TERM
 START F TERM
 Number of sets
   {nat}""".format(nat=a.nat)

    for iat in range(a.nat):
        sj += """
 START SET {iset}
 Number of atoms in set
   {nat}
 Labels of the atoms in this set
   {latsi}
 Prevent duplication of u term (0=NO; 1=YES)
   0
 Prevent duplication of chi term (0=NO; 1=YES)
   0
 Electron-nucleus expansion order N_f_eN
   {order_jastrow[en]}
 Electron-electron expansion order N_f_ee
   {order_jastrow[ee]}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   3.d0                 1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET {iset}""".format(iset=iat+1, nat=a.nzs[iat], latsi=a.lats[iat], order_jastrow=a.order_jastrow, spin_dep=a.spin_dep)

    sj += """
 END F TERM
 END JASTROW"""

    return sj


def generate_backflow(a):

    # version 2.12.1
    sbf = """
 START BACKFLOW
 Title
   None
 Truncation order
   %d
 START ETA TERM
 Expansion order
   %d
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   %d
 Cut-off radius ;     Optimizable (0=NO; 1=YES)
   4.50                            1
   4.50                            1
 Parameter ;          Optimizable (0=NO; 1=YES)
 END ETA TERM
 START MU TERM
 Number of sets
   %d"""%(a.order_trunk, a.order_bf['eta'], a.spin_dep, a.nat)

    lats = []
    for iat in range(a.nat):
        latsi = a.lats[iat]
        nai = a.nzs[iat]
        sbf += """
 START SET %d
 Number of atoms in set
   %d
 Labels of the atoms in this set
   %s
 Type of e-N cusp conditions (0->PP/cuspless AE; 1->AE with cusp)
   %d
 Expansion order
   %d
 Spin dep (0->u=d; 1->u/=d)
   %d
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   6.d0                            1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET %d"""%(iat+1, nai, latsi, a.type_eN, a.order_bf['mu'], a.spin_dep, iat+1)

    sbf += """
 END MU TERM
 START PHI TERM
 Number of sets
   %s"""%a.nat

    for iat in range(a.nat):
        nai = a.nzs[iat]
        latsi = a.lats[iat]
        sbf += """
 START SET %d
 Number of atoms in set
   %d
 Labels of the atoms in this set
   %s
 Type of e-N cusp conditions (0=PP; 1=AE)
   %d
 Irrotational Phi term (0=NO; 1=YES)
   0
 Electron-nucleus expansion order N_eN
   %d
 Electron-electron expansion order N_ee
   %d
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   %d
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   6.d0               1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET %d"""%(iat+1, nai, latsi, a.type_eN, a.order_bf['en'], a.order_bf['ee'], a.spin_dep, iat+1)

    sbf += """
 END PHI TERM
 START AE CUTOFFS
 Nucleus ; Set ; Cutoff length     ;  Optimizable (0=NO; 1=YES)
 """
    #for iat in range(a.nat):
    #    sbf += " %d    %d    0.200    1\n"%(iat+1, iat+1)
    #for i, _ in enumerate(a.na):
    #    sbf += '{i}         {i}         0.2                          1\n'.format(i=i+1))
    sbf += """ END AE CUTOFFS
 END BACKFLOW"""
    return sbf


def generate_input(a, task, dtdmc=2, bf=F, mdet=F, basis_type='gaussian', cycles=5):
    sbf = 'T' if bf else 'F'
    smd = 'T' if mdet else 'F'
    svmc = ''
    sdmc = ''
    method = 'emin'
    use_tmove = 'F'
    sbf = 'T' if bf else 'F'

    assert dtdmc in [1,2,4], '#ERROR: dtdmc not in [1,2,4]'
    weight_dmc = 1024.0 * dtdmc
    dt_dmc = 0.009259/dtdmc  # unit: 1/Ha

    if (not bf) or (not mdet):
        if task in ['vmc_opt']:
            nstep_equ_vmc = 5000
            nstep_vmc     = 1000000
            nconfig_w_vmc = 100000
            nblock_vmc = 10
        elif task in ['vmc_dmc']:
            nstep_equ_vmc = 5000
            nstep_vmc = 2048
            nconfig_w_vmc = 2048
            nblock_vmc = 1

            nstep_equ_dmc   = 10000
            nstep_stats_dmc = 50000
            nblock_dmc = 5
    else:
        raise Exception('Todo')

    if task in ['vmc_opt']:
        svmc = """
# VMC
vmc_equil_nstep   : {n1}           #*! Number of equilibration steps (Integer)
vmc_nstep         : {n2:<7}        #*! Number of steps (Integer)
vmc_nblock        : {nblock:<7}    #*! Number of checkpoints (Integer)
vmc_nconfig_write : {nconfig_w:<7}        #*! Number of configs to write (Integer)
vmc_decorr_period : 0              #*! VMC decorrelation period (0 - auto)
""".format(n1=nstep_equ_vmc, n2=nstep_vmc,  nblock=nblock_vmc, nconfig_w=nconfig_w_vmc)

        if bf:
            svmc += """
%block opt_plan                    #*! Multi-cycle optimization plan (Block)
1 method=varmin backflow=F det_coeff=F fix_cutoffs=T
"""
            for ib in range(2, nblock+1):
                svmc += """%d\n"""%ib
            svmc += """%endblock opt_plan\n"""
        else:
            svmc += """opt_cycles %d\n"""%(cycles)

    elif task in ['vmc_dmc']:
        sdmc = """
# DMC
dmc_equil_nstep   : {n1}    #*! Number of steps (Integer)
dmc_equil_nblock  : 1              #*! Number of checkpoints (Integer)
dmc_stats_nstep   : {n2}    #*! Number of steps (Integer)
dmc_stats_nblock  : {nblock}              #*! Number of checkpoints (Integer)
dmc_target_weight : {w}         #*! Total target weight in DMC (Real)
dtdmc             : {dt} #*! DMC time step (Real)
use_tmove         : {tmove}        #*! Casula nl pp for DMC (Boolean)
popstats          : T              #*! Collect population statistics (Boolean)
""".format( n1=nstep_equ_dmc, n2=nstep_stats_dmc, nblock=nblock_dmc, w=weight_dmc, dt=dt_dmc, tmove=use_tmove )
    else:
        raise Exception('Todo')

    st = """
#-------------------#
# CASINO input file #
#-------------------#

# {molecule} molecule (ground state)

# SYSTEM
neu               : {neu:<3}            #*! Number of up electrons (Integer)
ned               : {ned:<3}            #*! Number of down electrons (Integer)
periodic          : F              #*! Periodic boundary conditions (Boolean)
atom_basis_type   : {basis_type}    #*! Basis set type (text)

# RUN
runtype           : {runtype} #*! Type of calculation (Text)
newrun            : T              #*! New run or continue old (Boolean)
testrun           : F              #*! Test run flag (Boolean)
allow_nochi_atoms : T              #*! Permit atoms no chi (etc) terms (Boolean)

{svmc}

{sdmc}
# OPTIMIZATION
opt_method        : {method}       #*! Opt method (varmin/madmin/emin/...)
opt_jastrow       : T              #*! Optimize Jastrow factor (Boolean)
opt_detcoeff      : {mdet}         #*! Optimize determinant coeffs (Boolean)
opt_backflow      : {backflow}              #*! Optimize backflow parameters (Boolean)

""".format(neu=a.neu, ned=a.ned, basis_type=basis_type, method=method,\
         runtype=task, backflow=sbf, mdet=smd, svmc=svmc, sdmc=sdmc, molecule=a.label)

    st += """
# GENERAL PARAMETERS
use_gjastrow      : T              #*! Use a Jastrow function (Boolean)
backflow          : {backflow}              #*! Use backflow corrections (Boolean)
""".format(backflow=sbf)
    return st


def generate_inputs(a, label, bf=F, mdet=F, tasks=['vmc_opt','vmc_dmc']):
    for task in tasks:
        sj = generate_input(a, task, bf=bf, mdet=mdet)
        with open(label+'.%s.input'%task,'w') as fid: fid.write(sj)

    s1 = generate_jastraw(a)
    with open('%s.parameters.casl'%label, 'w') as fid: fid.write(s1)
    #if bf and mdet:
    #    raise Exception('It may be too expensive to optimize when both bf and mdet are T')
    if bf:
        s2 = '\n%s'%generate_backflow(a)
        s2 += """START MDET
Title
 multideterminant WFN generated from Orca output data

MD
  1
  1.00 1 0
END MDET"""
        with open('%s.correlation.data'%label, 'w') as fid: fid.write(s2)




if __name__ == "__main__":

    import sys

    fs = sys.argv[1:]

    for f in fs:
        a = cats(f)
        # obj, ipsp=F, ispin=F, order_trunk=3, mult=None, \
        #         order_jastrow = {'mu':8, 'chi':8, 'en':4, 'ee':4}, \
        #         order_bf = {'mu':9, 'eta':9, 'en':3, 'ee':3} )
        lb = f[:-4]
        generate_inputs(a, lb, tasks=['vmc_opt','vmc_dmc'])


