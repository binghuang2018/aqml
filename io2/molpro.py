
import os, sys
#import subprocess
#from subprocess import Popen as run

#import chemcoord as cc

import aqml.cheminfo.rw.xyz as crx
import tempfile as tpf
import numpy as np
import re

T,F = True,False

defaults = {}
try:
    defaults['molpro_exe'] = os.environ['MOLPRO_EXE'] #'/apps/molpro/molpro2018/molprop_2018_0_linux_x86_64_i8/bin/molpro'
except:
    raise Exception('#ERRROR: no env var MOLPRO_EXE was set')

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


def calculate(atoms, hamiltonian, basis, molpro_exe=None,
              charge=defaults['charge'],
              forces=defaults['forces'],
              title=defaults['title'],
              multiplicity=defaults['multiplicity'],
              wfn_symmetry=defaults['wfn_symmetry'],
              num_procs=None, mem_per_proc=None,
              task=defaults['task']):
    """Calculate the energy of a molecule using Molpro.

    Args:
        atoms (tuple):
            = (atomtypes, coords)
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

    #atoms = crx.read_xyz_simple(obj)
    symbs, coords = atoms
    na = len(symbs)
    s = '%d\n\n'%na
    for i in range(na): #.join( file(obj).readlines() )
        xi,yi,zi = coords[i]
        s += '%2s %12.8f %12.8f %12.8f\n'%(symbs[i],xi,yi,zi)

    fn = tpf.NamedTemporaryFile(dir='/tmp').name # obj[:-4]
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

    assert 'SCRATCH' in os.environ.keys()
    if not os.path.isfile(opf):
        cmd = '%s -t %d %s'%(molpro_exe, num_procs, ipf)
        print(cmd)
        iok = os.system(cmd)
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
        e0 = cmdout1("grep -E '^\s*!(RHF|UHF|RKS)\s\s*STATE\s\s*1.1\s\s*Energy' %s | tail -1 | awk '{print $NF}'"%opf)
    else:
        cmd = "grep -E '^\s*!%s total energy' %s | tail -1 | awk '{print $NF}'"%(h,opf); #print cmd
        es = cmdout1(cmd); #print es
        e0 = es[-1]
    print(' Get %s total energy: %s'%(h, e0))
    e = float(e0)
    output['energy'] = e0

    # get gradient (default unit: Hartree/Bohr)
    sgrad = cmdout("grep -A%d '%s GRADIENT FOR STATE 1.1' %s | tail -%d"%(na+3, h, opf, na))
    #print sgrad
    grad = np.array([ si.strip().split()[1:] for si in sgrad ], np.float) # /b2a
    output['gradient'] = grad

    # get geometry
    #atoms =
    output['symbols'] = np.array(atoms[0])
    output['coords'] = np.array(atoms[1])

    return e, grad #output


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


import io2.molpro_reader as imr
from berny import Berny, geomlib, optimize
import tempfile
import subprocess
import shutil
import aqml.cheminfo.molecule.core as cmc
import aqml.cheminfo.core as cc
import networkx as nx

def copy_class(objfrom, objto, names):
    for n in names:
        if hasattr(objfrom, n):
            v = getattr(objfrom, n)
            setattr(objto, n, v);

class Molpro(object):

    defaults = {}
    try:
        # '/apps/molpro/molpro2018/molprop_2018_0_linux_x86_64_i8/bin/molpro'
        defaults['molpro_exe'] = os.environ['MOLPRO_EXE']
    except:
        raise '#ERRROR: no env var MOLPRO_EXE was set'

    # use pyberny
    gconvs_berny = {
        'g09': {'gradientmax': 0.45e-3,
                'gradientrms': 0.3e-3,
                'stepmax': 1.8e-3,
                'steprms': 1.2e-3},
        'g09_tight': {  'gradientmax': 0.015e-3,
                        'gradientrms': 0.010e-3,
                        'stepmax': 0.06e-3,
                        'steprms': 0.04e-3},
        'g09_verytight': {  'gradientmax': 0.002e-3,
                            'gradientrms': 0.001e-3,
                            'stepmax': 0.006e-3,
                            'steprms': 0.004e-3},
        'molpro': { 'gradientmax': 0.3e-3,
                    'gradientrms': 0.2e-3,
                    'stepmax': 0.3e-3,
                    'steprms': 0.2e-3},
        } # note that in molpro, there is no specification for grad_rms and step_rms
    # use the own optimizers of molpro program
    gconvs_molpro = {'molpro':'', 'g09':',gaussian', \
        'g09_tight':',gradient=1.5e-5,step=6.0e-5,gaussian,srms=4e-5,grms=1e-5'}

    ccbsts = ['vdz','vtz','vqz','v5z', 'avdz','avtz','avqz','av5z']
    ccbsts_psp = [ si+'-pp' for si in ccbsts ]
    def2bsts = ['dzvp','tzvp','qzvp', 'adzvp','atzvp','aqzvp', ]
    def2bsts_psp = ['dzvpp','tzvpp','qzvpp', 'adzvpp','atzvpp','aqzvpp']
    bsts = ccbsts + ccbsts_psp + def2bsts + def2bsts_psp
    bsts_psp = ccbsts_psp + def2bsts_psp

    _dic_bst_heav = dict(zip(ccbsts,ccbsts_psp))
    _dic_bst_heav.update( dict(zip(def2bsts,def2bsts_psp)) )

    _dic_dfb = {
        'vdz':'vtz', 'vtz':'vqz', 'vqz':'avqz', 'avdz':'avtz', \
        'avtz':'avqz', 'avqz':'avqz', \
        'dzvp':'tzvp', 'tzvp':'qzvp', 'qzvp':'aqzvp', 'adzvp':'atzvp', \
        'atzvp':'aqzvp', 'aqzvp':'aqzvp', \
        'vdz-pp':'tzvpp','vtz-pp':'qzvpp','vqz-pp':'aqzvpp','avdz-pp':'atzvpp',\
        'avtz-pp':'aqzvpp','avqz-pp':'aqzvpp',\
        'dzvpp':'tzvpp', 'tzvpp':'qzvpp', 'qzvpp':'aqzvpp', 'adzvpp':'atzvpp', \
        'atzvpp':'aqzvpp', 'aqzvpp':'aqzvpp'
        }

    def get_sdef2(self, b):
        sdef2 = ''
        if b in self.def2bsts+self.def2bsts_psp:
          sdef2 = 'def2-'
        return sdef2

    def __init__(self, obj, param, wd='./'):

        self.wd = wd
        param0 = {'label':None, 'label_aux':'', 'df':T, 'df-hf':T, 'task': 'energy', \
                  'task0': 'energy', 'memory': '200', 'disp': F, \
                   'basis': 'vtz', 'method': 'mp2', 'ri':'jkfit', 'maxit':32, \
                   'diis':[F,], 'calcfc':[F], 'nproc':1, 'gconv':'molpro',\
                   'berny': F, 'lwave':F, 'qjob':F }
        for key in param0.keys():
            if key not in param.keys():
                param[key] = param0[key]
            else:
                if param[key] != param0[key]:
                    print( ' %s set to %s [default: %s]'%(key, param[key], param0[key]) )
        param['task0'] = param['task'] # 'task0' as a backup of task
        self.param = param

        if not param['df']: raise Exception(' ** only df=T is supported so far')

        if isinstance(obj,str):
            assert '/' not in obj, "#ERROR: input file should not contain '/' "
            assert os.path.exists(wd+obj), '#ERROR: input file does not exist!'
            fn = obj[:-4]
        else:
            fn = param['label']
            assert fn != None
        fo=fn+'.out'; fl=fn+'.log'
        self.fn = fn
        self.fo = fo
        self.fl = fl

        icalc = T
        energy, gradients = None, None
        if os.path.exists(wd+fo):
            om = imr.Molpro(wd+fo, keys=['e'], units=['ha','b'])
            energy, gradients = om.energy, om.gradients
            itn, igc, icalc = om.itn, om.igc, om.icalc
            if not itn:
                print( ' ** %s: job terminated abnormally'%fo )
            else:
                if not igc:
                    print( ' ** %s: optg not converged/is force calc'%fo )
        else:
            assert os.path.exists(fn+'.xyz')
            #print(' ** generate molpro inputs from xyz file')
            iok = True
        self.icalc = icalc
        self.energy = energy
        self.gradients = gradients

        smem = '%s,MW'%param['memory']
        meth = param['method']
        _s = ';disp,1' if param['disp'] else ''
        idft = F; icc2 = F; imp2 = F
        sb_g = ''
        f12 = F
        if meth in ['pbe','bp86','tpss','b3lyp',]:
            idft = T
            smeth = '{df-ks,%s,df_basis=jkfit%s}'%(meth,_s)
        elif meth in ['mp2', 'ccsd(t)']:
            imp2 = T
            assert param['df-hf']
            smeth = '{df-hf,df_basis=jkfit}\n{df-%s,df_basis=mp2fit}'%meth
        elif meth[-3:].lower() == 'f12':
            f12 = T
            assert param['df-hf']
            smeth = '{df-hf,df_basis=jkfit}\n'
            if meth in ['mp2f12', 'mp2-f12']:
                imp2 = T
                smeth += '{df-mp2-f12,ansatz=3*C(FIX,HY1),cabs=0,cabs_singles=0}'
            elif meth in ['cc2f12', 'ccsd(t)-f12']:
                icc2 = T
                smeth += 'df-ccsd(t)-f12'
            elif meth in ['lcc2f12', 'pno-lccsd(t)-f12']:
                icc2 = T
                smeth += '{df-mp2-f12,cabs_singles=-1}\n{pno-lccsd(t)-f12,domopt=tight}'
            else:
                raise Exception('unknow method')
            # for f12 method
            sb_g = "\nexplicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit"
        else:
            raise Exception('#ERROR: method for optg not supported')

        _bst = param['basis']
        ipsp = T if _bst in self.bsts_psp else F
        _bst_heav = _bst if ipsp else self._dic_bst_heav[_bst]

        bst = self.get_sdef2(_bst) + _bst
        bst_heav = self.get_sdef2(_bst_heav) + _bst_heav

        _dfb = self._dic_dfb[_bst]
        dfb = self.get_sdef2(_dfb) + _dfb
        _dfb_heav = self._dic_dfb[_bst_heav]
        dfb_heav = self.get_sdef2(_dfb_heav) + _dfb_heav

        els_heav = ['I', 'Te', 'Sn'] # atoms for which psp are used
        sb = "default=%s\n"%bst
        if not ipsp:
          for el in els_heav:
            sb += "%s=%s\n"%(el,bst_heav)

        jkfit = "\nset,jkfit,context=jkfit\ndefault=%s\n"%dfb
        if not ipsp:
          for el in els_heav:
            jkfit += '%s=%s\n'%(el,dfb_heav)

        mp2fit = "\nset,mp2fit,context=mp2fit\ndefault=%s\n"%dfb
        if not ipsp:
          for el in els_heav:
            mp2fit += '%s=%s\n'%(el,dfb_heav)

        assert param['ri'] in ['jkfit','optri']
        ri = "\nset,ri,context=%s\ndefault=%s\n"%(param['ri'],dfb)
        if not ipsp:
          for el in els_heav:
            ri += '%s=%s\n'%(el,dfb_heav)

        if idft:
            sb += jkfit
        elif imp2 or icc2:
            _sb = jkfit + mp2fit
            if f12: _sb += ri
            sb += _sb
        else:
            raise Exception('#ERROR: method not one of {dft, mp2, cc2}??')

        self.smeth = smeth
        self.smem = smem
        self.sb = sb
        self.sb_g = sb_g # global setting of df basis

    #@staticmethod
    def archive(self, icalc):
        wd = self.wd
        _dir = 'trash/' if icalc else 'dn/'
        fd = self.wd + _dir
        if not os.path.isdir(fd):
            os.mkdir(fd)
        sf = ''
        for fmt in ['com','out','xml','log']: sf += '%s.%s '%(wd+self.fn,fmt)
        cmd = 'mv %s %s/ 2>/dev/null'%(sf,fd)
        #print('cmd= ', cmd )
        os.system(cmd)

    def get_energy(self, gmol): #s1='',s2='',fn='./template'):
        """ full molpro calculation """

        sl = ''
        sl += self.s1
        geom = ''
        na = 0

        sb = ''
        sr = ''
        c = self.strain
        coords = gmol.coords.copy()
        if self.scan is not None:
            assert len(self.scan) == 2
            ia, ja = self.scan
            g = gmol.gnx.copy()
            g.remove_edge(ia,ja)
            gs = list(nx.connected_component_subgraphs(g))
            if len(gs) != 2:
                raise Exception('#ERROR: currently we can deal with 2 subgs!!')
            g1, g2 = gs
            ats1, ats2 = list(g1.nodes), list(g2.nodes)
            na1, na2 = len(ats1), len(ats2)
            v12 = gmol.coords[ja] - gmol.coords[ia]
            nv12 = v12/np.linalg.norm(v12)
            # fix g1, translate g2
            tv = nv12[np.newaxis, ...]
            coords[ats2] += tv * (1.00-c)
            sb = '_b_%d_%d'%(ia,ja)
            if c != 1.0:
                sr = '_s' + '%.2f'%c
        else:
            if c != 1.0:
                coords *= c
                sr = '_s' + '%.2f'%c

        for i, el in enumerate(gmol.symbols):
            x, y, z = coords[i]
            geom += '%-2s %12.6f %12.6f %12.6f\n'%(el,x,y,z)
            na += 1
        geom = '%d\n\n'%na + geom
        sg = "\ngeomtype=xyz\ngeometry = {\n%s}\n"%geom

        sl += "%s\n\n"%sg
        sl += self.s2
        #faux = self.param['label_aux']
        #fn2 = wd + fn # self.wd+self.fn+faux
        ipf = self.fn + sb + sr + '.com'
        opf = self.fn + sb + sr + '.out'

        #print(' * idx = ', idx )
        #if idx == 0 and (self.gradients is not None):
        #    energy, gradients = self.energy, self.gradients
        #else:
        with open(ipf, 'w') as fid: fid.write(sl)
        if self.param['qjob']:
            return
        assert 'SCRATCH' in os.environ.keys(), '#ERROR: env SCRATCH not set'
        cmd = '%s -t %d %s'%(self.defaults['molpro_exe'], self.param['nproc'], ipf)
        #print(cmd)
        if self.icalc:
            iok = os.system(cmd)
            assert not iok, '#ERROR: Molpro calculation failed'
        mr = imr.Molpro(opf, keys=['e'], units=['kcal','a'])
        mr.write('xyz',T)
        self.mr = mr
        copy_class(mr, self, ['props', 'zs', 'coords', 'symbols', 'energy', 'gradients'])


    def get_energy_and_gradient(self): #s1='',s2='',fn='./template'):
        """ molpro force calculation

        Note:
        It turns out to be significantly less efficient than
        the berny algorithim built in G09 ...
        Use with caution!!!! """
        try:
            atoms, lattice = yield
            while True:
                sl = ''
                sl += self.s1
                geom = ''
                na = 0
                for el, coord in atoms:
                    x, y, z= coord
                    geom += '%-2s %12.6f %12.6f %12.6f\n'%(el,x,y,z)
                    na += 1
                geom = '%d\n\n'%na + geom
                sg = "\ngeomtype=xyz\ngeometry = {\n%s}\n"%geom

                sl += "%s\n\n"%sg
                sl += self.s2
                #faux = self.param['label_aux']
                #fn2 = wd + fn # self.wd+self.fn+faux
                ipf = self.fn +'.com'
                opf = self.fn +'.out'

                #print(' * idx = ', idx )
                #if idx == 0 and (self.gradients is not None):
                #    energy, gradients = self.energy, self.gradients
                #else:
                with open(ipf, 'w') as fid: fid.write(sl)
                assert 'SCRATCH' in os.environ.keys(), '#ERROR: env SCRATCH not set'
                cmd = '%s -t %d %s'%(self.defaults['molpro_exe'], self.param['nproc'], ipf)
                #print(cmd)
                iok = os.system(cmd)
                assert not iok, '#ERROR: Molpro calculation failed'
                om = imr.Molpro(opf, keys=['e'], units=['ha','b'])
                energy, gradients = om.energy, om.gradients
                atoms, lattice = yield energy, gradients
        finally:
            print(' ++ optg terminated with success!')


    def initialize(self, strain=1.0, scan=None):
        """
        for single point energy or force calculation

        vars
        =============
        strain: scale atomic positions by a factor
        scan: must be a 2-entry list or tuple, specifying the bond to be scanned

        """
        optg_berny = F
        if (self.param['task0'] in ['opt','optg']) and self.param['berny']:
            optg_berny = T
            self.param['task'] = 'forces'
        self.optg_berny = optg_berny

        st = ''
        if (strain != 1.0) or (scan is not None):
            self.param['task'] = 'e'
            self.param['task0'] = 'e'
            self.icalc = T
        self.strain = strain
        self.scan = scan

        if self.param['task'] in ['opt','optg']:
            st = '{optg'
            maxit = self.param['maxit']
            if maxit: st += ',maxit=%s'%maxit # if maxit=None, then add nothing
            st += self.gconvs_molpro[self.param['gconv']]
            diis = self.param['diis']
            if diis[0]:
                st += ';method,diis,%s,step'%diis[1]
            cf = self.param['calcfc']
            if cf[0]:
                assert not diis[0]
                # calc hessian every `n steps
                # if n=0, then calc hessian at each step
                st += ';hessian,numerical=%d'%cf[1]
            st += '}'
        elif self.param['task'] in ['force','forces']:
            st = '{forces}'
        elif self.param['task'] in ['e','energy']:
            st = ''
        else:
            raise Exception('task not supported')
        self.st = st

        s1 = "memory,%s,M\n\n"%(self.smem)
        swfn = '' if self.param['lwave'] else '!'
        s1 += "!file,1,%s.int\n%sfile,2,%s.wfu\n\n"%(self.fn,swfn,self.fn)
        s2 = "basis={%s\n}%s\n\n%s\n%s\n---\n"%(self.sb, self.sb_g, self.smeth, st)
        self.s1 = s1
        self.s2 = s2

    def run(self):
        if self.optg_berny:
            # now optg
            gconv = self.gconvs_berny[ self.param['gconv'] ]
            gradientmax, gradientrms, stepmax, steprms = [ gconv[k] \
                  for k in ['gradientmax', 'gradientrms', 'stepmax', 'steprms'] ]
            optimizer = Berny(geomlib.readfile(self.fn+'.xyz'), \
                              gradientmax = gradientmax, \
                              gradientrms = gradientrms, \
                              steprms=steprms, stepmax=stepmax, \
                              maxsteps=self.param['maxit'])
            solver = self.get_energy_and_gradient()
            next(solver)
            coords_a = []; es_a = []; grads_a = []
            idx = 0
            for geom in optimizer:
                atoms = list(geom)
                if not isinstance(atoms[0], tuple):
                    atoms = atoms[-1]
                print(' * idx = ', idx )
                #if idx == 0 and (self.gradients is not None):
                #    energy, gradients = self.energy, self.gradients
                #else:
                energy, gradients = solver.send( (atoms, None) )
                coords_a.append( geom.coords )
                es_a.append( energy )
                grads_a.append( gradients )
                idx += 1
                optimizer.send((energy, gradients))
            self.coords_a = coords_a
            self.grads_a = grads_a
            self.es_a = es_a
            self.coords = coords_a[-1]
            self.gradients = grads_a[-1]
            self.energy = es_a[-1]
        else:
            mol = cc.molecules(self.fn+'.xyz')
            gmol = cmc.RawMol(mol)
            self.get_energy(gmol)


