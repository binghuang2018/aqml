#!/usr/bin/env python

import re, io2, os, sys
import numpy as np
import aqml.cheminfo as co
from aqml.cheminfo.core import *
import aqml.cheminfo.rw.xyz as crx
import shutil

T, F = True, False
spp = '\s\s*'

uc = io2.Units()

cardinal = {'vdz':2, 'vtz':3, 'vqz':4}

cmdout1 = lambda cmd: os.popen(cmd).read().strip()
cmdout = lambda cmd: os.popen(cmd).read().strip().split('\n')

iu = io2.Units()


class Molpro(object):

    jobs = ['optg','force','freq']
    jobs_a = ['optg','force','forces','freq','frequency']


    def __init__(self, f, keys=[], iprop=T, units=['kcal','a']):
        self.f = f
        self.units = units # kcal/mol and Angstrom
        # note that later `f may change (when more properties from
        # other file are added to the same mol), but `fn won't, it
        # always corresponds to the filename of the first `f entered.
        self.fn = f[:-4]
        self.fl = self.fn + '.log'
        self.fo = self.fn + '.out'
        self.fmt = f[-3:]
        self.e_const = {'kcal':iu.h2kc, 'kj':iu.h2kj, 'ha':1.0, 'h':1.0}[units[0]]
        self.d_const = {'a':1.0, 'b':iu.b2a}[units[1]]

        self.props = {}
        ks0 = ['e','forces']
        for k in ks0:
            if k not in keys:
                keys.append(k)
        self.read_molecule()
        self.get_version()
        if iprop:
            for k in keys:
                if k in ['forces','grads','gradients']:
                    self.get_grads()
                else:
                    self.get_properties([k])

    def check_status(self):
        fo = self.f
        llast = open(fo).readlines()[-1]
        itn = ('Molpro calculation term' in llast) # if terminated normally?
        igc = F # geometry converged?
        ic = F # if scf/geometry (has) converged
        if itn:
            ic = ('warning' not in llast)
            if ic:
                if self.is_jobtype('optg') or (not self.is_jobtype()): # the latter tells if it's SP calc
                    igc = T
        self.itn = itn
        self.igc = igc

    #@property
    #def version(self):
    #    if not hasattr(self, '_version'):
    #        self._calc = 'Molpro ' + self.get_version()
    #    return self._calc

    def get_version(self):
        cmd = "grep '^\s\s*Version ' %s"%self.fo
        ver = cmdout1(cmd).split()[1]
        self.props.update( {'name':'Molpro', 'version': ver} )

    def is_jobtype(self, *args):
        ioks = []
        if len(args) == 0:
            args = self.jobs
        for arg in args:
            assert arg in self.jobs_a
            iok = not os.system("grep -E '^\s*[^!]\s*\{?%s' %s >/dev/null 2>/dev/null"%(arg,self.f))
            ioks.append(iok)
        return np.any(ioks)

    def read_molecule(self):
        """ read geometry info, basis and hamiltonian """
        fo = self.f[:-4] + '.out'
        self.fo = fo
        icom = F
        ilog = F # read geom from log with success??
        iout = F
        iout_0 = F # use input geom at the beginning of out file
        ioptg = F
        _cs = open(self.f).readlines()
        icalc = F
        itn, igc = F, F
        self.itn = itn
        self.igc = igc
        ratio = 1.0
        if self.fmt in ['com','inp']: #read molecule from Molpro input file
            icom = T
            for i,ci in enumerate(_cs):
                if ci.strip()[:8] in ['geometry']:
                    break
            na = int(_cs[i+1])
            cs = _cs[i+3:i+3+na]
        elif self.fmt in ['out']:
            self.check_status()
            itn, igc = self.itn, self.igc
            if not np.all([itn,igc]): icalc = T
            ioptg = not os.system("grep ' PROGRAM \* OPT' %s >/dev/null"%fo)
            self.ioptg = ioptg
            if ioptg:
                if self.itn: #igc:
                    iout = T
                    #cmd = "sed -n '/Current geom/,/Geometry wr/p' %s"%fo
                    #cs = cmdout(cmd)[4:-4]

                    # The two lines above may fail sometimes, the code below are more robust
                    cmd = "grep -n ' Current geometry' %s | head -n 1 | cut -d: -f1"%fo
                    ln = int(cmdout1(cmd))
                    fid = open(fo)
                    for il in range(ln+1): next(fid)
                    na = int(next(fid))
                    next(fid)
                    cs = []
                    for _ in range(na): cs.append( next(fid) )

                else:
                    # retrieve last config from log file
                    fl = self.f[:-4] + '.log'
                    print(' *** read geom from log file %s'%fl)
                    assert os.path.exists(fl)
                    cmd = "grep -n ' Current geometry' %s | tail -n 1 | cut -d: -f1"%fl
                    try:
                        ln = int(cmdout1(cmd))
                        fid = open(fl)
                        for il in range(ln+1): next(fid)
                        na = int(next(fid))
                        next(fid)
                        cs = []
                        for _ in range(na): cs.append( next(fid) )
                        ilog = T
                    except:
                        print( '  ** no optg cycle found in log file! use geom from input' )
                        cmd = "sed -n '/ ATOMIC COORDINATES/,/ Bond lengths in Bohr/p' %s | grep      '^\s*[0-9]' | awk '{print $2,$4,$5,$6}'"%fo
                        #iout_0 = T # this simply means that input geom is to be used
                        cs = cmdout(cmd)
            else: # single point energy/force calc, coordinates in Bohr
                ratio = io2.Units().b2a
                cmd0 = "grep 'Molecule type: Atom' %s"%fo
                if cmdout1(cmd0):
                    ln = int(cmdout1("grep -n ' ATOMIC COORDINATES' %s | sed 's/:/ /g' | awk '{print $1}'"%fo)) + 4
                    cs = cmdout("sed -n '%dp' %s | awk '{print $2,$4,$5,$6}'"%(ln,fo))
                else:
                    cmd = "sed -n '/ ATOMIC COORDINATES/,/ Bond lengths in Bohr/p' %s | grep '^\s*[0-9]' | awk '{print $2,$4,$5,$6}'"%fo
                    #print('cmd=',cmd)
                    cs = cmdout(cmd) #[4:-2]
                    #print('cs=',cs)
            na = len(cs)
            #print('cmd=\n', cmd)
        else:
            raise Exception('#ERROR: file format not supported')
        self.icalc = icalc # need for further calcualtion??
        self.itn, self.igc = itn, igc

        # job type
        task = None
        if ioptg: # may be assigned T when fmt='out'
            task = 'optg'
        else:
            for key in [ 'force', 'freq']:
                if self.is_jobtype(key):
                    task = key
                    break
        if not task: task = 'energy'
        self.task = task

        self.ioptg = ioptg
        symbols = []; zs = []; coords = []
        #print('cs=',cs)
        for ci in cs:
            csi = ci.strip().split(); #print('csi=',csi)
            si = csi[0]
            try:
                zi = chemical_symbols.index(si)
            except:
                zi = chemical_symbols_lowercase.index( si.lower() )
            coords_i = np.array(csi[1:4],dtype=float) * ratio
            symbols.append(si); zs.append( zi )
            coords.append(coords_i)
        #print('zs=',zs)

        m = atoms(zs,coords)
        zs = np.array(zs,dtype=int)
        self.zs = zs
        nheav = (zs>1).sum()
        self.symbols = symbols
        self.nheav = nheav
        self.coords = np.array(coords)
        self.na = len(zs)
        self.m = m
        self.props.update( dict(zip(['m','na','nheav','zs','symbols','symbs','coords'], \
                                 [m,na,nheav,zs,symbols,symbols,self.coords])) )

        #
        # now method, i.e., hamitonian
        # first, get contents of input
        if self.fmt in ['out']:
            #ie = int(cmdout1("awk '/Commands\s\s*initialized/{print NR}' %s"%self.f)) # not work under macos
            ie = int(cmdout1("grep -nE 'Commands\s\s*initialized' %s | cut -d: -f1"%self.f))
            cs0 = _cs[:ie]
        elif self.fmt in ['com','inp']:
            cs0 = _cs
        else:
            raise Exception('#ERROR: format not supported')
        nlmax = len(cs0)
        _meths = ['df-hf','df-ks','hf','ks', \
                'mp2-f12','df-mp2-f12','pno-lmp2-f12','mp2','df-mp2', \
                'ccsd-f12', 'df-ccsd-f12', 'pno-lccsd-f12', 'ccsd', 'df-ccsd', \
                'ccsd(t)-f12', 'df-ccsd(t)-f12', 'pno-lccsd(t)-f12', 'ccsd(t)', 'df-ccsd(t)']
        _eprops = ['hf','ks',]*2 + \
                  ['mp2f12']*2 + ['lmp2f12'] + ['mp2']*2 + \
                  ['cc2f12']*2 + ['lcc2f12'] + ['cc2']*2 + \
                  ['cc2tf12']*2 + ['lcc2tf12'] + ['cc2t']*2
        _meths_patts = ['df-hf','df-ks','^-hf','^-ks', \
                'mp2-f12','df-mp2-f12','pno-lmp2-f12','mp2','df-mp2', \
                'ccsd-f12', 'df-ccsd-f12', 'pno-lccsd-f12', 'ccsd', 'df-ccsd', \
                'ccsd\(t\)-f12', 'df-ccsd\(t\)-f12', 'pno-lccsd\(t\)-f12', 'ccsd\(t\)', 'df-ccsd\(t\)']
        spp = '\s\s*'
        p1 = spp.join(['![UR](HF|KS)','STATE','1.1','Energy'])
        p2 = spp.join(['!MP2-F12', 'total', 'energy'])
        p3 = spp.join(['!MP2', 'total', 'energy'])
        p4 = spp.join(['!PNO-LMP2-F12\(PNO\)', 'total', 'energy'])

        # From Molpro manual,
        #     """ Thus, we currently recommend CCSD-F12A for AVDZ and AVTZ basis sets,
        #         and CCSD-F12B for larger basis sets (rarely needed).  """
        aux = 'a'
        p5 = spp.join(['!PNO-LCCSD-F12%s'%aux, 'total', 'energy'])
        p6 = spp.join(['!LCCSD\(T\)-F12%s'%aux, 'total', 'energy'])
        p7 = spp.join(['CCSD-F12%s'%aux, 'total', 'energy'])
        p8 = spp.join(['CCSD\(T\)-F12%s'%aux, 'total', 'energy'])
        p9 = spp.join(['CCSD', 'total', 'energy'])
        p10 = spp.join(['CCSD\(T\)', 'total', 'energy'])
        _epatts = [p1]*4 + \
                   [p2]*2 + [p4] + [p3]*2 + \
                   [p7]*2 + [p5] + [p9]*2 + \
                   [p8]*2 + [p6] + [p10]*2
        _levels = [0.35, 0.45, 0.5, 0.6, \
                  1.65, 1.45, 1.25, 1.15, 1.05, \
                  2.65, 2.45, 2.25, 2.15, 2.05, \
                  3.65, 3.45, 3.25, 3.15, 3.05 ]
        meths = []
        levels = []
        eprops = [] # energy properties
        epatts = [] # patterns to match different energies
        icnt = 0
        #itl = 0
        #print('cs0=',cs0)
        idft = F
        idf = F # density-fitting
        while T:
            #itl += 1
            #if itl == 20: break
            #print('icnt,nlmax=',icnt,nlmax)
            if icnt == nlmax: break
            ci = cs0[icnt].strip().lower()
            #print('ci=',cs0[icnt])
            if ci=='' or ci[0] == '!':
                icnt += 1
                continue
            else:
                for imeth,meth in enumerate(_meths):
                    mp = _meths_patts[imeth]
                    if meth not in meths:
                        patts = ['^%s$'%mp, '^%s[},\s!]'%mp, '[{\s]%s[},\s]'%mp ]
                        tfs = []
                        for p in patts:
                            tfi = F
                            if re.search(p,ci, flags=re.MULTILINE): tfi = T
                            tfs.append(tfi)
                        if np.any(tfs):
                            #print('++ meth, ci=', meth,ci)

                            if 'ks' in meth: # now get xc function
                                idft = T
                                pt1 = '([^{]*)ks,\s*([a-zA-Z][a-zA-Z0-9]*)[,}\s!]'
                                pt2 = '([^{]*)ks,\s*([a-zA-Z][a-zA-Z0-9]*)$'
                                ot1 = re.search(pt1,ci)
                                ot2 = re.search(pt2,ci, flags=re.MULTILINE)
                                if ot1:
                                    ot = ot1
                                elif ot2:
                                    ot = ot2
                                else:
                                    raise Exception('#ERROR: no match found for %s or %s!'%(pt1,pt2))
                                ots = ot.groups()
                                meth = ''.join(ots)
                                eprop = meth #ots[-1]
                            else:
                                eprop = _eprops[imeth]

                            if meth == 'df-mp2-f12' and re.search('cabs_singles\s*=\s*-1', ci):
                                continue
                            imp2 = T
                            if meth in ['ccsd-f12', 'df-ccsd-f12', 'ccsd(t)-f12','df-ccsd(t)-f12']:
                                meth2 = 'mp2-f12'
                            elif meth in ['ccsd', 'df-ccsd', 'ccsd(t)', 'df-ccsd(t)']:
                                meth2 = 'mp2'
                            elif meth in ['pno-lccsd-f12', 'pno-lccsd(t)-f12']:
                                meth2 = 'pno-lmp2-f12'
                            else:
                                imp2 = F
                            if imp2:
                                meths.append(meth2)
                                i2 =  _meths.index(meth2)
                                eprop2 = _eprops[i2]
                                epatt2 = _epatts[i2]
                                level2 = _levels[i2]
                                eprops.append(eprop2)
                                epatts.append(epatt2)
                                levels.append(level2)
                            meths.append(meth)
                            eprops.append(eprop)
                            epatts.append(_epatts[imeth])
                            levels.append(_levels[imeth])
            icnt += 1

        #meth_h = meths[-1]
        self.meths = meths
        self.eprops = eprops
        self.epatts = epatts
        self.meth = meths[-1] ##
        #self.h = eprops[-1] # meth_h
        self.props.update( dict(zip(['meths','meth'],[self.meths,self.meth])) )

        # now basis set
        idxl = 0
        while T:
            ci = _cs[idxl].strip()
            if ci[:5] == 'basis':
                break
            idxl += 1
        ci2 = _cs[idxl+1].strip()
        tf1, tf2 = ('{' not in ci), ('{' not in ci) # tf: true or false?
        if tf1: # and tf2:
            # i.e., simple basis set input, e.g., basis=vtz
            basis = ci.split('!')[0].split('=')[-1].strip().lower()
            basis_c = basis
        else:
            # i.e., detailed basis setting, e.g., basis={default=vtz; I=vtz-pp; ...}
            csb = '' # '{'+ci.split('!')[0].split('{')[1] #''
            if 'default' not in ci:
                idxl += 1
                basis = ci2.split('!')[0].strip().split('=')[1].lower()
            else:
                c1, c2 = ci.split('!')[0].split('{')
                basis = c2.split('=')[1].lower()
                #csb += c2
            ### search for '}'
            while T:
                cj = _cs[idxl].strip()
                if '}' in cj:
                    cj2 = cj.split('}')[0]
                    if cj2 != '':
                        csb = csb+';'+cj2 if csb != '' else cj2
                    break
                else:
                    if cj != '':
                        cj2 = cj.split('!')[0]
                        if cj2 != '':
                            csb = csb + ';' + cj2 if csb != '' else cj2
                idxl += 1
            basis_c = '{'+csb.split('basis={')[1].lower()+'}'
            #print('basis=',basis)
            #print('basis_c=',basis_c)
        basis = re.sub('-','',basis)
        self.basis = basis
        self.basis_c = basis_c
        self.props.update( dict(zip(['basis','basis_c'], [basis,basis_c])) )

    def get_geomopt(self):
        """ get geometry optimizer """
        cmd = "grep ' Geometry optimization using default procedure for command' %s | awk '{print $NF}'"%self.fo
        _gopt = cmdout1(cmd)
        gopt = None
        if (_gopt in ['DF-KS-SCF']): # and self.idft:
            gopt = self.meth + '/basis=' + self.basis_c
        else:
            raise Exception('#ERROR: optimizer not supported')
        return gopt

    def get_grads(self):
        na = self.na
        fn = self.f[:-4]
        grads = None
        ff = None # fianl file
        for fmt in ['out','log']:
            ft = fn + '.' + fmt
            if os.path.exists(ft):
                if not os.system("grep 'GRADIENT FOR' %s >/dev/null"%ft):
                    ff = ft
                    break
        if not ff:
            print(' ** no gradient found')
            self.gradients = grads
            return
        cmd0 = "grep ' GRADIENT FOR STATE ' %s >/dev/null"%ff
        iok = os.system(cmd0)

        ## the `cmd` below is not safe due to the fact that sometimes,
        ## there is line break within the force blocks in log file
        #cmd = "grep -A%d ' GRADIENT FOR STATE ' %s | tail -%d"%(na+3, ff, na)
        ## As a result, we use an alternative scheme, i.e., read line 1 first,
        ## then exhaust all following lines till the number of force vectors
        ## is correct
        cmd = "grep -n ' GRADIENT FOR STATE ' %s | tail -1 | sed 's/:/ /g' | awk '{print $1}'"%ff
        #print('cmd= ', cmd )
        if iok == 0:
            n1 = int(cmdout1(cmd)) + 3
            fid = open(ff,'r')
            for _ in range(n1):
                next(fid)
            iac = 0
            sgrads = []
            while T:
                if iac == na:
                    break
                li = next(fid)
                si = li.strip()
                if si == '':
                    continue
                else:
                    sgrads.append( si.split()[1:] )
                    iac += 1
            assert si.split()[0] == '%d'%na
            #print('grads) ', sgrads)
            grads = np.array(sgrads, dtype=float)
            #np.array([ si.strip().split()[1:] for si in sgrad ], np.float) # /b2a
        else:
            print(' ** reading gradients failed!!')
        self.gradients = grads # (default unit: Hartree/Bohr)

        #if self.itn and self.igc and self.ioptg:
        #print('itn,igc,ioptg=', self.itn, self.igc, self.ioptg)
        gopt = self.get_geomopt()
        self.props.update( {'gopt': gopt} )
        self.props.update( {'forces': grads} )


    def get_properties(self, keys):
        props = {}
        props_l = {} # the same as props, but without basis in key
        if (keys is None) or len(keys)==0 or (keys[0].lower() == 'e'):
            keys = self.eprops
        #print('keys=',keys)
        #print('eprops=',self.eprops)
        for _key in keys:
            is_optg_e = F
            key = _key.lower()
            _props = {}
            energetic_props = []
            if key in self.eprops:
                const = self.e_const
                patt = self.epatts[ self.eprops.index(key) ]
                if self.ioptg:
                    if ''.join(key.split('-')) in ['b3lyp', 'dfb3lyp', 'mp2', 'dfmp2', 'mp2f12', 'dfmp2f12']:
                        is_optg_e = T
                        cmd = "grep -B2 ' END OF GEOMETRY OPTIMIZATION' %s | head -n 1"%self.fo
                        cs = cmdout1(cmd).split()
                        #print('--cs=',cs)
                        v = eval(cs[2]) * const
                    else:
                        raise Exception('geom optimizer not supported!')
                else: #if not is_optg_e:
                    cmd = "grep -E '%s' %s | tail -n 1"%(patt,self.fo)
                    #print(cmd)
                    cs = cmdout1(cmd).split()
                    #print('cs=',cs)
                    v = eval(cs[-1]) * const
                key2 = key + self.basis
                energetic_props.append(key2)
                #props_l[key] = v
                _props[key2] = v
            elif key in ['dipole', ]:
                assert os.path.exists(self.fl)
                cmd = "grep ' Dipole moment \/Debye' %s | tail -1 | awk '{print $4,$5,$6}'"%self.fl
                dip = np.asarray(cmdout1(cmd).split(), dtype=float)
                _props[key] = dip
            elif key in ['homo','lumo','gap']: #'HOMO','LUMO','GAP']:
                assert os.path.exists(self.fl)
                cmd = "grep '^ [HL][OU]MO' %s | tail -3 | awk '{print $NF}' | sed 's/eV//g'"%self.fl
                mos = np.asarray( cmdout(cmd), dtype=float ) / io2.Units().h2e
                _props.update( dict(zip( ['homo','lumo','gap'], mos )) )
            else:
                raise Exception('#ERROR: method %s not supported'%key)
            props.update( _props )
        self.props.update( props )
        self.energetic_props = energetic_props
        #self.energy = props_l[ self.h ]

    def write(self, fmt, wp=T):
        if fmt == 'xyz':
            ci = ''
            #if cs is not None: ci = cs[i].strip()
            if len(self.props) > 0 and wp: # write property
                keys = self.props.keys()
                ys = [ self.props[key] for key in keys ]
                csi = ['%.6f '%yi for yi in ys ] + ['# ']
                csi += ' '.join(keys)
                ci += ''.join(csi)
            else:
                print(' ** no property writen to xyz file')
            fo = self.fn+'.xyz'
            cwd = os.path.dirname(fo)
            if cwd == '': cwd = os.getcwd()
            newdir = cwd+'/old'
            if os.path.exists(fo):
                if not os.path.exists(newdir):
                    os.makedirs(newdir)
                os.system('mv %s %s'%(fo,newdir))
            crx.write_xyz(self.fn+'.xyz', (self.symbols,self.coords), comments=ci)
        else:
            raise Exception('#ERROR: file format %s not supported yet'%fmt)

    #def perceive_ctab(self):
    #    ictab = F
    #    self.ictab = ictab

    #def write_sdf(self, fsdf):
    #    if not self.ictab:
    #        self.perceive_ctab()
    #    write_ctab()


def write_all(fs, keys, units=['kcal','a']):
    if type(fs) is str:
        nf = 1; fs = [fs]
    elif type(fs) is list:
        nf = len(fs); f1 = fs[0]
    f1 = fs[0]
    #f1, f2 = fs # ['target/frag_01.out', 'target-vtz/frag_01.out',]
    o = Molpro(f1, units=units)
    #o.read_molecule()
    #o.get_properties(keys)

    if nf == 2:
        o.f = fs[1]
        o.read_molecule()
        o.get_properties(keys)

    # if fs[0]='target/frag_01.out', xyz file with property
    # will be writen to target/frag_01.xyz
    o.write('xyz')



class molprojob(object): #co.atoms):

    """
    This module is more robust for reading molpro output and is highly
    recommended (cf. class Molpro() above)
    """

    def __init__(self, f, property_names=['e']): #patts=None, cols=None, keys=None):
        self.f = f
        #self.patts = patts
        self.property_names = property_names
        #self.cols = cols
        self.cs = open(f).readlines()
        self.fmt = f[-3:]

    _jobs = ['optg','force','freq']

    @property
    def job(self):
        if not hasattr(self, '_job'):
            jo = 'e'
            for j in self._jobs:
                cmd = "grep -E '^\s*[^!]\s*\{?%s' %s"%(j, self.f)
                if cmdout1(cmd):
                    jo = j
                    break
            self._job = jo
        return self._job

    _todo = """@property
    def icbs(self):
        if not hasattr(self, '_icbs'):
            self._icbs = self.get_icbs()
        return self._icbs

    def get_icbs(self):
        key = 'Extrapolate'
        icbs = F
        if cmdout('grep "%s" %s'%(key, self.f)) != '':
            icbs = T
        return icbs """

    @property
    def atoms(self):
        if not hasattr(self, '_atoms'):
            self._atoms = self.get_atoms()
        return self._atoms

    @property
    def is_job_done(self):
        return 'Molpro calc' in self.cs[-1]


    def get_atoms(self):
        fo = self.f
        fl = fo[:-4] + '.log'
        const = 1.0
        cmd = "grep 'OPT (Geometry optimization' %s"%fo
        cmd2 = "grep -n ' Current geometry' %s | head -n 1 | cut -d: -f1"%fo
        cmd3 = "grep -n ' Current geometry' %s | tail -n 1 | cut -d: -f1"%fl
        zs = []; coords = []
        s = cmdout1(cmd)
        s2 = cmdout1(cmd2)

        ioc = T # use output coords
        iofmt = 'out' # use geom from *.out file
        if s:
            if self.is_job_done:
                ln = int(s2) + 1
            else:
                if os.path.exists(fl):
                    s3 = cmdout1(cmd3)
                    #print('s=',s, 's2=',s2, 's3=',s3)
                    if s3:
                        ln = int(s3) + 1
                        iofmt = 'log'
                        self.cs = open(fl).readlines()
                    else:
                        ioc = F
                else:
                    ioc = F
        else:
            ioc = F
        self.ioc = ioc
        self.iofmt = iofmt

        if ioc:
            print('   ** read fully/partially optimized geom from %s.%s'%(fo[:-4], iofmt))
            #print('ln=',ln, 'ci= "%s"'%self.cs[ln:ln+2])
            na = int(self.cs[ln])
            for li in self.cs[ln+2:ln+2+na]:
                tsi = li.strip().split()
                zi = co.chemical_symbols_lowercase.index( tsi[0].lower() )
                zs.append(zi); coords.append([eval(x) for x in tsi[1:] ])
        else: # single point energy/force calc, coordinates in Bohr
            print('   ** read input geom from %s.out'%fo[:-4])
            const = io2.Units().b2a
            ln = int(cmdout1("grep -n ' ATOMIC COORDINATES' %s | sed 's/:/ /g' | awk '{print $1}'"%fo)) + 3
            while True:
                ci = self.cs[ln].strip()
                if ci == '':
                    break
                tsi = ci.strip().split()
                #print('ci=',ci)
                zi = co.chemical_symbols_lowercase.index( tsi[1].lower() )
                zs.append(zi)
                coords.append([ eval(x) for x in tsi[3:6] ]) #cmdout("sed -n '%dp' %s | awk '{print $2,$4,$5,$6}'"%(ln,fo))
                ln += 1
                #print('cs=',cs)
        return co.atoms(zs, coords)
        #co.atoms.__init__(self, zs, coords)


    bsts =  ['aug-cc-pvdz', 'aug-cc-pvtz', 'aug-cc-pvqz', \
             'cc-pvdz', 'cc-pvtz', 'cc-pvqz', \
             'def2-sv(p)', 'def2-svp', 'def2-tzvp', 'def2-qzvp']
    bsts_short = ['avdz', 'avtz', 'avqz', 'vdz', 'vtz','vqz', 'def2sv-p', 'def2svp', 'def2tzvp', 'def2qzvp']
    dctb = dict(zip(bsts, bsts_short))


    @property
    def basis(self):
        """ return short name of basis """
        if not hasattr(self, '_basis'):
            self.get_basis()
        return self._basis

    @property
    def basis_c(self):
        """ return the complete name of basis """
        if not hasattr(self, '_basis_c'):
            self.get_basis()
        return self._basis_c

    def get_basis(self):
        # now basis set
        _cs = self.cs
        idxl = 0
        while T:
            ci = _cs[idxl].strip()
            if ci[:5] == 'basis':
                break
            idxl += 1
        ci2 = _cs[idxl+1].strip()
        tf1, tf2 = ('{' not in ci), ('{' not in ci) # tf: true or false?
        if tf1: # and tf2:
            # i.e., simple basis set input, e.g., basis=vtz
            basis = ci.split('!')[0].split('=')[-1].strip().lower()
            basis_c = basis # complete specification of basis
        else:
            # i.e., detailed basis setting, e.g., basis={default=vtz; I=vtz-pp; ...}
            csb = '' # '{'+ci.split('!')[0].split('{')[1] #''
            if 'default' not in ci:
                idxl += 1
                basis = ci2.split('!')[0].strip().split('=')[1].lower()
            else:
                c1, c2 = ci.split('!')[0].split('{')
                basis = c2.split('=')[1].lower()
                #csb += c2
            ### search for '}'
            while T:
                cj = _cs[idxl].strip()
                if '}' in cj:
                    cj2 = cj.split('}')[0]
                    if cj2 != '':
                        csb = csb+';'+cj2 if csb != '' else cj2
                    break
                else:
                    if cj != '':
                        cj2 = cj.split('!')[0]
                        if cj2 != '':
                            csb = csb + ';' + cj2 if csb != '' else cj2
                idxl += 1
            basis_c = basis
            if re.search('basis={', csb):
                basis_c = '{'+csb.split('basis={')[1].lower()+'}'
            #print('basis=',basis)
            #print('basis_c=',basis_c)
        basis = re.sub('-','',basis)
        self._basis = basis
        self._basis_c = basis_c

    @property
    def meth(self):
        if not hasattr(self, '_meth'):
            self._meth = self.get_meth()
        return self._meth

    @property
    def xc(self):
        if not hasattr(self, '_xc'):
            fun = None
            patt = '\-?[ru]?ks,\s*([a-zA-Z][a-zA-Z0-9]*)'
            cmd = "grep -E '%s' %s"%(patt, self.f)
            #print('cmd= "%s"'%cmd)
            c = cmdout1(cmd)
            if c:
                fun = re.search(patt, c).groups()[0]
            self._xc = fun
        return self._xc

    _meths_c = ['pno-lccsd(t)-f12', 'pno-lccsd-f12', \
                'df-ccsd(t)-f12', 'df-ccsd-f12', \
              'ccsd(t)-f12', 'ccsd-f12', 'df-ccsd(t)', \
               'df-ccsd', 'ccsd(t)', 'ccsd', \
              'pno-lmp2-f12', 'df-mp2-f12', 'mp2-f12', \
              'df-mp2', 'mp2', 'df-hf', 'hf', 'b3lyp' ]
    _meths =   [ 'lcc2f12', 'lccf12', \
                'cc2f12', 'ccf12', \
              'cc2f12', 'ccf12', \
              'cc2', 'cc', \
              'cc2', 'cc', \
              'lmp2f12', 'mp2f12', \
              'mp2f12', 'mp2', \
              'mp2', 'hf', 'hf', 'b3lyp' ]
    _patts =   ['pno-lccsd\(t\)-f12', 'pno-lccsd-f12', \
                'df-[ru]?ccsd\(t\)-f12', 'df-[ru]?ccsd-f12', \
              '[ru]?ccsd\(t\)-f12', '[ru]?ccsd-f12', \
              'df-[ru]?ccsd\(t\)', 'df-[ru]?ccsd', \
              '[ru]?ccsd\(t\)', '[ru]?ccsd', \
              'pno-lmp2-f12', 'df-[ru]?mp2-f12',\
               '[ru]?mp2-f12', 'df-[ru]?mp2', \
               '[ru]?mp2', 'df-[ru]?hf', '[ru]?hf', 'b3lyp' ]

    def get_meth(self):
        """ get method (full name, rather than short name.
            I.e., one entry of `_meths_c`) used in *.out file """
        # assume input file is of format *.out
        assert self.fmt == 'out'
        cs0 = self.cs
        nlmax = len(cs0)

        # step 1, check if it's dft method
        idft = F
        meth = None
        if self.xc:
            idft = T
            meth = self.xc
        else:
            for i, patt in enumerate(self._patts):
                cmd = "grep -E '%s' %s"%(patt, self.f)
                if cmdout1(cmd):
                    meth = self._meths_c[i]
                    break
        assert meth, '#ERROR: meth=None?'
        return meth

    @property
    def epatts(self):
        if not hasattr(self, '_epts'):
            if self.xc:
                epts = { self.xc: { 'keys': [ [ '![UR]KS','STATE','1.1','Energy'] ],
                                    'meths': [ self.xc + self.basis ],
                                    'ffmt': 'log'} }
            else:
              if self.job == 'optg':
                epts = { \
                  'df-ccsd(t)-f12': { 'keys': [ [ '!CCSD\(T\)-F12', 'total', 'energy'],
                                                [ '!MP2-F12', 'total', 'energy' ], \
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                      'meths': [ mi  + self.basis for mi in [ 'dfcc2f12', 'dfmp2f12', 'hf'] ],
                                      'ffmt': 'log' },
                     'df-ccsd-f12': { 'keys': [ [ '!CCSD-F12', 'total', 'energy'],
                                                [ '!MP2-F12', 'total', 'energy' ], \
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                      'meths': [ mi  + self.basis for mi in [ 'dfcc2f12', 'dfmp2f12', 'hf'] ],
                                      'ffmt': 'log' },

                        }

              else:
                epts = { 'ccsd(t)': { 'keys': [ [ '!CCSD\(T\)', 'total', 'energy' ], \
                                                [ 'MP2', 'total', 'energy' ], \
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                       'meths': [ mi + self.basis for mi in ['cc2','mp2','hf'] ],
                                       'ffmt': 'out' },
                    # From Molpro manual,
                    #     """ Thus, we currently recommend CCSD-F12A for AVDZ and AVTZ basis sets,
                    #         and CCSD-F12B for larger basis sets (rarely needed).  """
                     'ccsd(t)-f12': { 'keys': [ [ '!CCSD\(T\)-F12a', 'total', 'energy'],
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                      'meths': [ mi  + self.basis for mi in [ 'cc2f12', 'hf'] ],
                                       'ffmt': 'out'  },
                   # Note that for df-ccsd(t)-f12, a different strategy
                   # was used (cf. ccsd(t)-f12) and one unique mp2f12
                   # energy is available (cf. multiple energies for the case
                   # without 'df-' )
                  'df-ccsd(t)-f12': { 'keys': [ [ '!CCSD\(T\)-F12', 'total', 'energy'],
                                                [ '!MP2-F12', 'total', 'energy' ], \
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                      'meths': [ mi  + self.basis for mi in [ 'cc2f12', 'mp2f12', 'hf'] ],
                                       'ffmt': 'out'  },
                  'pno-ccsd(t)-f12': { 'keys': [ [ '!LCCSD\(T\)-F12', 'total', 'energy'],
                                                [ '!LMP2-F12', 'total', 'energy' ], \
                                                [ '![UR]HF', 'STATE', '1.1', 'Energy' ] ],
                                      'meths': [ mi  + self.basis for mi in [ 'lcc2f12', 'lmp2f12', 'hf'] ],
                                       'ffmt': 'out'  },

                        }
            self._epts = epts
        return self._epts


    def get_energies(self):
        """ get energies of the last configuration?? """
        _meth_c = self.meth
        i0 = self._meths_c.index(_meth_c)
        _meth = self._meths[i0]

        #print('meth=',_meth_c, 'meths=',self.epatts)
        if _meth_c not in self.epatts:
            raise Exception('#ERROR: key not found')

        epts = self.epatts[_meth_c]
        spp = '\s\s*'
        patts = [ spp.join(ptsi) for ptsi in epts['keys'] ]
        meths = epts['meths']
        iptf = self.f[:-3] + epts['ffmt']

        es = {}
        for i, pt in enumerate(patts):
            cmd = "grep -E '%s' %s | tail -n1"%(pt, iptf)
            #print('cmd= "%s"'%cmd)
            c = cmdout1(cmd)
            #if _meth_c == 'df-mp2-f12':
            #   if re.search('cabs_singles\s*=\s*-1', c):
            #       continue
            v = eval( c.split()[-1] )
            es[meths[i]] = v
        return es


    @property
    def props(self):
        if not hasattr(self, '_props'):
            self._props = self.get_props()
        return self._props


    @property
    def apatts(self):
        """ patterns for all properties other than energy """
        if not hasattr(self, '_apt'):
            a = { 'mu': [] }
            self._apt = a
        return self._apt

    def get_props(self):
        props = {}
        for i,p in enumerate(self.property_names): #patts):
            if p == 'e':
                _props = self.get_energies()
                props.update( _props )
            else:
                if p not in self.apatts:
                    raise Exception('Todo: key currently absent!')
                patt, loc = self.apatts[p]
                cmdi = "grep '%s' %s | awk '{print %s}'"%(patt, self.f, loc)
                #print('cmd=',cmdi)
                val = eval( cmdout1(cmdi) )
                props[key] = val
        return props

    def write(self, fo, overwrite=T):
        props = self.props
        atoms = self.atoms
        atoms.props.update( props )
        if not os.path.exists(fo):
            atoms.write(fo)
        else:
            if overwrite:
                atoms.write(fo)
            else:
                try:
                  for p in cmdout1("sed -n '2p' %s"%fo).split():
                    k, v = p.split('=')
                    atoms.props.update( {k:eval(v)} )
                except:
                  print('   ** no pre-existing prop found in xyz file')
                atoms.write(fo)


if __name__ == "__main__":

    import ase, sys
    import argparse as ap

    ps = ap.ArgumentParser()
    ps.add_argument('-p', '--patts', dest='patts', nargs='*', type=str, help='patterns to be matched')
    ps.add_argument('-k', '--keys',  dest='keys', nargs='*', type=str, help='property to be written to xyz file')
    ps.add_argument('-c' , '--cols', nargs='*', type=int, help='position of matched value. default to the last entry')
    ps.add_argument('-ow', action='store_true', help='overwrite xyz file?')
    ps.add_argument('-i', '--ipts', dest='fs', nargs='*', type=str, help='Input files to be processed')

    ag = ps.parse_args( sys.argv[1:] )
    ow = ag.ow #{'T':T, 'F':F}[ag.ow]

    if ag.patts is None:
        ag.patts = []
        ag.cols = []
        ag.keys = []
    else:
        if ag.cols is None:
            ag.cols = [-1]
        print('patts=', [ "%s"%pt for pt in ag.patts ])

    for f in ag.fs:
        print(' now ', f)
        obj = molprojob(f, ag.patts, ag.cols, ag.keys)
        obj.write(f[:-4]+'.xyz', ow)

