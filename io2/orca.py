#!/usr/bin/env python

import re, os, sys, io2
import aqml.cheminfo as co
import aqml.cheminfo.core as cc
import numpy as np

T, F = True, False

uc = io2.Units()

cardinal = {'vdz':2, 'vtz':3, 'vqz':4}

bsts =  ['aug-cc-pvdz', 'aug-cc-pvtz', 'aug-cc-pvqz', \
         'cc-pvdz', 'cc-pvtz', 'cc-pvqz', \
         'def2-sv(p)', 'def2-svp', 'def2-tzvp', 'def2-qzvp']
bsts_short = ['avdz', 'avtz', 'avqz', \
              'vdz', 'vtz','vqz', \
              'def2sv-p', 'def2svp', 'def2tzvp', 'def2qzvp']
dctbo = dict(zip(bsts, bsts_short))
dctbi = dict(zip(bsts_short, bsts))

hs_short = [ 'mp2', 'lmp2', 'lcc', 'lcc2', 'cc', 'cc2' ] # Hamitonian
hs       = [ 'RI-MP2', 'DLPNO-MP2', 'DLPNO-CCSD', 'CCSD', 'CCSD(T)' ]
dcthi = dict(zip(hs_short, hs))
dctho = dict(zip(hs, hs_short))

cmdout = lambda cmd: os.popen(cmd).read().strip()
cmdout1 = lambda cmd: os.popen(cmd).read().strip().split('\n')

sgeom = lambda m: '\n'.join(['{:2s} {:12.6f} {:12.6f} {:12.6f}'.format(_,x,y,z) for _,(x,y,z) in zip(m.symbols,m.coords) ])
sgeom_cp = lambda m: '\n'.join(['{:2s} : {:12.6f} {:12.6f} {:12.6f}'.format(_,x,y,z) for _,(x,y,z) in zip(m.symbols,m.coords) ])


class orca(cc.molecule):

    def __int__(self, f, param):

        cc.molecule.__init__(self, f)

        label = f[:-4]
        self.lb = label
        assert f[-3:] in ['sdf', 'xyz', 'mol', 'pdb']

        # task: energy, opt, copt, gdiis-copt, ...
        param = {'nproc':1, 'mem': 1000, 'method':'b3lyp', 'basis':'Def2-TZVP', 'tightscf':T, \
                 'charge':None, 'mult':None, 'df':T, 'task':'opt', 'disp':F, 'wc':F, 'maxit':60, \
                 'hess':0, 'gc':'default'}
        self.param = param

    def write_input(self):
        """ write orca input file """
        param = self.param
        nproc = param['nproc']
        mem = param['mem'] # memory per core
        bst = param['basis']
        task = param['task']
        _charge = param['charge']
        charge = _charge if _charge else 0
        _mult = param['mult']
        sc = "%pal nprocs %d end\n%maxcore %d\n"%(nproc, mem)

        meth = param['meth']
        if meth in ['hf-3c', 'pbeh-3c', 'hf3c', 'pbeh3c']:
            if meth[-3] != '-': meth = meth[:-2]+'-3c'
            sm = '! %s nopop\n'%meth
            sb = ''
        elif meth in ['pbe','tpss','bp86','b3lyp','wb97x']:
            sm = '! %s TIGHTSCF nopop'%meth #
            if grid: sm += ' Grid5 FinalGrid6'
            sd3 = ' D3BJ\n' if disp else '\n'
            sm += sd3
            xtra = ''
            if meth in ['b3lyp','wb97x']:
                xtra = ' RIJCOSX' # for hybrid df
            if bst in ['sv(p)', 'svp', 'tzvp',]:
                sb = '! def2-%s def2/J%s\n'%(bst,xtra)
            elif bst in ['vdz','vtz',]:
                sb = '! cc-p%s def2/J%s\n'%(bst,xtra)
            else:
                raise Exception('Todo')
        elif meth in hs_short:
            if 'cbs' in bst:
                # ['ano-cbs-ep2','ano-cbs-ep3','cc-cbs-ep2','cc-cbs-ep3', 'cbs2','cbs3']:

                df = 'RI RIJCOSX ' if idf else 'Conv '
                mb =  {'lccsd(t)': 'DLPNO-CCSD(T)', 'mp2':'MP2', 'ccsd(t)':'CCSD(T)'}[meth]
                #dctbb = {'ano-cbs':'ano', 'cc-cbs':'cc', 'cbs':'cc'}
                ssb = bst.split('-')
                nc = len(ssb)

                if bst == 'cbs3':
                    sm = '! RHF ExtrapolateEP3(CC) TightSCF Conv'
                elif bst == 'cbs2':
                    bb = 'cc'
                    sm = '! %sRHF ExtrapolateEP2(2/3,%s,%s) TightSCF'%(df,bb,mb)
                elif bst == 'cbs':
                    sm = '! %s%s Extrapolate(2/3) TightSCF'%(df,mb)
                else:
                    raise Exception('Todo')
                sm += ' nopop\n'
                sb = ''
            else:
                sm = '! ' + dcthi[meth] + ' TightSCF\n'
                if bst in ['tzvp',]:
                    sb = '! def2-TZVP def2/J def2-TZVP/C RIJCOSX\n'
                elif bst in ['vdz', 'avdz', 'vtz','avtz']:
                    bn = {'vdz':'cc-pVDZ', 'avdz':'aug-cc-pVDZ', 'vtz':'cc-pVTZ', 'avtz':'aug-cc-pVTZ'}[bst]
                    sb = '! %s %s/C def2/J RIJCOSX\n'%(bn,bn) #
                    #sb = '! %s %s/C def2/J\n'%(bn,bn) #
                else:
                    raise Exception('Todo')
        else:
            raise Exception('Todo')

        dctj = {'e':'', 'optg':'! Opt\n', 'tightopt':'! TightOpt\n', 'force': '! ENGRAD\n'}
        assert task in dctj, '#ERROR: task not supported!'
        st = dctj[task]
        if task in ['optg',]:
            st += '\n%%geom\nmaxiter %s\n'%param['maxit']
            nhess = param['hess']
            if nhess:
                st += 'calc_hess true\nrecalc_hess %s\n'%nhess # calc Hess after `nhess ionic cycles
            if param['gc'] in ['tight']:
                st += 'TolE 1e-6\nTolRMSG 2e-4\nTolMaxG 3e-4\nTolRMSD 2e-4\nTolMaxD 3e-4\n'
            elif param['gc'] in ['loose']:
                st += 'TolE 1e-4\nTolRMSG 3e-4\nTolMaxG 4.5e-4\nTolRMSD 2e-2\nTolMaxD 3e-2\n'
            st += 'end'
            # NormalOpt (default)  TolE=5e-6, TolRMSG=1e-4, TolMaxG=3e-4, TolRMSD=2e-3, TolMaxD=4e-3
            # TIGHTOPT             TolE=1e-6, TolRMSG=3e-5, TolMaxG=1e-4, TolRMSD=6e-4, TolMaxD=1e-3
            # GAU (G09 default)                       3e-4        4.5e-4        1.2e-3        1.8e-3
            # GAU_LOOSE                             1.7e-3        2.5e-3        6.7e-3        1.0e-2

        if _mult:
            mult = _mult
        else:
            if self.na == 1:
                mult = {1:2, 3:2, 4:1, 5:2, 6:3, 7:4, 8:3, 9:2, \
                        11:2, 12:0, 13:2, 14:3, 15:4, 16:3, 17:2,\
                        33:4, 34:3, 35:2, 53:2}[self.zs[0]]
            else:
                mult = np.mod(np.sum(self.zs),2)+1

        if icp: # calculate CP-corrected energy
            rawm = cmc.RawMol( self )
            mols = rawm.monomers
            n = len(mols)

            assert task in ['e',]
            if n == 1:
                so = sc + sm + sb + st + '\n'
                s = so + '* xyzfile 0 %d %s.xyz'%(mult,self.lb)
                with open(self.lb+'.com', 'w') as fid: fid.write(s)
            elif n == 2:
                _sm = sm.strip() + ' PModel\n'
                so = sc + _sm + sb + st + '\n'
                _header = '$new_job\n'

                # monomers
                s1 = ''
                # monomer at dimer basis
                scp = '' # '%id "monomer_2"\n'
                mult = np.mod(np.sum(mols[0].zs),2)+1
                s1_2 = so + scp + '*xyz 0 %d\n'%mult + sgeom(mols[0]) + '\n' + sgeom_cp(mols[1]) + '\n*\n\n\n'
                mult = np.mod(np.sum(mols[1].zs),2)+1
                s2_2 = _header + so + scp + '*xyz 0 %d\n'%mult + sgeom_cp(mols[0]) + '\n' + sgeom(mols[1]) + '\n*\n\n\n'

                # dimer
                s = s1_2 + s2_2
                with open(self.lb+'_cp.com','w') as fid: fid.write(s)
            else:
                raise Exception('Todo')
        else:
            so = sc + sm + sb + st + '\n\n'
            if self.param['wc']: # write coord to orca input file
                scoord = '*xyz %d %d\n'%(chg,mult)
                scoord += ''.join( ['{si} {ci[0]} {ci[1]} {ci[2]}\n'.format(si=mol.symbols[ia], ci=mol.coords[ia]) for ia in range(mol.na) ] )
                scoord += '*\n'
            else:
                scoord = '* xyzfile %d %d %s.xyz\n'%(chg,mult,self.lb)
            s = so + scoord + '\n'
            with open(self.lb+'.com', 'w') as fid: fid.write(s)

    def run(self):
        assert 'orca4' in os.environ, '#ERROR: please specify env var "orca4"!'
        cmd = '$orca4 {lb}.com >{lb}.out'.format(lb=self.lb)
        iok = os.system(cmd)
        if not iok:
            sys.exit('Job failed')
        orcajob.__init__(self, lb)


class orcajob(object):

    def __init__(self, lb):
        self.f = lb + '.out'
        self.label = lb

    @property
    def icbs(self):
        if not hasattr(self, '_icbs'):
            self._icbs = self.get_icbs()
        return self._icbs

    def get_icbs(self):
        key = 'Extrapolate'
        icbs = F
        if cmdout('grep "%s" %s'%(key, self.f)) != '':
            icbs = T
        return icbs

    @property
    def atoms(self):
        if not hasattr(self, '_atoms'):
            self._atoms = self.get_atoms()
        return self._atoms

    def get_atoms(self):
        cmd = "grep 'basis set group' %s | tail -1 | awk '{print $2}' | grep -o '[0-9]*'"%self.f
        na = int( cmdout(cmd) ) + 1 # atom idx in orca starts from 0
        cmd = "grep -n 'CARTESIAN COORDINATES (ANGSTROEM)' %s | tail -1 | sed 's/:/ /g' | awk '{print $1}'"%self.f
        ln1 = 2 + int(cmdout(cmd))
        ln2 = ln1+na-1
        cmd = 'sed -n "%s,%sp" %s'%(ln1,ln2,self.f)
        zs = []; coords = []
        for li in cmdout1(cmd):
            tsi = li.strip().split()[:4]
            zs.append( co.chemical_symbols.index(tsi[0]) )
            coords.append( [ eval(vi) for vi in tsi[1:4] ] )
        return co.atoms(zs,coords)

    @property
    def meth(self):
        if not hasattr(self, '_meth'):
            self._meth = self.get_meth()
        return self._meth

    def get_meth(self):
        _meth = None
        idft = F
        ifd = F
        for meth in ['b3lyp', 'wb97x', 'tpss', 'pbe0', 'pbe']:
            if cmdout('grep -i %s %s'%(meth,self.f)):
                _meth = meth
                idft = T
                ifd = T
                break

        #if not ifd:
        meths_i = ['dlpno-mp2', 'ri-mp2', 'mp2']
        meths_o = ['lmp2',      'mp2',    'mp2' ]
        dct = dict(zip(meths_i, meths_o))
        for meth in hs:
            ot = cmdout('grep -i "%s" %s'%(meth, self.f))
            if ot and ('!' in ot):
                _meth = dct[meth]
                ifd = T
                break

        #if nof ifd:
        meths_i = ['dlpno-ccsd', 'dlpno-ccsd(t)', 'ccsd', 'ccsd(t)']
        meths_o = ['lcc',      'lcc2',    'cc', 'cc2' ]
        dct = dict(zip(meths_i, meths_o))
        for meth in meths_i:
            ot = cmdout('grep -i "%s" %s'%(meth, self.f))
            if ot and ('!' in ot):
                _meth = dct[meth]
                ifd = T
                break

        assert _meth
        # dispersion interaction?
        if idft:
            ot = cmdout('grep D3BJ %s'%self.f)
            if ot and ('!' in ot):
                _meth += 'd3'
        return _meth

    @property
    def basis(self):
        if not hasattr(self, '_basis'):
            self._basis = self.get_basis()
        return self._basis

    def get_basis(self):
        _bst = None
        for b in self.bsts:
            if cmdout('grep -i %s %s'%(b,self.f)):
                _bst = dctbo[b]
                break
        assert _bst, 'Plz add more reference basis'
        return _bst

    @property
    def method(self):
        if not hasattr(self, '_method'):
            self._method = self.meth + self.basis
        return self._method

    @property
    def e(self):
        if not hasattr(self, '_e'):
            self._e = self.get_energy()
        return self._e

    def get_energy(self):
        if not self.icbs: # Extrapolate tp CBS
            cmd = "grep 'FINAL SINGLE POINT' %s | tail -n 1 | awk '{print $NF}'"%self.f
            #print(cmd)
            e = eval( cmdout(cmd) ) # in hartree
            es = {self.method: e, 'e':e}
        else:
            cmd = "grep 'SCF energy with basis' %s | awk '{print $5}'"%f
            bsts = [ bdct[si[:-1].lower()] for si in cmdout1(cmd) ]; #print(bsts)
            scfmeths = [ 'hf'+si for si in bsts ]
            n1, n2 = [ cardinal[bst] for bst in bsts ]
            scfcbsmeth = 'hfcbsv%d%dz'%(n1,n2)

            cmd = "grep 'SCF energy with basis' %s | awk '{print $NF}'"%f
            t = cmdout(cmd); #print('t=',t)
            es_hf = [ eval(ei) for ei in t.split('\n') ]; #print(es_hf)
            dct = dict(zip(scfmeths, es_hf))

            cmd = "grep '^MP2 energy with basis' %s | awk '{print $6}'"%f
            t = cmdout(cmd)
            # print('t=',t)
            imp2 = F
            if t:
                imp2 = T
                es_corr = t.split('\n'); #print(es_corr)
                smeths = [ 'mp2'+si for si in bsts ]
                dct.update( dict(zip(smeths, [eval(ei)+es_hf[i] for i,ei in enumerate(es_corr)])) )

            cbsmeth = None

            icc2 = F
            cmd = "grep '^MDCI energy with basis ' %s"%f
            t = cmdout(cmd)
            if t:
                icc2 = T
                es_cc2 = []
                ts = t.split('\n')
                for i,ti in enumerate(ts):
                    #print('ti=',ti)
                    es_cc2.append( es_hf[i]+eval(re.findall('\-?[0-9][0-9]*\.[0-9][0-9]*', ti)[0]) )
                smeths = ['cc2'+si for si in bsts ]
                dct.update( dict(zip(smeths, es_cc2)) )
                cbsmeth = 'cc2cbsv%d%dz'%(n1,n2)

            icmp2 = F # corr from mp2?
            cmd = "grep 'CCSD(T) correlation energy with basis' %s | awk '{print $NF}'"%f
            t = cmdout(cmd)
            if t:
                icmp2 = T
                e_cc2_corr = eval(t.strip())
                bst1 = bsts[0]
                dct.update( {'cc2%s'%bst1: es_hf[0]+e_cc2_corr} )
                cbsmeth = 'cc2cbsmp2v%d%dz'%(n1,n2)

            assert cbsmeth, '#ERROR: `cbsmeth is None?'

            cmd = "grep ' CBS [Sct]' %s"%f
            #cmd = "grep 'Estimated CBS total energy' %s | awk '{print $NF}'"%f
            t = cmdout(cmd); #print('t=',t)
            e1, e1c, e2 = [ eval(re.findall('\-?[0-9][0-9]*\.[0-9][0-9]*', ei)[0]) for ei in t.split('\n') ]
            escbs = [e1, e2]; #print(escbs)

            meths = [scfcbsmeth, cbsmeth] # 'cc2cbs34']
            dct.update( dict(zip(meths, escbs)) )

            sdic = "'%s':{"%self.label,
            for k in dct:
                sdic += "'%s':%.8f, "%(k,dct[k])
            sdic += '}'
            print( sdic, ',')

            es = dct
        return es

    @property
    def grad(self):
        if not hasattr(self, '_grad'):
            self._grad = self.get_grad()
        return self._grad

    def get_grad(self):
        """Read Forces from ORCA output file."""
        file = open('%s.engrad'%self.lb, 'r')
        lines = file.readlines()
        file.close()
        igrad = F
        for i, line in enumerate(lines):
            if line.find('# The current gradient') >= 0:
                gradients = []; igrad = T; continue
            if igrad and "#" not in line:
                grad=line.split()[-1]
                tempgrad.append(float(grad))
                if len(tempgrad)==3:
                    gradients.append(tempgrad)
                    tempgrad=[]
            if '# The at' in line:
                getgrad="no"
        return -np.array(gradients) * Hartree / Bohr

    def write(self, fo):
        e = self.e
        atoms = self.atoms
        atoms.props.update( es )
        atoms.write(fo)


if __name__ == "__main__":

    import ase, sys

    fs = sys.argv[1:]

    for f in fs:
        fmt = f[-3:]
        if fmt in ['xyz']:
            obj = orca(f, param)
            obj.write_input()
            obj.run()
        elif fmt in ['out']:
            obj = orcajob(f)
            obj.write(f[:-4]+'.xyz')
        else:
            raise Exception('file format not supported')


