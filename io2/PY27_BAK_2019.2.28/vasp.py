#!/usr/bin/env python

import numpy as np
import ase.io as aio
import ase.data as ad
import os,sys
import ase

global ves
ves = {'H':1, 'C':4, 'N':5, 'O':6, 'P':5, 'S':6, 'F':7, 'Cl':7, \
       'Fe':14, 'Pd':12, 'Pt':12, 'Si':4, 'B':3, 'Au':11}

class vasp(object):

    def __init__(self, obj, label=None, iauto=False, fixZs=[], job='OPT', \
                 iwrite=False, vdw=False, npar=4, ismear=0, sigma=0.4, encut=340, \
                 lcharg=False, lwave=False, sediffg='-0.02', \
                 ibrion='2', ispin='1', vacuum=8.0, nproc=1, version='std'):
        self.obj = obj 
        self.fixZs = fixZs
        self.job = job
        self.vdw = vdw 
        self.npar = npar 
        self.ismear = ismear
        self.sigma = sigma
        self.encut = encut
        self.lcharg = lcharg
        self.lwave = lwave
        self.sediffg = sediffg
        self.ibrion = ibrion
        self.ispin = ispin

        self.nproc = nproc
        self.version = version # ['gam', 'std', ]

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
        self.m = atoms
        self.label = label 

        if iauto: 
            self.auto(vacuum=vacuum)
        else:
            ps = atoms.positions
            uc = np.floor( [ np.max(ps[:,i]) - np.min(ps[:,i]) + vacuum for i in range(3) ] )
            mU.cell = uc
            mU.center()
            self.m = mU

        if iwrite: self.write()


    def auto(self, vacuum=8.0):
        """
        reorient molecule so that the longest a/b/c lies through z axis
        """
        mU = self.m.copy()

        psU = mU.positions
        uc = np.floor( [ np.max(psU[:,i]) - np.min(psU[:,i]) + 14 for i in range(3) ] ) + 1.0

        mU.cell = uc; ucU = mU.cell
        mU.center()

        # re-orientate molecule so that the longest a/b/c lies through z axis
        diy = True # rotate manually
        if diy:
            mU.rotate('y', np.pi/2, center = 'COU')
        else:
            axises = ['x','y','z']
            axis = axises[ np.argsort( uc )[-1] ]
            if axis != 'z':
                axis_r = 'x'
                if axis == 'x':
                    axis_r = 'y'
                mU.rotate(axis_r, np.pi/2, center = 'COU')

        psU = mU.positions
        uc = np.floor( [ np.max(psU[:,i]) - np.min(psU[:,i]) + vacuum for i in range(3) ] )
        #uc2 = int( uc[2]//4 + 1 )*4 # make uc[2] could be divided by 4
        #uc[2] = uc2
        mU.cell = uc

        mU.center()
        self.m = mU


    def write(self):

        mU = self.m
        label = self.label

        ucU = mU.cell; psU = mU.positions; 
        naU = len(mU); seqs = np.arange(naU); seqsU = []

        slreal = '.FALSE.'
        if any( np.linalg.norm(ucU, axis=1) > 24 ):
            slreal = '.TRUE.'

        Suc = '\n'.join( [ ' '.join( [ '%12.6f'%uij for uij in ui ] ) for ui in ucU ] )

    # now sort the sequence of atoms by Z and group all atoms with the same Z into one bag
        zs = mU.numbers;
        zsu = np.unique(zs) # it's already sorted
        nzu = len(zsu)

        symbs = []
        for iz0 in zsu:
            si0 = ad.chemical_symbols[iz0]
            if si0 not in symbs:
                symbs.append(si0)

        Ssymbs = ' '.join(symbs)
        Snas = ''; Sps = ''; ne = 0
        for i in range(nzu):
            zi = zsu[i]
            filter_i = (zs == zi)
            nai = np.sum(filter_i); ne += nai*ves[ad.chemical_symbols[zsu[i]] ]
            Snas += '%d '%nai
            seqsU = seqs[filter_i]
            oflags = 'T T T'
            if zi in self.fixZs:
                oflags = 'F F F'
            for j in seqsU:
                Sps += ' '.join( [ '%12.6f'%pij for pij in psU[j] ] ) + ' ' + oflags + '\n'

        s = '%s\n1.00\n'%label[:-1]
        s += Suc
        s += '\n%s\n%s\nSelective dynamics\nCartesian\n'%(Ssymbs,Snas)
        s += Sps
        iok = os.system("echo '%s' >%sPOSCAR"%(s,label))

        job = self.job.lower()
        iscf = True 
        if job in ['cls_2','band',]:
            iscf = False 
        self.iscf = iscf 

        slcharg = '.TRUE.' if self.lcharg or job[:3] == 'cls' else '.FALSE.'
        slwave = '.TRUE.' if self.lwave or job[:3] == 'cls' else '.FALSE.'

        # now INCAR file
        nband = int(((ne/2.0)*1.1//24 + 1)*24)
        svdw = ''
        if self.vdw:
            svdw = 'LVDW= .TRUE.; IVDW=11\n'

        if job in ['static', 'single-point', 'cls','cls_1']:
            s = """SYSTEM = %s
ISTART = 0; ICHARG = 2
ISMEAR = %s; SIGMA = %s
ENCUT = %s; GGA = PE; NBANDS = %d
NPAR = %s
ALGO = Fast
PREC = Normal
LREAL= %s
AMIN = 0.01
ISPIN = %s
%sLCHARG = %s
LWAVE = %s"""%(label[:-1], self.ismear, self.sigma, self.encut, \
                nband, self.npar, self.slreal, self.ispin, svdw, \
                slcharg, slwave)
        elif job in ['cls_2',]: # step 2 for calculating Core level shift (CSL)
            # non scf calc
            s = """SYSTEM = %s
ISTART = 1; ICHARG = 11
ISMEAR = %s; SIGMA = %s
ENCUT = %s; GGA = PE; NBANDS = %d
NPAR = %s
ALGO = Fast
PREC = Normal
LREAL= %s
AMIN = 0.01
ISPIN = %s
%sLCHARG = %s; LWAVE = %s
ICORELEVEL = 1"""%(label[:-1], self.ismear, self.sigma, self.encut, \
                nband, self.npar, self.slreal, self.ispin, self.svdw, \
                slcharg, slwave)
        elif job in ['opt', ]:
            s = """SYSTEM = %s
ISTART = 0; ICHARG = 2
ISMEAR = %s; SIGMA = %s
ENCUT = %s; GGA = PE; NBANDS = %d
NPAR = %s
ALGO = Fast
PREC = Normal
LREAL= %s
AMIN = 0.01
ISPIN = %s
%sEDIFFG = %s
LCHARG = %s; LWAVE = %s
NSW = 200
IBRION = %s"""%(label[:-1], self.ismear, self.sigma, self.encut, \
                nband, self.npar, self.slreal, self.ispin, self.svdw, \
                self.sediffg, slcharg, slwave, self.ibrion)
        elif job in ['md', ]:
            s = """SYSTEM = %s
ISTART = 0; ICHARG = 2

ENCUT = 240
LREAL = A                      ! real space projection
PREC  = Normal                 ! chose Low only after tests
EDIFF = 1E-5                   ! do not use default (too large drift)
ISMEAR = 0; SIGMA = 0.100      ! Gaussian smearing
ALGO = Very Fast               ! recommended for MD (fall back ALGO = Fast)
MAXMIX = 40                    ! reuse mixer from one MD step to next
NPAR = 2                       ! one orbital on 4 cores
ISYM = 0                       ! no symmetry
NELMIN = 4                     ! minimum 4 steps per time step, avoid breaking after 2 steps

! MD (do little writing to save disc space)
IBRION = 0 ; NSW = 200 ;  NWRITE = 0 ; LCHARG = .FALSE. ; LWAVE = .FALSE.
TEBEG =   2000 ; TEEND =  2000
! canonic (Nose) MD with XDATCAR updated every 50 steps
POTIM = 1.0; SMASS = -3        ! Nose Hoover thermostat"""%label
        else:
            raise '#ERROR: `job type not supported'
        iok = os.system("echo '%s' >%sINCAR"%(s,label))

        # now POTCAR file

        print 'symbols = %s'%Ssymbs
        iok = os.system("genpot -xc PBE -prefix %s %s"%(label,Ssymbs))

        # now KPOINTS file
        s = """Monkhorst-Pack
0
Monkhorst-Pack
 1  1  1
 0  0  0"""
        iok = os.system("echo '%s' >%sKPOINTS"%(s,label))
        return

    def run(self):
        """
        run vasp 
        """
        if not self.iscf:
            chgcar = self.label+'.CHGCAR'
            assert os.path.exists(chgcar)
            iok = os.system('cp %s.CHGCAR CHGCAR'%self.label 

        for suffix in ['INCAR','POSCAR','POTCAR','KPOINTS', \
                       'CHGCAR']:
            iok1 = os.system('cp %s.%s %s'%(self.label, suffix, suffix))
            assert iok1 
        iok2 = os.system('mpirun -np %s vasp.%s >out'%(self.nproc, self.version))
        assert iok2
        for suffix in ['INCAR','POSCAR','POTCAR','KPOINTS', \
                       'out','OUTCAR','CHG','CHGCAR','CONTCAR','DOSCAR', \
                       'PROCAR','EIGENVAL','IBZKPT','OSZICAR','PCDAT',\
                       'vasprun.xml','WAVECAR','XDATCAR']:
            iok3 = os.system('mv %s %s.%s'%(suffix, self.label, suffix))
            assert iok3


if __name__ == '__main__':

    import stropr as so

    args = sys.argv[1:]; idx = 0

    if '-h' in args:
        print 'tovasp [-job OPT] [-vdw] [-ismear 0] [-sigma 0.4] [-lcharg F] [-lwave F]'
        print '       [-encut 340] [-vac 10.0] [-ibrion 2] [-npar 4] [-fixZs 15_16]'
        print '       [file1.xyz] [file2.xyz] ...'
        sys.exit(2)

    dic = {'T':True, 'True':True, 'F': False, 'False':False}

    keys = ['-vdw',]; vdw,idx = so.haskey(args,keys,idx)
    keys = ['-ismear',]; hask,ismear,idx = so.parser(args,keys,'0',idx)
    keys = ['-sigma',]; hask,sigma,idx = so.parser(args,keys,'0.1',idx)
    keys = ['-encut',]; hask,encut,idx = so.parser(args,keys,'340',idx)
    keys = ['-vac','-vacuum',]; hask,svacuum,idx = so.parser(args,keys,'8.0',idx); vacuum = float(svacuum)

    keys = ['-lcharg',]; hask,slcharg,idx = so.parser(args,keys,'F',idx); lcharg = dic[slcharg]
    keys = ['-lwave',]; hask,slwave,idx = so.parser(args,keys,'F',idx); lwave = dic[slwave]

    keys = ['-ediffg',]; hask,sediffg,idx = so.parser(args,keys,'-0.02',idx)
    keys = ['-ibrion',]; hask,ibrion,idx = so.parser(args,keys,'2',idx)
    keys = ['-ispin',]; hask,ispin,idx = so.parser(args,keys,'1',idx)

    keys = ['-irun',]; hask,sirun,idx = so.parser(args,keys,'False',idx)
    irun = dic[sirun]

    keys = ['-job',]; hask,job,idx = so.parser(args,keys,'OPT',idx)
    assert hask, '#ERROR: -job [task] has to be specified'

    keys = ['-npar',]; hask,npar,idx = so.parser(args,keys,'4',idx)
    assert hask, '#ERROR: -npar [INT] has to be specified'

    keys = ['-nproc',]; hask,snproc,idx = so.parser(args,keys,'1',idx)
    assert hask, '#ERROR: -nproc [INT] has to be specified'

    keys = ['-version',]; hask,version,idx = so.parser(args,keys,'std',idx)

    keys = ['-fixZs',]; hask,fixZs_,idx = so.parser(args,keys,'',idx)
    fixZs = []
    if len(fixZs_) > 0: fixZs = [ int(si) for si in fixZs_.split('_') ]

    fs = args[idx:]
    for f in fs:
        s = vasp(f, fixZs=fixZs, vacuum=vacuum, label=os.path.basename(f[:-3] ), \
                job=job, vdw=vdw, npar=npar, ismear=ismear, sigma=sigma, \
                encut=encut, lcharg=lcharg, lwave=lwave, sediffg=sediffg, \
                ibrion=ibrion, ispin=ispin, nproc=nproc, version=version )
        s.write()
        if s.iscf:
            print ' -- info: please use standard submit script'
            if irun:
                print ' -- info: u r enforced to run job one by one'
                s.run()
        else:
            print ' -- Now we calculate molecules one by one'
            s.run()

