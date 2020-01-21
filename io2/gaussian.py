#!/usr/bin/env python

import os,sys,re
from ase import Atom, Atoms
import io2
import io2.data as data
from io2.gaussian_reader import GaussianReader as GR0
import ase.data as ad
import numpy as np
import multiprocessing
import cheminfo as co

import fortranformat as ff

a2b = 0.52917721092
NAME = co.chemical_symbols


global dic_sm, atom_symbs, prop_names
dic_sm = {'H':2,'B':2,'C':3,'N':4,'O':3,'F':2, \
          'Si':3,'P':4,'S':3,'Cl':2,'Ge':3,'As':4,\
          'Se':3,'Br':2,'I':2}
atom_symbs = ['C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', \
                  'S', 'Se', 'As', 'Cl', 'Si', 'Ge', 'Br']
prop_names = ['NMR', 'ALPHA', 'MU', 'AE', 'G','U0','U','H', \
              'HOMO','LUMO','GAP', 'IP','EA', 'OMEGA1', \
              'MP','BP']

cmdout2 = io2.cmdout2
cmdout = io2.cmdout

T, F = True, False

def get_ln(f, patt, idx=None):
    """ get line number of patt """
    #xtra = 'tail -n 1 | ' if idx==-1 else ''
    cmd = "grep -ni '%s' %s | sed 's/:/ /g' | awk '{print $1}'"%(patt, f)
    #print("%s"%cmd)
    lns = [ int(si) for si in cmdout(cmd) ]
    if idx is None:
        return lns
    else:
        return lns[idx]


class rawcfg(co.atoms):
    """
    raw configuration
    """
    def __init__(self, idx, zs, coords, grads, props=None):
        co.atoms.__init__(self, zs, coords, props=props)
        self.idx = idx # idx of config in all configs
        self.grads = grads
        #self.e = e


class GR(object): #co.atoms):

    gpatts = [ ' Input orientation:', ' Standard orientation:', ]

    @staticmethod
    def str2array(li):
        """ for geometry or forces """
        x = []
        for ci in li:
            csi = ci.split()
            x.append( [ eval(si) for si in csi[-3:] ] )
        return np.array(x)

    def __init__(self, f, property_names=['es'], unit='h', istart=0):
        """ initialization of atomic object """
        self.f = f
        cs = open(f).readlines()
        irotated = F
        try:
            ln = get_ln(f, self.gpatts[0], idx=-1) + 4
        except:
            print('   *** molecule was rotated. Forces are wrong!')
            if 'grads' in property_names:
                raise Exception(' To write forces, please re-orient mol??')
            irotated = T
            ln = get_ln(f, self.gpatts[1], idx=-1) + 4
        self.irotated = irotated
        zs = []; coords = []
        while T:
            ci = cs[ln]
            if '---' in ci:
                break
            else:
                csi = ci.split()
                zs.append( int(csi[1]) )
                coords.append( [ eval(si) for si in csi[-3:] ] )
                ln += 1
        if len(zs) == 0:
            raise Exception('#read coords failed!')

        #co.atoms.__init__(self, zs, coords)
        self.zs = zs
        self.coords = coords

        self.cs = cs
        self.pns = property_names

    def get_mult(self):
        """ spin multiplicity """
        mult = 1
        return mult

    def get_net_charge(self):
        chg = 0
        return chg

    def write(self, f):
        t = co.atoms(self.zs, self.coords, self.props)
        t.write(f)

    @property
    def props(self):
        if not hasattr(self, '_props'):
            self._props = self.get_properties()
        return self._props

    def get_properties(self):
        """ read properties from g09 output file """
        props = {}
        pns = self.pns
        #print( '    pns = ', pns, self.meth, self.basis )
        if ('e' in pns) or ('es' in pns):
            props.update( {self.meth+self.basis: self.es[-1]} )
        if 'mu' in pns:
          props.update( {'mu': self.mu} )
        if ('alpha' in pns) or ('polar' in pns):
          props.update( {'alpha': self.alpha} )
        if  'nmr' in pns:
          props.update( {'nmr': self.nmr} )
        if 'grads' in pns:
          props.update( {'grads': self.grads} )
        if 'chgs' in pns:
            props.update( {'chgs': self.chgs} )
        return props

    @property
    def istat(self):
        """ is job terminated normally? """
        if not hasattr(self, '_istat'):
            el = open(f).readlines()[-1] # end line
            self._istat = 'Normal termination' in el
        return self._istat

    @property
    def task(self):
        if not hasattr(self, '_task'):
            self._task = self.get_task()
        return self._task

    def get_task(self):
        task = 'e'
        if io2.cmdout2('grep -i " opt[(\s]" %s'%( self.f )):
            task = 'optg'
        return task


    @property
    def meth(self):
        if not hasattr(self, '_meth'):
            self._meth = self.get_meth()
        return self._meth

    @property
    def cmeth(self):
        "contents related to method section in output file """
        if not hasattr(self, '_cm'):
            s = ''
            # get content related to method
            l = get_ln(self.f, '#', idx=0) - 1
            while T:
                ci = self.cs[l]
                if '----' in ci:
                    break
                s += ci.strip()
                l += 1
            self._cm = s.lower()
        return self._cm

    gns = ['g4mp2', 'g4', 'g2mp2', 'g2']

    def get_meth(self):

        idft = F; ign = F
        ifd = F

        #print('** cmeth = ', self.cmeth)
        for meth in ['bv5lyp', 'b3lyp', 'wb97x', 'tpss', 'pbe0', 'pbe']:
            if meth in self.cmeth: #cmdout2('grep -i %s %s'%(meth,self.f)):
                _meth = meth
                idft = T
                ifd = T
                break

        for meth in self.gns:
            if meth in self.cmeth: # cmdout2('grep -i %s %s'%(meth,self.f)):
                _meth = meth
                ign = T
                ifd = T
                break

        meths_i = ['mp2', 'ccsd(t)', 'ccsd', 'qcisd(t)', 'qcisd']
        meths_o = ['mp2', 'cc2',   'cc',     'qci2',   'qci']
        dct = dict(zip(meths_i, meths_o))
        for meth in meths_i:
            #ot = cmdout2('grep -i "%s" %s'%(meth, self.f))
            if meth in self.cmeth: #ot and ('!' in ot):
                _meth = dct[meth]
                ifd = T
                break

        assert _meth, ' ** failed to extract method!'
        # dispersion interaction?
        if idft:
            #ot = cmdout2('grep GD3 %s'%self.f)
            if 'disp' in self.cmeth: #ot and ('!' in ot):
                _meth += 'd3'
        return _meth

    bsts =  ['aug-cc-pvdz', 'aug-cc-pvtz', 'aug-cc-pvqz', \
             'cc-pvdz', 'cc-pvtz', 'cc-pvqz', \
             'def2-sv(p)', 'def2-svp', 'def2-tzvp', 'def2-qzvp', \
             '6-31g(d)']
    bsts_short = ['avdz', 'avtz', 'avqz', 'vdz', 'vtz','vqz', 'def2sv-p', 'def2svp', 'def2tzvp', 'def2qzvp', '631gd']
    dctb = dict(zip(bsts, bsts_short))

    @property
    def basis(self):
        if not hasattr(self, '_basis'):
            self._basis = self.get_basis()
        return self._basis

    def get_basis(self):
        _bst = None
        if self.meth in self.gns:
            _bst = ''
        else:
            for b in self.bsts:
                if b in self.cmeth: #cmdout('grep -i "%s" %s'%(b,self.f)):
                    _bst = self.dctb[b]
                    break
        assert _bst is not None, 'Plz add more reference basis'
        return _bst

    def update_dic(self, ss1, ss2):
        for i, s1 in enumerate(ss1):
            self.dic[s1] = ss2[i]

    def get_ei(self, ein): # `ein -- energy_i name, i.e., HF, MP2, CCSD(T)
        self.update_dic([ein, ], [ self.dic[ein]*self.const, ])

    def get_config(self, idx=None, gradmax=None, gradrms=None, dispmax=None):
        cfg = None
        if idx is None:
            if gradmax:
                ifd = F
                for i in range(self.nconfig):
                    cfg = self.traj[i]
                    if np.max(np.abs(cfg.grads)) <= gradmax:
                        ifd = T
                        break
                if not ifd:
                    raise Exception(' ** failed to find any config satisfying the given rule')
            else:
                raise Exception('** No rule for config')
        else:
            assert isinstance(idx, int)
            cfg = self.traj[idx]
        assert cfg is not None
        return cfg

    @property
    def nconfig(self):
        if not hasattr(self, '_nconfig'):
            self._nconfig = len(self.traj)
        return self._nconfig

    @property
    def traj(self):
        if not hasattr(self, '_traj'):
            self._traj = self.get_traj()
        return self._traj

    def get_traj(self):
        """ get trajectory of geometries for opt/md jobs """
        cfgs = []
        na = self.na
        f = self.f
        if self.task in ['opt', 'optg']:
            coords = []; grads = []
            gpatt = self.gpatts[0]
            if self.irotated:
                patt = self.gpatts[1] # 'Z-Matrix orientation'
            #    lns = get_ln(f, patt, idx=0)
            #    assert len(lns) == 1
            #    i = lns[0] + 4
            #    coords.append( GR.str2array(self.cs[i:i+na]) )

            patt = gpatt # ' Input orientation:'
            for ln in get_ln(f, patt):
                i = ln + 4
                coords.append( GR.str2array(self.cs[i:i+na]) )

            patt = '       Forces (Hartrees'
            grads = []
            for ln in get_ln(f, patt):
                i = ln + 2
                #print('csi= ', self.cs[i-1:i+na])
                grads.append( GR.str2array(self.cs[i:i+na]) )

            n1 = len(coords); n2 = len(grads)
            if n1 != n2:
                print('    *** warning: num of configs and forces not match')
            n = min(n1,n2) # sometimes,
            key = self.meth + self.basis
            for j in range(n):
                cfgs.append( rawcfg(j, self.zs, coords[j], grads[j], {key:self.es[j]}) )
        else:
            print(' ** No traj avail')
        return cfgs

    @property
    def es(self):
        if not hasattr(self, '_es'):
            self._es = self.get_energies()
        return self._es

    def get_energies(self):
        f = self.f
        xtra = '' # 'tail -n 1 | '
        if self.meth in ['b3lyp', 'b3lypd3', 'bv5lyp', 'bv5lypd3']: #B3LYP', 'RB3LYP', 'UB3LYP']:
            cmd = "grep 'LYP) =' %s | %sawk '{print $5}'"%(f, xtra) #out
            #print('cmd= ', cmd)
            ss = cmdout2(cmd)
        elif self.meth in ['cc', 'cc2', 'qci', 'qci2']:
            #{'UQCISD(T)':'QCISD(T)', 'QCISD(T)':'QCISD(T)', 'UCCSD(T)':'CCSD(T)', 'CCSD(T)':'CCSD(T)'}
            cmd = "grep -i '^ q?c[ci]sd' %s | %sawk '{print $2}'"%(f, xtra)
            ss = cmdout2(cmd)
        elif self.meth in ['mp2']:
            cmd = "grep 'EUMP2 = ' %s | awk '{print $NF}'"%f
            ss = cmdout2(cmd)
        else:
            print(' meth="%s"'%self.meth)
            print('#ERROR: not implemented yet'); sys.exit(3)
        es = []
        for se in ss.split('\n'):
            if 'D' in se:
                e = eval( 'E'.join( se.split('D') ) )
            else:
                e = eval(se)
            es.append(e)
        return es

    def fmo(self):
        if not hasattr(self, '_fmo'):
            self._fmo = self.get_fmo_energy()
        return self._fmo

    def get_is_spin_polarised(self):
        isp = F
        cmd = ''
        raise Exception('Todo')
        return isp

    def get_fmo_energy(self):
        """
        get HOMO, LUMO and their gap
        """
        f = self.f
        cmd1 = "awk '/^ The electronic state is/{print NR}' %s | tail -n 1"%f
        Ln1 = int(io2.cmdout2(cmd1)) + 1
        cmd2 = "awk '/^          Condensed to atoms \(all electrons\)/{print NR}' %s | tail -n 1"%f
        Ln2 = int(io2.cmdout2(cmd2)) - 1
        cmd = "sed -n '%d,%dp' %s | grep Beta"%(Ln1,Ln2,f)
        cont0 = io2.cmdout2(cmd)
        if self.isp:
            print(' ## spin polarised!')
            homo = None; lumo = None
        else:
            cmd = "sed -n '%d,%dp' %s | grep ' Alpha virt. eigenvalues --' | head -n 1"%(Ln1,Ln2,f)
            ct = io2.cmdout2(cmd);
            lumo = eval( ct.split()[-5] ) * self.uo.h2e
            cmd = "sed -n '%d,%dp' %s | grep ' Alpha  occ. eigenvalues --' | tail -1"%(Ln1,Ln2,f)
            ct = io2.cmdout2(cmd)
            homo = eval( ct.split()[-1] ) * self.uo.h2e
        return homo, lumo

    @property
    def gap(self):
        return self.fmo[-1] - self.fmo[0]

    def get_thermo(self, scale_factor=0.965):
        """
        for DFT, the `scale_factor is 0.965
        """
        f = self.f
        cmd = "grep '^ Freq' %s"%f
        # data in lines below could be retrieved together
        #
        # Sum of electronic and zero-point Energies=           -174.095490
        # Sum of electronic and thermal Energies=              -174.091365
        # Sum of electronic and thermal Enthalpies=            -174.090421
        # Sum of electronic and thermal Free Energies=         -174.122620
        conts = io2.cmdout(cmd)
        assert conts != [], '#ERROR: no thermochem was done!'
        if not isGn:
            cmd = "awk '/^ Sum of electronic and/{print $NF}' %s"%f
            U0, U, H, G = np.array( io2.cmdout(cmd) ).astype(np.float) * self.const
            self.U0, self.U, self.H, self.G  = U0, U, H, G
            cmd = "awk '/^ Zero-point correction=/{print $3}' %s"%f
            zpe = eval(io.cmdout(cmd)[0]) * self.const # the last two entries are "0.81475823 Hartree/atom"
            self.zpe = self.zpve = zpe
        else:
            print('#ERROR: cannot handle output of Compositional method like G4MP2 yet!')
            sys.exit(2)
        E_ = U0 - zpe
        #assert abs(E_ - E) < 0.0001, '#ERROR: '

        # people usually scale ZPE by a factor, for B3LYP, it's 0.965??
        U0c = E_ + scale_factor*zpe
        self.U0c = U0c
        self.update_dic(['U0', 'U0c', 'U', 'H', 'G'], [U0, U0c, U, H, G])

    @property
    def mu(self):
        if not hasattr(self, '_mu'):
            self._mu = self.get_dipole_moment()
        return self._mu

    def get_dipole_moment(self):
        cmd = "grep -A1 ' Dipole moment (field-independent basis, Debye):' %s | tail -1 | awk '{print $NF}'"%self.f
        mu = eval( io2.cmdout2(cmd) )
        return mu

    @property
    def alpha(self):
        if not hasattr(self, '_alpha'):
            self._alpha = self.get_polarizability()
        return self._alpha

    def get_polarizability(self):
        cmd = "awk '/Isotropic polarizability/{print $6}' %s"%self.f
        a = None
        try:
            a = eval( io2.cmdout2(cmd) )
        except:
            print(' * WARNING: no Isotropic polarizability found')
        return a

    @property
    def nmr(self):
        if not hasattr(self, '_nmr'):
            self._nmr = self.get_nmr()
        return self._nmr

    def get_nmr(self):
        cmd = "awk '/  Isotropic =  /{print $5}' %s"%self.f
        vals = io2.cmdout(cmd)
        nmr = np.array([ eval(val) for val in vals ])
        return nmr

    @property
    def chgs(self):
        if not hasattr(self, '_chgs'):
            self._chgs = self.get_mulliken_population()
        return self._chgs

    def get_mulliken_population(self):
        # note that the output Mulliken charge starting line may be different
        # for different versions of Gaussian, i.e., with or without `atomic` in between
        # "Mullike" and "charges:", so a "robust" regular expression is used
        cmd1 = "awk '/^ Mulliken [a-zA-Z]*\s?charges:/{print NR}' %s | tail -1"%self.f
        Ln1 = int(io2.cmdout2(cmd1)) + 2
        cmd2 = "awk '/^ Sum of Mulliken [a-zA-Z]*\s?charges =/{print NR}' %s | tail -1"%self.f
        Ln2 = int(io2.cmdout2(cmd2)) - 1
        cmd = "sed -n '%d,%dp' %s"%(Ln1,Ln2,self.f)
        cs = io2.cmdout2(cmd).split('\n')
        pops = np.array([ eval(ci.split()[2]) for ci in cs ])
        return pops

    def get_force(self):
        iou = io2.Units()
        const = iou.h2e / iou.b2a # from hartree/bohr to eV/A
        #cmd = "grep '^\s*[XYZ][0-9]*   ' %s | awk '{print $3}'"%self.f #
        cmd1 = "awk '/^ Variable       Old X    -DE/{print NR}' %s | tail -1"%self.f
        Ln1 = int(io2.cmdout2(cmd1)) + 2
        cmd2 = "awk '/^         Item               Value     Threshold  Converged/{print NR}' %s | tail -1"%self.f
        Ln2 = int(io2.cmdout2(cmd2)) - 1
        cmd = "sed -n '%d,%dp' %s"%(Ln1,Ln2,self.f)
        #print cmd
        cs = io2.cmdout2(cmd).split('\n')
        vals = [ eval(ci.split()[2]) for ci in cs ]
        #print vals
        #print len(vals)
        forces = np.array(vals).reshape((self.na, 3)) * const
        #abs_forces = np.linalg.norm( forces, axis=1 )
        #self.forces = abs_forces[:, np.newaxis]
        return forces


    def get_cf_and_chg(self):
        """ cf: chemical formula
            chg: charge """
        cmdi = "grep Stoichiometry %s | tail -1"%f
        ct = io2.cmdout2(cmdi)
        info = ct.split()[-1]
        if ('(' in info) and (info[-1] == ')'):
            # charged mol
            cf, c0_ = info.split('(') # cf: chemical formula;
            if c0_[-2] == '-':
                chg = -int(c0_[:-2]) # negatively charged
            elif c0_[-2] == '+':
                chg = int(c0_[:-2]) # positively charged
            else:
                #print 'c0_ = ', c0_, ', ct = ', ct
                #sys.exit(2)
                chg = 0 # spin-polarized case, e.g., C4H9O(2)
        else:
            chg = 0
            cf = info
        return cf, chg

    def get_spin(self):
        cmdi = "grep 'Multiplicity =' %s | tail -1"%self.f
        ct = cmdout2(cmdi).split()
        chg = int(ct[2]); sm = int(ct[-1])
        return sm, chg



class GRs(object):

    def __init__(self, fs, properties=['AE'], unit='kcal', write_Y=False, nproc=1, istart=0):

        self.n = len(fs)
        typeP = type(properties)
        if typeP is str:
            properties = [ properties.upper(), ]
        elif typeP is list:
            properties = [ prop.upper() for prop in properties ]
        else:
            print('#ERROR,')
            raise
        npr = len(properties)

        if nproc == 1:
            self.objs = []
            for i,f in enumerate(fs):
                #print i+1, f
                ipt = [ f, properties, unit, istart ]
                self.objs.append( self.processInput(ipt) )
        else:
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ [ fi, properties, unit ] for fi in fs ]
            self.objs = pool.map(self.processInput, ipts)

        ys = []
        #print ' - npr = ', npr
        for ipr in range(npr):
            ysi = []
            for obj in self.objs:
                #print ' - obj.properties = ', obj.properties
                yi = obj.properties[ipr]; #print ' ysi = ', ysi
                ysi.append( yi )
            ys.append( ysi )
        #print ys
        self.dic = dict( list(zip(properties, ys)) )
        if write_Y:
            for ipr in range(npr):
                np.savetxt('%s.dat'%properties[ipr], ys[ipr], fmt='%.2f')

    def processInput(self, ipt):
        f, properties, unit, istart = ipt
        obj = GR(f, properties=properties, unit=unit, istart=istart)
        return obj

    def get_statistics(self):
        """
        `zs, `nas, `nhass, `coords, etc
        """
        zs = []
        zsr = []
        nas = []
        nhass = []
        zsu = set([])
        coords = []
        for i in range(self.n):
            obj_i = self.objs[i]
            coords.append( obj_i.coords )
            zsi = obj_i.zs
            nhass.append( (np.array(zsi)>1).sum() )
            nas.append( len(zsi) )
            zsu.update( zsi )
            zsr += list(zsi)
            zs.append( zsi )
        zsu = list(zsu)
        nzu = len(zsu)
        zsu.sort()
        nzs = np.zeros((self.n, nzu), np.int32)
        for i in range(self.n):
            for iz in range(nzu):
                tfs = np.array(zs[i]) == zsu[iz]
                nzs[i,iz] = np.sum(tfs)
        self.nzs = nzs
        self.zsu = zsu
        self.zs = zs
        self.zsr = np.array(zsr,np.int32)
        self.nas = np.array(nas,np.int32)
        self.nhass = np.array(nhass,np.int32)
        self.coords = coords





def parse_ifile(ifile):

    f = open(ifile, "r")
    lines = f.readlines()
    f.close()

    tokens = lines[0].split()

    natoms = int(tokens[0])
    deriv  = int(tokens[1])
    charge = int(tokens[2])
    spin   = int(tokens[3])

    print("-------------------------------------")
    print("--  GOPTIMIZER INPUT  ---------------")
    print("-------------------------------------")
    print()
    print("  Number of atoms:     ", natoms)
    print("  Derivative requested:", deriv)
    print("  Total charge:        ", charge)
    print("  Spin:                ", spin)
    print()

    coords = np.zeros((natoms,3))
    atomtypes = []

    for i, line in enumerate(lines[1:1+natoms]):

        tokens = line.split()
        a = NAME[int(tokens[0])]

        c = np.array([float(tokens[1]), float(tokens[2]),float(tokens[3])])*a2b

        coords[i,0] = c[0]
        coords[i,1] = c[1]
        coords[i,2] = c[2]

        atomtypes.append(a)

    print("  Found the following atoms:")
    print("  --------------------------")
    print()


    for i in range(natoms):
        print("  Atom %3i  %-3s   %20.12f %20.12f %20.12f" % \
                (i, atomtypes[i], coords[i,0], coords[i,1], coords[i,2]))


    print()
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print()


    return natoms, deriv, charge, spin, atomtypes, coords


def write_ofile(ofile, energy, natoms, dipole=None, gradient=None,
        polarizability=None, dipole_derivative=None):

    # Define output formats
    headformat = ff.FortranRecordWriter("4D20.12")
    bodyformat = ff.FortranRecordWriter("3D20.12")

    # Print output header:
    if dipole is None:
        dipole = np.zeros((3))

    head = [energy, dipole[0], dipole[1],dipole[2]]


    f = open(ofile, "w")

    # Write headers
    headstring = headformat.write(head)
    f.write(headstring + "\n")


    # Write gradient section
    if gradient is None:
        gradient = np.zeros((natoms,3))

    assert gradient.shape[0] == natoms, "ERROR: First dimension of gradient doesn't match natoms."
    assert gradient.shape[1] == 3, "ERROR: Second dimension of gradient is not 3."

    for i in range(natoms):
        output = bodyformat.write(gradient[i])
        f.write(output+ "\n")

    # Write polarization section
    if polarizability is None:
        polarizability = np.zeros((2,3))

    for i in range(2):
        output = bodyformat.write(polarizability[i])
        f.write(output+ "\n")

    # Write dipole derivative section
    if dipole_derivative is None:
        dipole_derivative = np.zeros((3*natoms,3))

    for i in range(3*natoms):
        output = bodyformat.write(dipole_derivative[i])
        f.write(output+ "\n")

    f.close()



if __name__ == "__main__":
    import sys
    import stropr as so

    args = sys.argv[1:]

    idx = 0
    keys=['-p','-properties']; hask,s,idx = so.parser(args,keys,'E',idx)
    assert hask
    props = s.split(',')

    fs = args[idx:]

    #test
    obj = GRs(fs, properties=props)
    for propi in props:
        for (f,p) in zip(fs,obj.dic[propi]):
            print(f,p)

