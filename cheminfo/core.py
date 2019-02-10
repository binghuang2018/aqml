# -*- coding: utf-8 -*-

from cheminfo import *
from cheminfo.rw.xyz import *
from cheminfo.rw.sdf import *
import cheminfo.data.atoms as cda
import numpy as np
import ase, os, io2, sys
import cheminfo as co
import cheminfo.molecule.core as cmc
from ase.calculators.dftd3 import DFTD3
import multiprocessing as mt
import numpy.linalg as LA
import sympy as spy

def obj2m(obj, property_names=None, isimple=F, idx=None, unit=None):
    assert unit is not None

    if property_names:
        if ('nmr' in property_names) or ('chgs' in property_names):
            isimple = T

    if isinstance(obj, str):
        if (not os.path.isfile(obj)):
            raise Exception('#ERROR: file does not exist')
        if (obj[-3:] == 'xyz'):
            func = read_xyz_simple if (isimple and (idx is None)) else read_xyz
            nas, zs, coords, nsheav, props = func(obj, property_names=property_names, idx=idx)
        elif (obj[-3:] == 'sdf'):
            property_names = []
            _mi = read_sdf(obj, iconn=F)
            nas = [ len(_mi) ]; zs = _mi.numbers; coords = _mi.positions; nsheav = [ (zs>1).sum() ]
            props = {}
        else:
            raise Exception('#ERROR: non-xyz file')
    elif isinstance(obj, list):
        nas = []; zs = []; coords = []; nsheav = []; props = {}
        for _obj in obj:
            #print('type(obj)=',type(obj))
            if (_obj[-3:] == 'xyz'):
                func = read_xyz_simple if (isimple and (idx is None)) else read_xyz
                _nas, _zs, _coords, _nsheav, _props = func(_obj, property_names=property_names, idx=idx)
            elif (_obj[-3:] == 'sdf'):
                property_names = []
                _mi = read_sdf(_obj, iconn=F)
                _nas = [ len(_mi) ]; _zs = _mi.numbers; _coords = _mi.positions; _nsheav = [ (_zs>1).sum() ];
                _props = {}
            else:
                raise Exception('#ERROR: not sdf/xyz file')
            zs += list(_zs)
            coords += list(_coords)
            nsheav += list(_nsheav)
            nas += list(_nas)
            #print('props=', _props)
            property_names = list(_props.keys())
            for p in property_names:
                yi = _props[p]
                if p in props.keys():
                    props[p] += [yi]
                else:
                    props[p] = [yi]
    #elif 'molecules' in obj.__str__():
    #    nas, zs, coords, nsheav, props = obj._nas, obj._zs, obj._coords, obj._nsheav, obj.props
    elif 'catoms' in obj.__str__():
        nas, zs, coords, nsheav, props = obj.nas, obj.zs, obj.coords, obj.nsheav, obj.props
    else:
        print('obj=',obj)
        raise Exception('#ERROR: unknown input `obj!')

    const = 1.0
    for p in props:
        if unit in ['h','ha']:
            const = io2.Units().h2kc
        v = props[p]
        if p in ['chg','chgs','nmr','cls','force','forces','grad','grads','alpha']:
            const = 1.0
        props[p] = np.array(v) * const

    return nas, zs, coords, nsheav, props


def get_nsheav(nas,zs):
    ims2 = np.cumsum(nas)
    ims1 = np.concatenate(([0],ims2[:-1]))
    nsheav = []
    for i in range(len(nas)):
        nsheav.append( (np.array(zs)[ims1[i]:ims2[i]]>1).sum() )
    return np.array(nsheav,np.int)


class catoms(object):
    """ collection of atoms
    I.e., a list of atoms objects
    """
    def __init__(self, nas, zs, coords, nsheav=None, props={}): #property_names=None):
        self.nas = np.array(nas, dtype=int)
        self.zs = np.array(zs,dtype=int)
        self.coords = np.array(coords)
        self.symbols = np.array([ chemical_symbols[zi] for zi in self.zs ])
        _props = {}
        if len(props) > 0:
            for k in props.keys():
                _props[k] = np.array(props[k])
        self.props = _props
        self.nsheav = get_nsheav(nas,zs) if nsheav is None else np.array(nsheav,np.int)
        self.nm = len(self.nas)
        self.ias2 = np.cumsum(self.nas)
        self.ias1 = np.concatenate(([0],self.ias2[:-1]))



class atoms(object):
    """
    a single molecule
    """
    def __init__(self,zs,coords,props=None):
        if isinstance(zs[0],str):
            zs = [ chemical_symbols.index(_) for _ in zs ]
        self._zs = list(zs)
        self.zs = np.array(zs, np.int)
        self._symbols = [ chemical_symbols[zi] for zi in zs ]
        self.symbols = np.array(self._symbols)
        self._coords = [ list(coords_i) for coords_i in coords ]
        self.coords = np.array(coords)
        self.na = len(zs)
        self.nheav = (self.zs>1).sum()
        self.props = {}
        if props is not None:
            self.props = {'E':props} if isinstance(props,(float,np.float64)) else props

    def write(self, f):
        so = '%d\n'%( self.na )
        if len(self.props) > 0:
            for key in self.props.keys():
                so += '%s=%s '%(key,str(self.props[key]))
        so += '\n'
        for si,(x,y,z) in zip(self._symbols, self._coords):
            so += '%-6s %15.8f %15.8f %15.8f\n'%(si,x,y,z)
        with open(f,'w') as fid: fid.write(so)


class molecule(co.atoms):
    """
    a single molecule read from a file or ...
    """
    def __init__(self, obj, props=None, isimple=F, unit='h'):
        nas, zs, coords, nsheav, props = obj2m(obj, props, isimple=isimple, unit=unit)
        co.atoms.__init__(self, zs, coords, props=props)



class Matrix(object):

    def __init__(self, ns):
        _ns = spy.Matrix(ns)
        reduce_form, idx = _ns.rref()
        self.ngs = ns[:, idx]
        self.rank = len(idx)
        self.idx = idx # idx of independent cols


class molecules(catoms):
    """
    a list of molecules
    """

    def __init__(self, objs, property_names=None, idx=None, isimple=F, unit='h'):
        """ initialization """
        """
        unit: the unit of input data, to be converted to kcal/mol
        """
        if isinstance(objs, list):
            assert len(objs)>0, '#ERROR: input has length 0?'
        _nas, _zs, _coords, _nsheav, _props = obj2m(objs, property_names=property_names, idx=idx, isimple=isimple, unit=unit)
        self.objs = objs
        if property_names:
            if property_names[0] in ['a','all']:
                #print('keys=', list(_props.keys()))
                property_names = list(_props.keys())
        self.property_names = property_names
        catoms.__init__(self, _nas, _zs, _coords, _nsheav, _props)
        self.i_set_scu = F

    def update(self, newobjs, idx=None):
        _nas, _zs, _coords, _nsheav, _props = obj2m(newobjs, property_names=self.property_names,idx=idx, isimple=isimple)
        self.append( _nas, _zs, _coords, _nsheav, _props )

    def clone(self):
        return molecules(catoms(self.nas,self.zs,self.coords,self.nsheav,self.props))

    @property
    def ys(self):
        if not hasattr(self, '_ys'):
            pns = list(self.props.keys())
            assert len(pns)==1
            #    raise Exception('multiple properties avail, please specify one only')
            pn = pns[0]
            self._ys = np.array(self.props[pn])
        return self._ys

    @property
    def nsi(self):
        if not hasattr(self, '_nsi'):
            self._nsi = self.get_nsi()
        return self._nsi

    def get_nsi(self):
        """
        get the number of mols for each `NI
        """
        nsi = []
        nsheav_u = np.unique(self.nsheav)
        for i,na in enumerate(nsheav_u):
            nm = (self.nsheav == na).sum()
            nmc = (self.nsheav <= na).sum()
            nsi.append( [na, nm, nmc] )
        return nsi

    @property
    def ase_objs(self):
        """ to a list of ase.Atoms objects """
        if not hasattr(self, '_ase'):
            objs = []
            ims2 = np.cumsum(self.nas)
            ims1 = np.concatenate(([0],ims2[:-1]))
            for i in range(self.nm):
                ib = ims1[i]; ie = ims2[i]
                obj = ase.Atoms(self.zs[ib:ie], self.coords[ib:ie])
                objs.append(obj)
            self._ase = objs
        return self._ase

    def __getitem__(self, i):
        if isinstance(i,int):
            idx=[i]
        else:
            idx = i
        nas = self.nas[idx]
        nsheav = self.nsheav[idx]
        zs = []; coords = []; p = {}
        for k in self.props.keys():
            p[k] = self.props[k][idx]
        ims2 = np.cumsum(self.nas)
        ims1 = np.concatenate(([0],ims2[:-1]))
        for i,ii in enumerate(idx):
            ib = ims1[ii]; ie = ims2[ii]
            zs = np.concatenate((zs,self.zs[ib:ie]))
            if i==0:
                coords = self.coords[ib:ie]
            else:
                coords = np.concatenate((coords,self.coords[ib:ie]) )
        return molecules(catoms(nas, zs, coords, nsheav, p))

    def __add__(self, o):
        nas = np.concatenate((self.nas, o.nas)).astype(np.int)
        zs = np.concatenate((self.zs, o.zs)).astype(np.int)
        coords = np.concatenate((self.coords, o.coords))
        assert o.nsheav is not None, '#ERROR: o.nsheav is None?'
        nsheav = np.concatenate((self.nsheav, o.nsheav))
        _props = self.props.copy()
        for k in o.props.keys():
            _props[k] = np.concatenate((_props[k], o.props[k]))
        return molecules(catoms(nas, zs, coords, nsheav, _props))


    def get_nzs(self):
        nzs = []
        nm = len(self.nas)
        for i in range(nm):
            ib, ie = self.ias1[i], self.ias2[i]
            zsi = self.zs[ib:ie]
            nzsi = []
            for zi in self.zsu:
                nzsi.append( (zsi==zi).sum() )
            nzs.append(nzsi)
        return np.array(nzs, dtype=int)

    @property
    def nzs(self):
        if not hasattr(self, '_nzs'):
            self._nzs = self.get_nzs()
        return self._nzs


    @property
    def zsu(self):
        if not hasattr(self, '_zsu'):
            self._zsu = np.unique(self.zs)
        return self._zsu

    @property
    def zmax(self):
        return np.max(self.zs)

    def set_scu(self, opt):
        """ set the smallest constituting unit of a mol to either
        atom or group """
        self.scu = opt
        self.i_set_scu = T

    @property
    def nscus(self):
        if not hasattr(self, '_nscus'):
            self._nscus = self.get_nscus()
        return self._nscus

    @property
    def scus(self):
        if not hasattr(self, '_scus'):
            self._scus = self.get_scus()
        return self._scus

    def get_scus(self):
        """ get the smallest constituting units of mols """
        grps = []

        if not self.i_set_scu:
            self.set_scu('atom')
        if self.scu in ['a','atom']:
            for i in range(self.nm):
                ib, ie = self.ias1[i], self.ias2[i]
                zsi = self.zs[ib:ie]
                grpsi = {}
                ias = np.arange(len(zsi))
                for zj in self.zsu:
                    grpsi.update( {zj: ias[zsi==zj]} )
                grps.append(grpsi)
        elif self.scu in ['g', 'group']:
            for i in range(self.nm):
                ib, ie = self.ias1[i], self.ias2[i]
                zsi = self.zs[ib:ie]
                coordsi = self.coords[ib:ie]
                mi = cmc.RawMol( (zsi,coordsi) )
                grps.append(mi.nscu) #grpsi )
        else:
            raise Exception(' `scu not accepted')
        return grps

    @property
    def scuu(self):
        """ get the smallest consistituting groups that are unique """
        if not hasattr(self, '_scuu'):
            _scuu = set()
            for i in range(self.nm):
                _scuu.update( list(self.scus[i].keys()) )
            t = list(_scuu); t.sort()
            self._scuu = t
        return self._scuu

    def get_nscus(self):
        """ get the number of smallest consistituting groups in each mol """
        grpsu = self.scuu
        nscus = np.zeros((self.nm, len(grpsu)))
        for i in range(self.nm):
            grpsi = self.scus[i]
            for j,grp in enumerate(grpsu):
                if grp in grpsi: nscus[i,j] = len(grpsi[grp])
        return np.array(nscus,dtype=int)

    def get_ngs(self):
        return self.get_nscus()

    @property
    def ngs(self):
        """ counts of each group in mols """
        if not hasattr(self, '_ngs'):
            self._ngs = self.get_ngs()
        return self._ngs

    @property
    def iasz(self):
        """
        return relative idx of atoms in each subset of atoms
        associated with different Z
        """
        if not hasattr(self, '_iasz'):
            nat = int(np.sum(self.nas))
            ias = np.arange(nat).astype(np.int)
            iast = ias.copy()
            for i,zi in enumerate(self.zsu):
                ias_z = ias[self.zs==zi]
                iast[ias_z] = np.arange(len(ias_z))
            self._iasz = iast
        return self._iasz

    @property
    def namax(self):
        """ maximal number of heavy atoms of molecules """
        return np.max(self.nsheav)

    def update_property(self, obj):
        self.property_names += list(obj.props.keys())
        self.props.update( obj.props )

    def get_atomization_energies(self, prog, meth):
        """ get AE """
        udct = {'h':io2.Units().h2kc}
        const = 1.0
        if meth in ['a','all']:
            meths = list(self.props.keys())
        else:
            meths = [meth]
        aes = {}
        for mt in meths:
            _es0 = np.zeros(1+self.zsu[-1])
            for zi in self.zsu:
                si = chemical_symbols[zi]
                #try:
                ei0 = cda.dct[prog][si][mt]; #print('ei0=',ei0)
                #except:
                #    ei0 = cda.dct[prog][mt][si]
                _es0[zi] = ei0 * udct['h']
            _aes = []
            for i,na in enumerate(self.nas):
                ib, ie = self.ias1[i], self.ias2[i]
                zsi = self.zs[ib:ie]
                ei0 = np.sum([_es0[zsi]])
                #print('props=', self.props)
                _aes.append( self.props[mt][i] * const - ei0 )
            aes[mt] = np.array(_aes)
            #self.props[mt] = np.array(aes)
        self.aes = aes

    def calc_ae_dressed(self, idx1, idx2=None, meth=None, ref=None):
        """
        calc dressed atom or/and bond energies as baseline

        vars
        ====================
        ref: if set to None, then E_atom will be regressed by input mols
             otherwise, E_atom will be regressed from a set of small mols
             located in folder `ref/ (i.e., h2,ch4,n2,o2,f2,s8,p4,cl2,br2,i2)
        """
        ims = np.arange(self.nm)
        if isinstance(idx1,int):
            idx1 = ims[:idx1]
        if idx2:
            if isinstance(idx2,int):
                idx2 = ims[idx2:]
        else:
            idx2 = np.setdiff1d(ims, idx1)
        if meth is None:
            meths = list(self.props.keys())
            if len(meths) > 1:
                raise Exception('#ERROR: multiple property avail. Please specify one!')
            meth = meths[0]
        ns1, ns2 = self.ngs[idx1], self.ngs[idx2]
        nel = len(ns1[0])
        uc = io2.Units()
        const = 1.0 #{'h': uc.h2kc, 'kcal':1.0 }[iu]
        if hasattr(self, 'ys'):
            ys = self.ys
        else:
            ys = self.props[meth]
        ys *= const
        ys1, ys2 = ys[idx1], ys[idx2]
        istat = T
        if ref is None: # default case
            esb,_,rank,_ = np.linalg.lstsq(ns1,ys1,rcond=None)
            if rank < nel:
                print( ' ** number of molecules .le. number of elements' )
                esb = np.zeros(nel)
                istat = F
        else:
            esb = get_reference_atomic_energy(ref, meth, self.zsu)
        ys1p = np.dot(ns1,esb)
        #print ' +++++ ys1.shape, ys1p.shape = ', ys1.shape, ys1p.shape
        dys1 = ys1 - ys1p
        ys2_base = np.dot(ns2,esb)
        dys2 = ys2 - ys2_base
        return istat, dys1, dys2, ys2_base

    def calc_ecbs(self):
        """ calculate extrapolated energies: HF/cbs, CC2/cbs ..."""
        nm = len(self.nas)
        ms0 = self.property_names # methods0
        d = self.props
        d2 = []
        _bsts = ['vdz','vtz','vqz']
        for l in range(nm):
            #print('meths,ys=',ms0,d)
            ysl = [ d[m0][l] for m0 in ms0 ]

            dl = dict(zip(ms0, ysl))
            _mshf = ['hf'+si for si in _bsts]
            cbsts = ['v23z','v34z']

            # cc2cbs2vNz
            for m in ['hf','mp2','cc2']:
                _ms = [m+si for si in _bsts ]
                for i in range(2):
                    cbst = cbsts[i]; ms = _ms[i:i+2] # 'v23z'; ms = _ms[:2]
                    mshf = _mshf[i:i+2]
                    if set(ms) <= set(ms0):
                        esi = [dl[ms[0]],dl[ms[1]]] if m=='hf' else [dl[ms[0]]-dl[mshf[0]],dl[ms[1]]-dl[mshf[1]]]
                        o = cda.extrapolate(esi, cbst)
                        e = o.hfcbs if m=='hf' else o.corrcbs + dl['hfcbs'+cbst]
                        se = m+'cbs'+cbst
                        #if m=='mp2':
                        #    print( {se:e} )
                        if se not in dl.keys():
                            dl.update({se:e})
                        else:
                            if abs(e-dl[se])>0.0004:# Hartree
                                print(" #mol%05d, meth=%s: default=%.8f, new=%.8f"%(l+1,se, dl[se], e))
                                raise Exception('inconsistent?')
            #print('dl=',dl)
            # cc2cbsmp2vNz
            for i in range(2):
                cbst = cbsts[i]
                bst = _bsts[i]
                meths = ['mp2cbs'+cbst] + [ mi+bst for mi in ['mp2','cc2'] ]
                mmp2 = 'mp2cbs'+cbst
                if mmp2 in dl.keys():
                    mp2cbs, cc2, mp2 = [ dl[_mt] for _mt in [mmp2, 'cc2'+bst, 'mp2'+bst ] ]
                    se = 'cc2cbsmp2'+cbst
                    e = mp2cbs + cc2 - mp2
                    if se not in dl:
                        dl.update({se:e})
                    else:
                        if abs(e-dl[se]) > 0.0005:
                            print(" #mol%05d, meth=%s: default=%.8f, new=%.8f"%(l+1,se, dl[se], e))
                            raise Exception('inconsistent?')
            ms1 = dl.keys()
            d2i = []
            for mt in ms1: d2i.append( dl[mt] )
            d2.append( d2i )
        d2 = np.array(d2)
        dct = {}
        for im,mi in enumerate(ms1):
            dct[mi] = d2[:,im]
        self.props = dct

    def calc_dftd3_energy(self, xc, iabc=F, nprocs=1):
        """ Grimme's D3 correction to energy """
        ipts = [ [fi,xc,iabc] for fi in self.objs ]
        if nprocs == 1:
            es = []
            for ipt in ipts:
                es.append( get_dftd3_energy(ipt) )
        else:
            pool = mt.Pool(processes=nprocs)
            es = pool.map(get_dftd3_energy, ipts)
        return np.array(es)


def get_dftd3_energy(ipt):
    """ Grimme's D3 correction to energy """
    fxyz, func, iabc = ipt
    sabc = ' -abc' if iabc else ''
    cmd = "dftd3 %s -func %s -bj%s | grep Edisp | awk '{print $NF}'"%(fxyz,func,sabc)
    #print(cmd) #; sys.exit(2)
    e = eval(os.popen(cmd).read().strip())
    return e


def get_reference_atomic_energy(ref, meth, zs=None):
    """ get reference atomic energy from a set of mols
    located under directory `ref
    """
    fs = io2.cmdout('ls %s/*.xyz'%ref)
    ms = molecules(fs, [meth])
    #print('fs=',fs,'ms=',ms.props)
    esb,_,rank,_ = np.linalg.lstsq(ms.nzs, ms.props[meth], rcond=None)
    dct = np.zeros((zsu[-1]+1))
    for i,zi in enumerate(ms.zsu):
        dct[zi] = esb[i]
    if zs is None:
        return dct
    return dct[zsu]

