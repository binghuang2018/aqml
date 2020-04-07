
import numpy as np
import io2,os,sys,ase
import algo.krr as krr
import ase.io as aio
import io2.gaussian as iog
import aqml.cheminfo.rw.xyz as crx
import representation.slatm_x as sl
from aqml.cheminfo.core import *
import aqml.cheminfo.core as cc
import aqml.cheminfo.rdkit.core as crc
import aqml.cheminfo.oechem.amon as coa
import aqml.cheminfo.oechem.oechem as coo
import aqml.cheminfo.oechem.amon_extend as coae
import cml.sd as dd
import algo.efit as aefit
from lmfit import Parameters

h2kc = 627.5094738898777
F = False; T = True

cmdout1 = lambda cmd: os.popen(cmd).read().strip()
cmdout = lambda cmd: os.popen(cmd).read().strip().split('\n')

# convert SMILES into hexadecimal string forward and backward
str2hex = lambda x: "".join( [ hex(ord(si)).lstrip("0x").rstrip("L")  for si in x ] )
hex2str = lambda x: ''.join(chr(int(x[i:i+2], 16)) for i in range(0, len(x), 2))

class amons(object):

    def __init__(self, objs, wd='targets/', reduce_namons=T, wg=F, \
                 i3d=T, a1=T, level=2, exta=0, k=7, k2=7, \
                 ff='mmff94', owt=F):

        """
        vars
        ==============
        level: extended amons level, 1 or 2
        exta: maximal N_I of extended amons
        owt: overwrite target (when writing sdf file for target)
        """
        if isinstance(objs, str): objs = [objs]
        if not os.path.exists(wd): os.mkdir(wd)

        self.ff = ff

        ts = []
        fts = []
        # first get 3d geom of targests
        for obj in objs:
            om = crc.RDMol(obj, forcefield=ff)
            om.optg()
            om.optg2()
            ats = om.ats
            can = coo.oem2can( coo.smi2oem(obj)[1] )
            can_hex = str2hex( coo.oem2can( coo.smi2oem(obj)[1] ) )
            f1 = wd+'%s.xyz'%can #_hex
            f2 = wd+'%s.sdf'%can #_hex
            if owt or (not (os.path.exists(f2) and os.path.exists(f1))):
                om.write_sdf(f2)
                om.write_xyz(f1)
            else:
                ats = cc.obj2mol(f1, ['E'])
            fts.append( f2 )
            ts.append( ats )
        self.ts = ts

        a1 = [] # generic amons
        if a1:
            imap = F if len(fts) == 1 else T
            oa = coa.ParentMols(fts, reduce_namons, label=None, \
                          imap=imap, fixGeom=F, i3d=i3d, wg=wg, \
                          k=k,k2=k2, iprt=T, forcefield=ff, \
                          thresh=0.01, debug=F)
            for mi in oa.ms:
                om = crc.RDMol(mi, forcefield=ff)
                #om.optg()
                om.iFFOpt = T
                om.optg2()
                a.append( om.ats )
        self.a1 = a1

        a2 = []
        assert exta >= 0, '#ERROR: `exta: N_I of extended amons shoud be >= 0'
        if exta:
            # add extended amons
            oa2 = coae.transform(objs[0])
            oa2.get_newolds()
            newms, newms2 = oa2.T(level=level)
            oa2.get_amons_extended(k=exta)
            for mi in oa2.amons_extended:
                om = crc.RDMol(mi, forcefield=ff)
                om.optg()
                om.optg2()
                a2.append( om.ats )
        self.a2 = a2
        self.ms = a1 + a2 + ts

    def append(self, newms):
        """ add new molecules for training """
        a3 = []
        for mi in newms:
            om = crc.RDMol(mi, forcefield=self.ff)
            om.optg()
            om.optg2()
            a3.append( om.ats )
        self.ms = self.a1 + self.a2 + a3 + self.ts



class _db(object):
    param = { 'morse': {'1_1-6_3': [108.5835, 32.2093, 1.0847],
                        '6_3-6_3': [133.3824, 3.9740, 1.3098],
                        }
            }


def get_param(_mols, n2=1, idxsr=[], icn=T, iconn=T, iconj=F, \
              use_morse_db=F, cs=None, cs3=None, check_boundary=T):
    """
    fit morse potential and save optimized morse parameters to a file

    Leave the last `n2 molecules for test (i.e., query molecules)
    """
    nm = _mols.nm
    _ims = np.arange(nm-n2)
    ims1 = np.setdiff1d(_ims, idxsr)
    n1 = len(ims1)
    ims2 = np.concatenate( (idxsr, np.arange(nm-n2,nm)) )
    #n2 = len(ims2)
    ims = np.concatenate((ims1,ims2)).astype(np.int)
    mols = _mols[ims]
    mols.ys = _mols.ys[ims]
    reg = aefit.fmorse(mols)
    con = aefit.initb_g(mols, icn=icn, iconn=iconn, iconj=iconj)
    bts = con.mbts2
    xs = con.rs
    N = len(bts)
    reg.prepare_vars(ims1, N, xs)
    itest = F
    if n2>0: itest = T

    params0 = None
    if use_morse_db:
        paramd = _db().param['morse']
        keys = ['a','b','c']
        params0 = Parameters()
        for ibt,bt in enumerate(bts):
            param_i = paramd[bt]
            for j in range(3):
                params0.add('%s_%i'%(keys[j],ibt+1), value=param_i[j])

    param = reg.regressor_lm(itest=itest, check_boundary=check_boundary, \
                             params0=params0, cs=cs, cs3=cs3)
    print('dys1,dys2 = ', reg.dys1.shape, reg.dys2.shape)
    _dys = np.concatenate((reg.dys1,reg.dys2)) if itest else reg.dys1

    seq = np.argsort(ims)
    dys = _dys[seq]
    return bts, param, dys #reg















class qml(object):

    def __init__(self, objs, rcut=4.8, fitmorse=F, property_names=None, \
                 idxsr=None, iae=F, no_strain=F, Delta=F, saveblk=F,\
                 unit='kcal', prog='g09', itarget=F, use_morse_db=F, \
                 check_boundary=T, xparam={}):
        """
        itarget: use target molecule to calc dmax (to save memory) or not?
        """

        self.saveblk = saveblk
        self.itarget = itarget
        if isinstance(objs,(tuple,list)):
            fs = []
            for obj in objs:
                if isinstance(obj,str):
                    if os.path.exists(obj):
                        if os.path.isdir(obj):
                            fs += cmdout('ls %s/*.xyz'%obj)
                        else: # assume a file
                            fs += [obj]
                    else:
                        print( 'input object: %s'%obj)
                        raise Exception('#ERROR: not a file/dir??')
                else: # assume aqml.cheminfo.core.atoms class
                    #if obj.__class__.__name__ == 'atoms':
                    #    mols.update([obj])
                    #else:
                    raise Exception('Not a class or aqml.cheminfo.core.atoms?')
            mols = cc.molecules(fs, property_names)
        else: # assume aqml.cheminfo.core.atoms class
            if objs.__class__.__name__ == 'molecules':
                mols = objs
            else:
                raise Exception('Not a class or aqml.cheminfo.core.molecules?')

        pns = property_names
        pn1 = pns[0]

        # attach strains: an array of T/F
        imcs = []
        strains = []
        for i in range(mols.nm):
            rawm = coo.ConnMol(mols[i])
            strains.append(rawm.strained)
            imcs.append( rawm.is_mcplx )
        mols.strains = np.array(strains)
        mols.imcs = np.array(imcs, dtype=np.bool)

        if iae:
            for pn1 in pns:
                is_energetic_prop = T
                if is_energetic_prop:
                    mols.get_atomization_energies(pn1,prog=prog)
            #mols.ys = mols.props[pn1]

        if len(property_names) == 2 and Delta:
            pn2 = property_names
            ys1, ys2 = mols.props[pn2], mols.props[pn1]
            ys = mols.props[pn2] - mols.props[pn1]
        else:
            ys = mols.props[pn1] #np.array([ mols.props[p] for p in pns ]).T
        #print('shape of ys = ', ys.shape)
        uc = io2.Units()
        const = dict(zip([ 'h', 'ev', 'kcal'], \
                         [ uc.h2kc, uc.e2kc, 1.0]))
        mols.ys = ys * const[unit.lower()]
        self.ys = ys

        #rcut = 4.8 #2.7 #4.8
        coeffs = [1.0]
        local = T
        self.xparam={'local':local, 'kernel':'g', 'rcut':rcut, 'reuses':[F,F,F], \
                     'saves':[F,F,F], 'coeffs':coeffs, 'ws':[1.,1.,1.]}
        for k in xparam:
            self.xparam[k] = xparam[k]

        self.mols = mols
        self.fitmorse = fitmorse
        self.idxsr = idxsr
        self.no_strain = no_strain # for morse param fit
        self.check_boundary =  check_boundary
        self.use_morse_db =    use_morse_db


    def init_param(self, _midxs, i_use_molcplx_for_dressed_ae=T):

        #print('_midxs=',_midxs)
        _mols = self.mols[_midxs]
        _mols.ys = self.mols.ys[_midxs]
        if i_use_molcplx_for_dressed_ae:
            _mols.imcs = np.array([F]*len(_midxs), dtype=np.bool)
        else:
            _mols.imcs = self.mols.imcs[_midxs]
        #print('imcs=', _mols.imcs)
        self._mols = _mols
        _ims = np.arange(_mols.nm-1)
        _strains = self.mols.strains[_midxs][:-1] # exclude the last test molecules

        if self.fitmorse:
            idxsr = [] # mol idx to be removed for fitting morse
            if self.no_strain:
                idxsr = _ims[_strains] if self.idxsr is None else self.idxsr
            # note that the `ys below is actually `dys
            bts, fparam, ys = get_param(_mols, n2=self.n2, idxsr=idxsr, icn=T, \
                              iconn=T, iconj=F, use_morse_db=self.use_morse_db, \
                              check_boundary=self.check_boundary)
            self.bts = bts
            self.fparam = fparam
            self._mols.ys = ys


    def test_target(self, namax=7, cab=F, icg=F, izeff=F, fmap=None, \
                    mib=0, iaml=T, wd='./', llambdas=[1e-4], usebl=T,\
                    n1s=[], n2=1, exclude=[]):
        """ one or multiple target molecules
        cab: calculate similarity (distance) between dissimilar atom
              types (type=element). Must be either T or F
        mib: target molecule idx to begin with (when iaml=T and fmap is not None)
        icg: use connectivity graph (including vdw bonds, conj bonds)
             By conj bonds, we mean that all conjugated atoms are
             `connected', to some extent. I.e., suppose atoms ias=[1,2,3,4]
             are conjugated, then cg[a,b] = 1 for any `a and `b in `ias
        izeff: use effective nuclear charge (Z*) instead of Z as prefactor
               in SLATM 2- and 3-body
        exclude: mol idxs to be excluded for training
        """
        self.n2 = n2
        nm = self.mols.nm
        maps = None
        loops = 1
        if iaml:
            _n1s = []
            if n2==1:
                maps = np.array([np.arange(nm-1)], dtype=int)
            elif n2>1: # read map file
                if fmap:
                    print(' ** found `fmap file: %s, use query-specific amons as training set'%fmap)
                    _dt = dd.io.load(fmap)
                    maps = _dt['maps']
                    loops = n2
                else:
                    print(' ** no `fmap file specified, use all mols with idx ranging from 0 to nm-n2 as a single training set')
        else:
            assert len(n1s)>0, '#ERROR: when iaml=F, please specify `n1s'
            _n1s = n1s

        coeffs = self.xparam['coeffs']
        #usebl = F if self.fitmorse else T

        n1so = []; maes = {}; rmses = {}; dys = {}; errsmax = {}; fparams = {}
        for i in range(loops):
            print('\n\n\n')
            #print('   i,Loops=', i,loops)
            if i < mib: continue

            # vital !!
            # otherwise, `_n1s will be used when calling obj.get_idx() later
            #       obj.get_idx(iaml=iaml, idx=idx, n1s=_n1s, n2=n2i, namax=namax)
            if iaml: _n1s = []

            if maps is None:
                n1 = nm - n2
                midxs = np.arange(nm)
                idx = -n2
                n2i = n2
            else:
                _midxs = maps[i]
                assert np.max(_midxs) <= nm-n2-1
                midxs = np.concatenate( (_midxs[_midxs>-1], [nm-n2+i]) )
                idx = -1
                n1 = len(midxs)-1
                n2i = None
                print('i=',i+1, 'n1=',n1)

            self.init_param(midxs)
            if self.fitmorse: fparams[i] = self.fparam

            #ys = np.array([ mols.props[k] for k in mols.props.keys() ]).T
            #print('ys=', self._mols.ys.shape)
            obj = None
            obj = krr.krr(self._mols.ys)
            #print('nzs = ', self._mols.nzs.shape, 'zsu=',np.unique(self._mols.zs))
            obj.init_m(self._mols)

            if iaml:
                namax = np.max(self._mols.nsheav[:-n2])
                print(' found `namx=', namax)
            #print('_n1s=',_n1s)
            obj.get_idx(iaml=iaml, idx=idx, n1s=_n1s, n2=n2i, namax=namax)
            mko = sl.slatm(obj,ck=True, wd=wd, param=self.xparam, cab=cab, icg=icg, \
                         izeff=izeff, itarget=self.itarget, saveblk=self.saveblk)
            ks1,ks2 = mko.mk1,mko.mk2
            for l,llambda in enumerate(llambdas): #[1e-2, 1e-4, 1e-8]: #, 1e-10]:
              for i,c in enumerate(coeffs):
                print('--llambda,c = ', llambda,c)
                obj.run([ks1[i],ks2[i]], usek1o=F, llambda=llambda, usebl=usebl, \
                        exclude=exclude)
                key = '%d,%d'%(l,i)
                if key in maes:
                    maes[key].append( obj.maes )
                    rmses[key].append( obj.rmses )
                    dys[key].append( obj.dys )
                    errsmax[key].append( obj.errsmax )
                else:
                    maes[key] = [obj.maes]
                    rmses[key] = [obj.rmses]
                    dys[key] = [obj.dys]
                    errsmax[key] = [obj.errsmax]
            n1so.append( obj.n1so )
        self.maes = maes
        self.rmses = rmses
        self.dys = dys
        self.n1s = n1so
        self.errsmax = errsmax
        if self.fitmorse: self.fparams = fparams

