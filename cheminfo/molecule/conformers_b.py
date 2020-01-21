#!/usr/bin/env python

import io2, os, sys
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.ML.Cluster import Butina
import cheminfo.openbabel.obabel as cob
import scipy.spatial.distance as ssd
import numpy as np
from io2.gaussian_reader import GaussianReader as GR0
#import io2.xyz as ix
from cheminfo.molecule.molecule import *
from cheminfo.molecule.nbody import NBody
from cheminfo.rw.xyz import write_xyz
from cheminfo.core import *
#import cheminfo.molecule.amon_f as cma
import cheminfo.oechem.amon as coa
import indigo
import tempfile as tpf
import cheminfo.rdkit.core as crk
import deepdish as dd
try:
    import representation.x as sl
except:
    pass

h2kc = io2.Units().h2kc
T, F = True, False
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

_hyb = { Chem.rdchem.HybridizationType.SP3: 3, \
            Chem.rdchem.HybridizationType.SP2: 2, \
            Chem.rdchem.HybridizationType.SP: 1, \
            Chem.rdchem.HybridizationType.UNSPECIFIED: 0}

bt2bo = { Chem.BondType.SINGLE:1.0,
          Chem.BondType.DOUBLE:2.0,
          Chem.BondType.TRIPLE:3.0,
          Chem.BondType.AROMATIC:1.5,
          Chem.BondType.UNSPECIFIED:0.0}

bo2bt = { '1.0': Chem.BondType.SINGLE,
          '2.0': Chem.BondType.DOUBLE,
          '3.0': Chem.BondType.TRIPLE,
          '1.5': Chem.BondType.AROMATIC,
          '0.0': Chem.BondType.UNSPECIFIED}

class _atoms(object):
    """ `atoms object from file formats other than xyz"""
    def __init__(self, f):
        import ase.io as aio
        m = aio.read(f)
        self.zs = m.numbers
        self.coords = m.positions
        self.na = len(self.zs)


def get_torsions(m):
    """ get torsion types & dihedral angles
    returns a dictionary {'1-6-6-8':[100.], } """
    mr = RawMol(m.zs, m.coords)
    mr.connect()
    g = mr.g
    obj = NBody(m.zs, m.coords, g, unit='degree', iheav=T, icn=T)
    obj.get_dihedral_angles()
    dic = obj.mbs4
    return dic


class CM(object):
    """
    coulomb matrix object
    """
    def __init__(self, atoms, param={'M':'cml1','rp':1.,'wz':T,'sort':T}):
        self.param = param
        self.atoms = atoms
        self.cml1 = T if param['M'] in ['cml1'] else F

    def generate_coulomb_matrix(self):
        """ Coulomb matrix

        sorted CM has serious limitation when used to identify unique conformers.
        E.g., for CH3-CH3 molecule, as the L1 norm of all H-containing columns are
        the same, so shuffling such columns leads to different CM, though the molecule
        remains unchanged

        The limitation can be alleviated through the use of L1 norm of each column!!
        """
        atoms = self.atoms
        na = atoms.na
        mat = np.zeros((na,na))
        _ds = ssd.squareform( ssd.pdist(atoms.coords) )
        dsp = _ds**self.param['rp']
        np.fill_diagonal(dsp, 1.0)
        zs = atoms.zs
        _X, _Y = np.meshgrid(zs,zs)
        if self.param['wz']:
            mat = _X*_Y/dsp
            diag = -np.array(zs)**2.4
        else:
            mat = 1/dsp
            diag = np.zeros(na)
        np.fill_diagonal(mat,  diag)
        if self.param['sort']:
            L1s = np.array([ np.sum(np.abs(mat[i])) for i in range(na) ])
            ias = np.argsort(L1s)
            if self.cml1:
                x = L1s[ias]
            else:
                #x = mat[ias][:,ias]
                x = np.ravel(mat[ias,:][:,ias])
        else:
            #x = mat
            x = np.ravel(mat)
        #print 'x = ', x
        return x


def cdist(objs, objs2=None, param={}):
    _param = {'M':'cml1','rp':1.0,'sort':T,'wz':F,'metric':'cityblock'}
    for key in list(param.keys()):
        if key in list(_param.keys()):
            if param[key] != _param[key]:
                _param[key] = param[key]
    M = _param['M']
    _xs1 = []
    nc = len(objs)
    for obj in objs:
        if M in ['cm','cml1']:
            _xobj = CM(obj,_param)
            xi = _xobj.generate_coulomb_matrix()#; print '              xi = ', xi
            _xs1.append( xi )
        else:
            raise '#ERROR: unknown `M'
    xs1 = np.array(_xs1)
    metric = _param['metric'] #
    if objs2 is not None:
        _xs2 = []
        nc = len(objs2)
        for obj in objs2:
            if M in ['cm','cml1']:
                _xobj = CM(obj,_param)
                xi = _xobj.generate_coulomb_matrix()#; print '              xi = ', xi
                _xs2.append( xi )
            else:
                raise '#ERROR: unknown `M'
        xs2 = np.array(_xs2)
        ds = ssd.cdist(xs1,xs2,metric) #'cityblock')
    else:
        ds = ssd.squareform( ssd.pdist(xs1,metric) ) #'cityblock') )
    return ds


def update_h5(fd):
    if not os.path.exists(fd+'/i-raw'):
        print(' ** Error: %s/i-raw not exists, h5 mapping file cannot be updated'%fd)
        raise
    fs = io2.cmdout('ls %s/*.sdf'%fd)
    fs0 = io2.cmdout("ls -1 %s/i-raw/*.com"%fd)
    nf0 = len(fs0)
    assert len(fs) > 0, '#ERROR: no SDF file exists'
    _remap = np.arange(nf0)
    set_r = list(set(fs0).difference(fs))
    for fi in set_r:
        _remap[ fs0.index(fi) ] = -1

    idx = []
    idxr = []; icount = 0
    _objsc = []
    for i,f in enumerate(fs):
        i2 = fs0.index(f)
        if smi != _amons[im]: # graph may have changed after optg
            _remap[i2] = -1
        else:
            _remap[i2] = i
        if im not in idx:
            idx.append(im)
            idxr.append(icount)
            icount += 1
        else:
            idxr.append( idx.index(im) )

    _remap = np.concatenate((_remap,[-1])) # _remap[-1] = -1
    return h5u


class AmonDict(object):
    """
    stores all amons (a dictionary) for a dataset
    and look up the dictionary for relevant AMONs
    when a query molecule is specified

    Note that we assume AMONs are associated with
    xyz files of the same format as in class `OptedMols.
    In particular, the second line has to read like:
        'E=%.5fHa,rawfile=%s' (no ' at the beginning & end)
    """

    def __init__(self, fd, fd2=None, fcanr=None, h5f=None, \
                 imb=False, props=['E'], ctk='oechem'):
        """
        fcanr: file containing cononical SMILES for reference (from which conformers
                were generated)
        """
        self.ctk = ctk # cheminfomatic toolkit to be used, rdkit, indigo, openbabael or oechem

        if fd[-1] == '/': fd = fd[:-1]
        self.fda = fd

        fs = io2.cmdout('ls %s/frag*.sdf'%fd)
        nf = len(fs)
        ns = [ len(f) for f in fs ]
        assert len(np.unique(ns)) == 1, '#ERROR: filename length may differ'
        assert nf > 0, '#ERROR: no *.sdf found'
        objs = []
        s = []
        s1 = [] # unique smiles strings
        _ncs = np.zeros(nf)
        imc = 0
        for f in fs:
            obj = crk.RDMol(f)
            objs.append(obj)
            smi = obj.prop['smiles_%s'%ctk]
            s.append(smi)
            if smi not in s1:
                _ncs[imc] = 1
                imc += 1
                s1.append(smi)
            else:
                _ncs[imc] += 1

        if fcanr is not None:
            _amons = [ si.strip() for si in open(fcanr).readlines() ]
            if _namon == len(s1):
                print("#ERROR: #amons from sdf's inconsistent with that in %s"%fcanr)
                print("        This means that some amons (graph) may have been")
                print("        hiscarded due to dissociation")
                raise
        else:
            _amons = s1
        _namon = len(_amons)
        self.amons = _amons

        nm = len(s1)
        ncs = _ncs[:nm]
        cidxs = np.arange(nf).astype(int)
        c2amap = cidxs.copy()
        ics2 = np.array(np.cumsum(ncs),dtype=int)
        ics1 = np.array(np.concatenate(([0],ics2[:-1])),dtype=int)
        for im in range(nm):
            ib, ie = ics1[im], ics2[im]
            c2amap[ib:ie] = im
        a2cmap = [ list(cidxs[ics1[iamon]:ics2[iamon]]) for iamon in range(nm) ]

        cmaps = [F]
        # if provided h5 file, read conformer IDs from it
        if h5f is not None:
            dt = dd.io.load(h5f)
            _cmaps = dt['maps']
            if np.max(maps) != nf-1:
                print(' * Error: #sdf files .ne. max(maps)+1!!')
                raise
            cmaps = [T,_cmaps]

        # add new conformers from another folder
        ic2 = nf
        if fd2 is not None:
            fs2 = io2.cmdout('ls %s/frag*sdf'%fd2)
            nf2 = len(fs2)
            s2 = []
            for f2 in fs2:
                obj2 = crk.RDMol(f2)
                smi = obj2.property['smiles_%s'%self.ctk]
                s2.append(smi)
                if smi in _amons:
                    im2 = _amons.index(smi)
                    c2amap[ic2] = im2
                    a2cmap[im2] += [ic2]
                    ic2 += 1
                    objs.append(obj)
            assert len(s2) == nf2
            if cmaps[0]:
                # now update maps in *.h5 file
                _maps = cmaps[1]
                ncmax, nt = _maps.shape
                icsmax = np.arange(ncmax).astype(int)
                _maps2 = []
                ncs_added = []
                for j in range(nt):
                    fil = (_maps[j] > -1 )
                    ims_j = c2amap[ _maps[j][fil] ]
                    ics_j = []
                    for k in ims_j:
                        t2 = np.array(a2cmap[k],dtype=int)
                        ics_j += list(t2[t2>=nf])
                    ncs_added.append(len(ics_j))
                    _maps2.append( list(icsmax[fil])+ics_j )
                nadd = max(ncs_added)
                print(' ## for some query molecule, %d conformers at most are added as new amons'%nadd)
                maps2 = np.zeros((nt,ncmax+nadd),dtype=int)
                for j in range(nt):
                    ncj = len(_maps2[j])
                    maps2[:ncj] = _maps2[j]
                cmaps = [T,maps2]

        self.c2amap = c2amap # (amon/conformer) to (molecule graph) map
        self.a2cmap = a2cmap
        self.cmaps = cmaps

        # if `fcanr is not None, check dissociated molecules
        #if fcanr is not None:
        #    _remap = np.arange(nf).astype(int)
        #    for i,f in enumerate(fs):
        #        im = int(f.split('frag_')[1][:-4].split('_')[0])-1 # molecule index (graph! not conformer idx)
        #        smi = s[i]
        #        if smi != _amons[im]: # graph may have changed after optg
        #            _remap[i] = -1

        self.nmt = nm
        self.nct = ic2

        objsc = []; ys = []
        zs=[]; nsheav=[]; nas=[]; coords=[]
        for i,f in enumerate(fs):
            objc = objs[i]
            zs += list(objc.zs)
            nas.append(objc.na)
            coords += list(objc.coords)
            nsheav.append(objc.nheav)
            if imb: # calc many-body terms
                objc.get_angles(wH=F,key='ia')
                objc.get_dihedral_angles(wH=F,key='ia')
            objsc.append(objc)
            ys.append( [ objc.prop[key] for key in props ] )
        self.nas = np.array(nas,np.int)
        self.zs = np.array(zs,np.int)
        self.coords = np.array(coords)
        self.nsheav = np.array(nsheav,np.int)
        self.objsc = objsc

        self.ics1 = ics1
        self.ics2 = ics2
        self.props = props
        self.ys = np.array(ys)


def get_angles34(_dic, iass):
    """ turn absolute idx in query to absolute idx in subm
    for 3- and 4-body terms, i.e., angular and torsional terms
    E.g., sumb='C=CC', q='C=CCC=C', iass=[ (0,1,2), (4,3,2) ]
          _dic = { (0,1,2):a1, (1,2,3):a2, (2,3,4):a3 } (3-body terms)
    """
    _keys = list(_dic.keys())
    types = []; vals = []
    for _ias in iass:
        _idxr = np.arange( len(_ias) )
        _amap = dict(list(zip(_ias,_idxr))) # turning abs idx in q to relative idx in subm
        _set = set(_ias)
        dic2 = {}
        for _key in _keys: # _keys covers all 3- (4-) body terms, some of them may not be present in `_set
            if set(_key) <= _set:
                key2 = tuple([ _amap[_] for _ in _key ])
                if key2[0] > key2[-1]:
                    key2 = tuple([ _amap[_] for _ in _key[::-1] ])
                dic2[key2] = _dic[_key]
        newkeys = list(dic2.keys())
        newkeys.sort()
        types.append( newkeys )
        vals.append( np.array([dic2[newkey] for newkey in newkeys],dtype=int) )
    return types, vals


def check_angles3(atypes, ias, angs_a, angs_q, thresh, idev=F):
    adic = {} # amon
    qdic = {} # query
    mbs = []
    for i,iasi in enumerate(ias):
        mbi = tuple( [ atypes[j] for j in iasi ] )
        if mbi[0] > mbi[2]: mbi = mbi[::-1]
        if (mbi in mbs):
            adic[mbi] += [angs_a[i]]
            qdic[mbi] += [angs_q[i]]
        else:
            mbs.append( mbi )
            adic[mbi] = [angs_a[i]]
            qdic[mbi] = [angs_q[i]]
    diffs = []
    for key in adic.keys():
        avals = adic[key]; avals.sort()
        qvals = qdic[key]; qvals.sort()
        diffs += list(np.array(avals)-np.array(qvals))
    iok3 = T
    if len(diffs) == 0:
        ots = [iok3, 0. ] if idev else iok3
    else:
        diffs = np.array(diffs); #print ' angs diff = ', diffs
        if np.any(np.abs(diffs)>thresh): # (diffs>thresh).sum() - (diffs<-thresh).sum() >= 2:
            iok3 = F
        ots = [iok3, np.mean(np.abs(diffs))] if idev else iok3
    return ots

def check_angles4(atypes, ias, angs_a, angs_q, thresh, idev=F):
    adic = {} # amon
    qdic = {} # query
    mbs = []
    for i,iasi in enumerate(ias):
        mbi = tuple( [ atypes[j] for j in iasi ] )
        if mbi[1] > mbi[2]:
            mbi = mbi[::-1]
        else:
            if mbi[0] > mbi[3]:
                mbi = mbi[::-1]
        if (mbi in mbs):
            adic[mbi] += [angs_a[i]]
            qdic[mbi] += [angs_q[i]]
        else:
            mbs.append(mbi)
            adic[mbi] = [angs_a[i]]
            qdic[mbi] = [angs_q[i]]
    diffs = []
    for key in adic.keys():
        avals = adic[key]; avals.sort()
        qvals = qdic[key]; qvals.sort()
        diffs += list(np.array(avals)-np.array(qvals))
    iok4 = T
    if len(diffs) == 0:
        ots = [iok4, 0.0 ] if idev else iok4
    else:
        diffs = np.array(diffs); #print ' dangs diffs = ', diffs
        if np.any(np.abs(diffs)>thresh): #if (diffs>thresh).sum() - (diffs<-thresh).sum() >= 2:
            iok4 = F
        ots = [iok4, np.mean(np.abs(diffs))] if idev else iok4
    return ots

class AmonQ(object):
    """ Get amons for query mol """
    def __init__(self, dt):
        assert type(dt) is AmonDict
        self.props = dt.props
        self.zs_a = dt.zs
        self.nas_a = dt.nas
        self.coords_a = dt.coords
        self.nsheav_a = dt.nsheav
        self.ias2 = np.cumsum(dt.nas)
        self.ias1 = np.array([0]+list(self.ias2[:-1]))
        self.amons = dt.amons
        self.objsc = dt.objsc
        self.nct = dt.nct
        self.ys_a = dt.ys
        self.c2amap = dt.c2amap
        self.a2cmap = dt.a2cmap
        self.fs = np.array(dt.fs)
        self.nmt = dt.nmt
        self.fmt = '%%0%dd'%(len(str(dt.nmt)))
        self.fda = dt.fda
        self.ctk = dt.ctk
        self.cmaps = dt.cmaps

    def query(self, fq, idQ=None, k=7):
        if self.cmaps[0]:
            #assert len(fsq) > 0, '#ERROR: `fsq not specified!'
            assert idQ is not None
            self.idQ = idQ #fsq.index(fq)
        self.fq = fq
        #zs, coords, ydic = read_xyz_simple(f,opt='z')
        objq = crk.RDMol(fq)
        zs, coords, ydic = objq.zs, objq.coords, objq.prop
        smi = ydic['smiles_%s'%self.ctk]
        _ys = []
        for key in self.props:
            _yi = ydic[key] if key in ydic.keys() else np.nan
            _ys.append(_yi)
        ys_q = np.array([_ys]) #[ [ydic[key] for key in self.props ] ])
        nheav = (np.array(zs)>1).sum()
        #ao = cma.amon(smi, k) # amon object. Note that can's are of indigo format
        #assert ao.iok
        #amons_q, ns_q, ats_q = ao.get_amons(iao=T) # idxs of atoms (in query mol) as output as well
        reduce_namons=T
        ao = coa.ParentMols([smi],reduce_namons,wg=F,imap=T,k=7)
        #amons_q = []
        #for cci in ao.cans:
        #    mobj = indigo.Indigo()
        #    m2 = mobj.loadMolecule(cci)
        #    amons_q.append( m2.canonicalSmiles() )
        #self.ats_q = ao.atsa
        self.amons_q = ao.cans #amons_q ## now stick to can of oechem format
        #self.ns_q = ao.nsa
        self.ys_q = ys_q
        self.zs_q = np.array(zs,np.int)
        self.nsheav_q = [nheav]
        self.coords_q = np.array(coords)
        # tor
        #self.x_q = get_torsions( atoms(zs,coords) ) #; sys.exit()
        objq.get_atypes()
        objq.get_angles(wH=F,key='ia')
        objq.get_dihedral_angles(wH=F,key='ia')
        self.objq = objq
        # slatm
        #xobj = sl.slatm([len(zs)],zs,coords)
        #xobj.get_x(rb=F, mbtypes=None, param={'racut':4.8})
        #self.x_q = xobj.xsa
        #self.mbtypes = xobj.mbtypes

    def amons_filter_cm(self,smile,csi,param):
        """ check if the given conformer `conf is close to some
        local structure in the query mol by checking d(cmi,cmj)!!
        """
        thresh = param['thresh']
        M = param['M']
        nc = len(csi)
        #if nc==1: return [0]
        ics = np.arange(nc)
        atypes, patt = crk.smi2patt(smile)
        _ = Chem.MolFromSmarts(patt)
        matches_q = np.array( self.objq.m.GetSubstructMatches(_), dtype=int )
        nq = len(matches_q)
        # heavy atoms only
        #print matches_q, self.zs_q, self.coords_q
        csr = [ atoms(self.zs_q[_ias],self.coords_q[_ias]) for _ias in matches_q ]
        cs = []
        for ci in csi:
            zsi, coordsi = ci.zs,ci.coords
            cs.append( atoms(zsi[zsi>1],coordsi[zsi>1]) )
        ds = cdist(cs, csr, param) #[-1]; #print '  ds = ', ds[-1,:-1]
        ds1 = cdist(cs, None, param) # between conformers
        np.fill_diagonal(ds1, 99999)
        #return ics[ np.any(ds<=param['thresh'][0],axis=1) ]
        icsc = [] # ics that are chosen
        dsmin = np.min(ds,axis=0)
        for i in range(nq):
            if nc==1: icsc += [0]
            dsi = ds[:,i] + np.array([ds[:,i]]).T; #print '--dsi=',dsi
            _a,_b = np.where(np.triu(dsi)>np.triu(ds1))
            a,b = list(_a),list(_b); #print '--a,b=',a,b
            n = len(a); ips = np.arange(n); #print '--n=',n
            if n>0:
                dsp = [ dsi[a[j],b[j]] for j in ips ]; #print '--ips,dsp=', ips,dsp
                ipc = ips[dsp==np.min(dsp)][0]
                icsc += [a[ipc],b[ipc]]
            else:
                icsc += list( ics[ds[:,i]==np.min(ds[:,i])] ) # for each subm in query, choose one amon
        #print ' -- nc, nq, icsc = ', nc, nq,icsc
        return np.unique( icsc )

    def amons_filter_aslatm(self,conf):
        """ check if all local atomic env in conf is close to
        some in the query (based on aSLATM)
        """
        param = self.param
        iok = T
        xobj = sl.slatm([len(conf.zs)],conf.zs, conf.coords)
        xobj.get_x(rb=F, mbtypes=self.mbtypes, param={'racut':4.8})
        xc = xobj.xsa
        ds = qd.l2_distance(xc,self.x_q)# xc)
        return np.any(ds <= param['thresh'][0])

    def amons_filter_tor(self,smile,cs,param): #=[36.,45.]):
        """ filter the given conformers (`cs) using the criteria
        of angle & torsional angles
        so as to find the ones that are valid amons of query

        Attention:
        =====================================================
        1) angs & torsions found in amons and query may
           differ, e.g., amon=CCCC, query=C1CCC1C
           Furthermore, the number of matches from query is always
           larger than that from amons, so we must only compare the
           matches from amons!!
        2) torsions of the same type have to be sorted by dihedral angs
           E.g., for a mol
                             C5       H
                  O1        / \       O9
                   \\      /   \     /
                     C2---C4    C7==N8
                    /      \   /
                   C3       \ /
                             C6
           Given a pattern like [O;X2]~[N;X2]~[C;X3]1~[C;X4]~[CX3]~[C;X4]1
           (converted from SMILES ON=C1CCC1), there are two ADJACENT torsions:
           O9-N8-C7-C5 and O9-N8-C7-C6, for which the dang's differ
           by ~180 degrees! Thus we have to sort such dang's (must be adjacent
           ones that share exactly the same env: TypeA1-TypeA2-TypeA3-TypeA4) and
           then compare! Otherwise, we may finally reject this patter as a valid
           AMON (in reality they must be chosen!)
        """

        #print ' smiles=',smile
        thresh = param['thresh']
        M = param['M']
        nc = len(cs)
        ics = np.arange(nc)
        atypes, patt = crk.smi2patt(smile)
        _ = Chem.MolFromSmarts(patt)
        matches_q = self.objq.m.GetSubstructMatches(_)
        #print ' matches_q = ', matches_q
        nmatch = len(matches_q)
        types3_q, angs_q = get_angles34(self.objq.angs, matches_q)
        types4_q, dangs_q = get_angles34(self.objq.dangs, matches_q)
        icsc = [] # chosen ics
        for i,objc in enumerate(cs):
            # Now check if the i-th conformer match any of the local subm
            # in the query mol
            matches_c = objc.m.GetSubstructMatches(_)
            types3_c, angs_c = get_angles34(objc.angs, matches_c)
            types4_c, dangs_c = get_angles34(objc.dangs, matches_c)
            #dic3_c = dict(zip(types3_c[0],angs_c[0]))
            #dic4_c = dict(zip(types4_c[0],dangs_c[0]))
            iok = False
            for j in range(nmatch):
                _a = angs_q[j]
                dic3_q = dict(list(zip(types3_q[j],_a)))
                _d = dangs_q[j]
                dic4_q = dict(list(zip(types4_q[j],_d)))
                # Note that {keys in typesN_c} <= {keys in typesN_q}
                # '==' holds when subgraph isomorphism is met; otherwise,
                # '<' holds when say, subm='CCCC', q='C1CCC1'
                if len(types3_q[j])!=len(types3_c[0]) or len(types4_q[j])!=len(types4_c[0]):
                    # This is possible. E.g., subm='CNCC=N', q='CC(C)N1CC(=NO)C1' (see below)
                    #
                    #            C4         C2
                    #           /   \      /
                    #      C7=C5     N3--C1
                    #     /     \   /      \
                    #    O8       C6        C0
                    #
                    # Two matched subm are C1-N5-C6-C5=C7 and C4-N4-C6-C5=C7, the latter is
                    # actually a ring and the corresponding len(types3_q) is different to CNCC=N
                    # thus, we have to skip such case
                    continue
                _angs_q = np.array([ dic3_q[t3] for t3 in types3_c[0] ])
                _dangs_q = np.array([ dic4_q[t4] for t4 in types4_c[0] ])
                #dfs3 = angs_c[0] - _angs_q
                #print ' types3_c[0] = ', types3_c[0]
                #print ' angs_c[0] = ', angs_c[0]
                iok3 = check_angles3(atypes, types3_c[0], angs_c[0], _angs_q, thresh[0])
                #dfs4 = dangs_c[0] - _dangs_q
                iok4 = check_angles4(atypes, types4_c[0], dangs_c[0], _dangs_q, thresh[1])
                #iok3 = np.all(np.abs(dfs3)<=thresh[0])
                #iok4 = np.all(np.abs(dfs4)<=thresh[1])
                #if not iok3: print ' ang mismatch', list(angs_c[0]),list(_angs_q)
                #if not iok4: print ' tor mismatch', list(dangs_c[0]),list(_dangs_q)
                if iok3 and iok4:
                    iok = True
                    break
            if not iok: continue
            icsc.append(i)
        return icsc

    def get_cids(self,param={'M':'cml1','rp':1.0,'wz':F,'thresh':[0.1]},diagnose=F):
        """ get amon conformer idxs
        Two senarios are allowed
        1) conformer ids provided by a h5 file, i.e., self.cmaps[0]=[True,`maps_read_from_h5f]
        2) select conformers on-the-fly through a similarity match algo
        """
        self.param = param
        _aids = []; _cids = []; _ias = []
        # first get all conformers
        aids0 = []; cids0 = []; #ats0 = []
        #print 'amons_q = ', set(self.amons_q)
        #print 'amons = ', self.amons

        use_dic = T
        if self.cmaps[0]: # use cids from maps.h5
            cmaps = self.cmaps[1]
            use_dic = F
            nm = len(cmaps)
            cmap = cmaps[self.idQ]; n1 = len(cmap)
            ics0 = np.arange(n1)
            if (cmap==0).sum()>1: # old algo, `maps array padded with 0
                raise '#plz regenrate amons using new algo, padded with -1 instead of 0 in `maps.h5'
            cids0 = list(cmap[cmap>-1])
            aids0 = np.unique([ self.c2amap[ic] for ic in cids0 ])
        else: # select amons from a dict
            for iaq, aq in enumerate(self.amons_q):
                iad = self.amons.index(aq) # idx of amon in the AmonDict
                aids0.append( iad )
                cids0 += self.a2cmap[iad]

        if diagnose:
            _fd = 'indiv/'+self.fq.split('frag')[1][:-4].split('_')[1]
            _fd2 = _fd + '/neglected'
            if not os.path.exists(_fd): os.system('mkdir -p %s'%_fd)
        if param['thresh'][0] > 0: # negative value of `thresh indicates using all amons
            M = param['M']
            amons_filter = {'tor': self.amons_filter_tor, 'cm': self.amons_filter_cm,
                            'cml1': self.amons_filter_cm}[M]
            if M in ['tor','torsion']:
                assert len(param['thresh']) == 2, '#ERROR: len(thresh)!=2'
            for _aid in aids0:
                if use_dic: # use amons dictionary
                    cids_t = self.a2cmap[_aid]
                else: # use ids from `maps.h5
                    cids_t = list( set(self.a2cmap[_aid]).intersection(cids0) )
                cids = np.array(cids_t,dtype=int)
                #print ' -- cids = ', cids_t
                _ = cids[0]
                zsi = self.zs_a[ self.ias1[_]:self.ias2[_] ]
                if len(zsi[zsi>1]) >= 4: #4:
                    smi = self.amons[_aid]
                    csi = [self.objsc[ic] for ic in cids] # `cs: conformers !Todo
                    #print "aid=", _aid
                    #print "fs = ['%s']"%( "','".join( self.fs[cids] ) )
                    ccidsr = amons_filter(smi,csi,param) #['thresh'])
                    if len(ccidsr) == 0:
                        #print ' [no conformer chosen] f0=%s'%(self.fs[cids[0]])
                        fs_noconf = self.fs[cids]
                        if diagnose and len(fs_noconf) > 0:
                            if not os.path.exists(_fd2): os.system('mkdir -p %s'%_fd2)
                            #print ' files that are totally ignored are now in %s'%_fd2
                            os.system('cp %s %s'%(' '.join(fs_noconf), _fd2))
                        continue
                    else:
                        fs_conf = self.fs[cids[ccidsr]]
                        if diagnose and len(fs_conf) > 0:
                            os.system('cp %s %s'%(' '.join(fs_conf), _fd))
                else:
                    ccidsr = list(range(len(cids)))
                    fs_conf = self.fs[cids[ccidsr]]
                    if diagnose and len(fs_conf) > 0:
                        os.system('cp %s %s'%(' '.join(fs_conf), _fd))
                if _aid not in _aids: _aids.append(_aid)
                ccids = cids[ccidsr]
                _cids += list(ccids)
                for _cid in ccids:
                    # now add all atoms in these conformers to `_ias
                    _ias += list(range(self.ias1[_cid],self.ias2[_cid]))
        else:
            _cids = cids0
            _aids = aids0
            _ias = []
            for cid in cids0:
                _iasc = np.arange(self.ias1[cid],self.ias2[cid])
                _ias += list(_iasc)
        if not set(_aids)<=set(aids0):
            print(' #ERROR: set(_aids)<=set(aids0) does not hold true!!')
        aids_r = np.setdiff1d(aids0,_aids)
        #if diagnose:
        #    _fd = 'indiv/'+self.fq.split('frag')[1][:-4].split('_')[1]
        #    if not os.path.exists(_fd): os.system('mkdir -p %s'%_fd)
        #    sf = ' '.join([ self.fda+'/frag_'+self.fmt%(ir+1)+'_*.sdf' for ir in aids0])
        #    os.system( 'cp %s %s'%(sf,_fd) )
        print('  #### amons ignored: ', ''.join(['%d: %s, '%(ir+1,self.amons[ir]) for ir in aids_r]))
        #print('  #### amons used: ', ''.join(['%d: %s, '%(ir+1,self.amons[ir]) for ir in _aids]))
        print('  #### total num of conformers selected: %d out of %d'%(len(_cids),len(cids0)))

        self.aids = _aids
        self.cids = _cids
        self.ias = _ias

    def wrapup(self):
        # now merge training amons & query molecule
        self.nas = np.array( list(self.nas_a[self.cids]) + [len(self.zs_q)], np.int)
        self.zs = np.concatenate((self.zs_a[self.ias], self.zs_q)).astype(np.int)
        self.coords = np.concatenate((self.coords_a[self.ias], self.coords_q))
        self.nsheav = np.array(list(self.nsheav_a[self.cids])+self.nsheav_q, np.int)
        _ys = np.concatenate( (self.ys_a[self.cids], self.ys_q), axis=0)
        if _ys.shape[1] == 1:
            self.ys = _ys[:,0]
        else:
            self.ys = _ys


def diagnose1(fa,fq):
    """ do diagnose for one amon and the query """
    oa = crk.RDMol(fa)
    oa.get_angles(wH=F,key='ia')
    oa.get_dihedral_angles(wH=F,key='ia')
    oq = crk.RDMol(fq)
    oq.get_angles(wH=F,key='ia')
    oq.get_dihedral_angles(wH=F,key='ia')
    smi = oa.prop['smiles_indigo']
    patt = crk.smi2patt(smi)
    print('patt=',patt)
    mp = Chem.MolFromSmarts(patt)
    iass_a = oa.m.GetSubstructMatches(mp)
    iass_q = oq.m.GetSubstructMatches(mp)

    types_a, vals_a = get_angles34(oa.dangs, iass_a)
    #dic_a = dict(zip(types_a[0],vals_a[0]))
    types_q, _vals_q = get_angles34(oq.dangs, iass_q)
    n = len(types_q)
    vals_q = []
    for i in range(n):
        types = types_q[i]; vals = _vals_q[i]; #print ' -- types = ', types
        dic_q = dict(list(zip(types,vals)))
        _vals = []
        for key in types_a[0]:
            _vals.append(dic_q[key])
        vals_q.append(_vals)
    print(' amon:')
    print(types_a[0])
    print(list(vals_a[0]))
    print(' query:')
    for i in range(n): print(vals_q[i])
    print(' Difference:')
    for i in range(n): print(list( vals_q[i]-vals_a[0]))

