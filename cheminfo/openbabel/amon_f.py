#!/usr/bin/env python

"""
Enumerate subgraphs & get amons
"""

import cheminfo.math as cim
import cheminfo.rw.pdb as crp
import cheminfo.graph as cg
import networkx as nx
from itertools import chain, product
import numpy as np
import os, re, copy, time
#from rdkit import Chem
import openbabel as ob
import pybel as pb
from cheminfo import *
import cheminfo.openbabel.obabel as cib
from cheminfo.rw.ctab import write_ctab

#Todo
# stereochemistry: e.g., "CC(=C)C(CC/C(=C\COC1=CC=CC=C1)/C)Br"
#                        "NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCN=C(N)N)"

#global dic_smiles
#dic_smiles = {6:'C', 7:'N', 8:'O', 14:'Si', 15:'P', 16:'S'}

chemical_symbols = ['X',  'H',  'He', 'Li', 'Be',
                    'B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si',
                    'P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se',
                    'Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',
                    'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                    'At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu',
                    'Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']


class RawMol(object):
    """
    molecule object with only `zs & `coords
    """
    def __init__(self, zs, coords):
        self.zs = zs
        self.coords = coords

    def generate_coulomb_matrix(self):
        """ Coulomb matrix"""
        na = len(self.zs)
        mat = np.zeros((na,na))
        ds = ssd.squareform( ssd.pdist(self.coords) )
        np.fill_diagonal(ds, 1.0)
        X, Y = np.meshgrid(self.zs, self.zs)
        mat = X*Y/ds
        np.fill_diagonal(mat, -np.array(self.zs)**2.4 )
        L1s = np.linalg.norm(mat, ord=1, axis=0)
        ias = np.argsort(L1s)
        self.cm = mat[ias,:][:,ias].ravel()




class Parameters(object):

    def __init__(self, wg, fixGeom, k, k2, ivdw, dminVDW, \
                 forcefield, thresh, do_ob_ff, idiff, iters):
        self.wg = wg
        self.fixGeom = fixGeom
        self.ff = forcefield
        self.k = k
        self.k2 = k2
        self.ivdw = ivdw
        self.dminVDW = dminVDW
#       self.threshDE = threshDE
        self.thresh = thresh
        self.do_ob_ff = do_ob_ff
        self.iters = iters
        self.idiff = idiff


def merge(Ms): #Mli1, Mli2):
    """merge two or more `ctab"""
    nas = []
    zs = []; coords = []; charges = []; boms = []
    for M in Ms:
        zs1, coords1, bom1, charges1 = M
        zs.append( zs1)
        na1 = len(zs1); nas.append(na1)
        coords.append( coords1)
        charges.append( charges1)
        boms.append(bom1)
    zs = np.concatenate( zs )
    coords = np.concatenate(coords, axis=0)
    charges = np.concatenate(charges)
    na = sum(nas); nm = len(nas)
    bom = np.zeros((na,na), np.int)
    ias2 = np.cumsum(nas)
    ias1 = np.array([0] + list(ias2[:-1]))
    for i in range(nm):
        ia1 = ias1[i]; ia2 = ias2[i]
        bom[ia1:ia2,ia1:ia2] = boms[i]
    return zs, coords, bom, charges


class Sets(object):

    def __init__(self, param):

        self.cans = [] #cans
        self.ms = [] #ms
        self.rms = [] #rms
        self.es = [] #es
        self.nhas = [] #nhas
        self.ms0 = [] #ms0
        self.maps = [] #maps
        self.cms = [] # coulomb matrix
        self.param = param

    def check_eigval(self):
        """ check if the new kernel (after adding one molecule) has
        some very small eigenvalue, i.e., if it's true, it means that
        there are very similar molecules to the newcomer, thus it won't
        be included as a new amon"""
        iok = True
        thresh = self.param.thresh

    def update(self, ir, can, Mli):
        """
        update `Sets

        var's
        ==============
        Mli  -- Molecule info represented as a list
                i.e., [zs, coords, bom, charges]
        """
        zs, coords, bom, charges = Mli
        rmol = RawMol(zs, coords)
        if self.param.idiff == 1: rmol.generate_coulomb_matrix()
        nha = (zs > 1).sum()
        self.ncan = len(self.cans)
        if can in self.cans:
            ican = self.cans.index( can )
            # for molecule with .LE. 3 heavy atoms, no conformers
            if (not self.param.fixGeom) and nha <= 3:
                # but u still need to tell if it belongs to the
                # `ir-th query molecule (so, the amon `m0 might
                # have appeared as an amon of another query molecule
                # considered previously.
                # Note that we use a 3-integer list for labeling the
                # generated amons, i.e., [ir,ican,iconfonmer].
                amon_idx = [ir, ican, 0]
                if amon_idx not in self.maps:
                    self.maps.append( amon_idx )
            else:
                m0, m, ei = self.Opt(Mli)
                ms_i = self.ms[ ican ] # stores the updated geom
                rms_i = self.rms[ ican ]
                ms0_i = self.ms0[ ican ] # stores the original geom
                nci = len(ms_i)
                es_i = self.es[ ican ]

                inew = True
                if self.param.idiff == 0: # use difference of energy as citeria
                    dEs = np.abs( np.array(es_i) - ei )
                    if np.any( dEs <= self.param.thresh ): inew = False
                elif self.param.idiff == 1:
                    xs = np.array([ rmol.cm, ] )
                    ys = np.array([ ma.cm for ma in self.rms[ican] ])
                    #print ' -- ', xs.shape, ys.shape, can
                    drps = ssd.cdist(xs, ys, 'cityblock')[0]
                    if np.any( drps <= self.param.thresh ): inew = False
                elif self.param.idiff == 2:
                    if not self.check_eigval():
                        inew = False
                else:
                    raise '#ERROR: not supported `idiff'

                if inew:
                    self.ms[ ican ] = ms_i + [m, ]
                    self.rms[ ican ] = rms_i + [ rmol, ]
                    self.ms0[ ican ] = ms0_i + [m0, ]
                    self.es[ ican ] = es_i + [ei, ]
                    self.maps.append( [ir, ican, nci] )
        else:
            m0, m, ei = self.Opt(Mli)
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nhas.append( nha )
            self.ms.append( [m, ] )
            self.rms.append( [rmol, ] )
            self.ms0.append( [m0, ] )
            self.es.append( [ei, ] )
            self.ncan += 1

    def update2(self, ir, can, Mli):
        """
        update mol set if we need SMILES only
        """
        self.ncan = len(self.cans)
        zs = Mli[0]
        nha = (zs > 1).sum()
        if can not in self.cans:
            print '++', can #, '\n\n'
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nhas.append( nha )
            self.ncan += 1
        else:
            ican = self.cans.index( can )
            entry = [ir, ican, 0]
            if entry not in self.maps:
                self.maps.append( entry )
        #print ' -- maps = ', self.maps

    def Opt(self, Mli):
        """
        postprocess molecular fragement retrieved
        from parent molecule by RDKit
        """

        #import io2.mopac as im
        import tempfile as tpf

        zs, coords, bom, charges = Mli
        ctab = oe.write_sdf_raw(zs, coords, bom, charges)

        # get RDKit Mol first
        m0 = Chem.MolFromMolBlock( ctab, removeHs=False ) # plz keep H's
        m0_copy = copy.deepcopy(m0)
        rd = cir.RDMol( m0_copy, forcefield=self.param.ff )

        if self.param.wg:
# the default case, use openbabel to do constrained optimization
            if self.param.do_ob_ff:
                ob1 = cib.Mol( ctab, fmt='sdf' )
                ob1.fixTorsionOpt(iconstraint=3, ff="MMFF94", \
                           optimizer='cg', steps=[30,90], ic=True)
                rd = cir.RDMol( ob1.to_RDKit(), forcefield=self.param.ff )
            else:
# u may prefer to do a partial optimization using PM7 in MOPAC
# for those H atoms and their neighboring heavy atoms
                pass # no ff opt
        if hasattr(rd, 'energy'):
            e = rd.energy
        else:
            e = rd.get_energy()
        m = rd.m
        return m0, m, e

    def _sort(self):
        """ sort Mlis """
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nhas = np.array(self.nhas)
        ncs = [ len(ms_i) for ms_i in self.ms ]
        cans = np.array(self.cans)
        nhas_u = []
        ncs_u = []
        seqs_u = []
        cans_u = []
        ms_u = []; ms0_u = []

        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.k2+1):
            seqs_i = seqs[ i == nhas ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                ms_j = self.ms[j]; ms0_j = self.ms0[j]
                ncj = len(ms_j)
                ncs_u.append( ncj )
                nhas_u.append( nhas[j] )
                ms_u.append( ms_j ); ms0_u.append( ms0_j )

        seqs_u = np.array(seqs_u)

        # now get the starting idxs of conformers for each amon
        ias2 = np.cumsum(ncs_u)
        ias1 = np.concatenate( ([0,],ias2[:-1]) )

        # now get the maximal num of amons one molecule can possess
        nt = 1+maps[-1,0]; namons = []
        for i in range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps_u stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps_u = np.zeros((nt, namon_max))
        for i in range(nt):
            filt_i = (maps[:,0] == i)
            maps_i = maps[filt_i, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan, jc = maps_i[j,:] # `jcan: the old idx of can
                jcan_u = seqs[ seqs_u == jcan ] # new idx of can
                maps_u[i, jcnt] = ias1[jcan_u] + jc
                jcnt += 1
        self.ms = ms_u
        self.ms0 = ms0_u
        self.cans = cans_u
        self.nhas = nhas_u
        self.ncs = ncs_u
        self.maps = maps_u


    def _sort2(self):
        """ sort Mlis for wg = False"""
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nhas = np.array(self.nhas)
        cans = np.array(self.cans)

        nhas_u = []
        seqs_u = []
        cans_u = []
        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.k2+1):
            seqs_i = seqs[ i == nhas ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                nhas_u.append( nhas[j] )

        seqs_u = np.array(seqs_u)

        #print 'maps = ',maps

        # now get the maximal num of amons one molecule can possess
        nt = maps[-1,0]+1; namons = []
        for i in range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps_u stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps_u = np.zeros((nt, namon_max))
        for i in range(nt):
            filt_i = (maps[:,0] == i)
            maps_i = maps[filt_i, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan = maps_i[j,1] # `jcan: the old idx of can
                jcan_u = seqs[ seqs_u == jcan ] # new idx of can
                maps_u[i, jcnt] = jcan_u
                jcnt += 1
        self.cans = cans_u
        self.nhas = nhas_u
        self.maps = maps_u
        self.ncs = np.ones(ncan).astype(np.int)


def accommodate_chgs(chgs, bom):
    """update bom based on `chgs
    e.g., C=N#N, bond orders = [2,3],
    Considering that `chgs = [0,+1,-1],
    bond orders has to be changed to [2,2]"""
    bom2 = copy.copy(bom)
    na = len(chgs)
    ias = np.arange(na)
    ias1 = ias[chgs == 1]
    for i in ias1:
        iasc = ias[ np.logical_and(chgs==-1, bom[i]>0) ]
        nac = len(iasc)
        if nac > 0:
            #assert nac == 1
            j = iasc[0]
            bij = bom[i,j] - 1
            bom2[i,j] = bij
            bom2[j,i] = bij
    return bom2





class vars(object):
    def __init__(self, bosr, zs, chgs, tvs, g, coords):
        self.bosr = bosr
        self.zs = zs
        self.chgs = chgs
        self.tvs = tvs
        self.g = g
        self.coords = coords


class MG(vars):

    def __init__(self, bosr, zs, chgs, tvs, g, coords, use_bosr=True):
        """
        use_bosr: set to True for generating amons, i.e., we need the
                  bond orders between the atom_i and all its neighbors,
                  where `i runs through 1 to N_A;
                  It must be set to False when inferring the BO's between
                  atoms given only the xyz file, i.e., with graph being
                  the only input
        """
        vars.__init__(self, bosr, zs, chgs, tvs, g, coords)
        self.use_bosr = use_bosr

    def update_m(self, once=True, debug=False, icon=False):

        import cheminfo.fortran.famon as cf

        g = self.g
        chgs = self.chgs
        vs = g.sum(axis=0).astype(np.int)
        tvs = self.tvs # `tvs has been modified according to `chgs
        zs = self.zs
        bosr = self.bosr
        na = len(zs)
        ias = np.arange(na)

        #icon = True
        if icon:
            print ' zs = ', zs
            print 'tvs = ', tvs
            print 'dvs = ', tvs - vs

        #print 'g = ', g
        #t1 = time.time()
        #print ' ## e1'
        nrmax = na/2
        nbmax = (g>0).sum()/2
        iok, bom = cf.update_bom(nrmax,nbmax,zs,tvs,g,icon)
        if icon: print '     +++ Passed with `iok = ', iok


        #t2 = time.time()
        #print '      update_m: ', t2-t1
        #print ' ** iok = ',iok
        #print ' ** bom = ', bom
        if not iok:
            #print ' zs = ', zs
            #print ' vs = ', vs
            #print 'tvs = ', tvs
            #print ''
            return [],[]

        boms = [bom]
        cans = []; ms = []
        iok = True
        for bom in boms:

            # note that the order of calling `get_bos() and `accommodate_chgs()
            #  matters as `bosr was obtained based on modified `bom, i.e., all
            # pairs of positive & negative charges (the relevant two atoms are
            # bonded) were eliminated
            bos = get_bos(bom)

            # now restore charges for case, e.g., NN bond in C=N#N, or -N(=O)=O
            bom_U = accommodate_chgs(chgs, bom)
            vs = bom_U.sum(axis=0)


            # for query molecule like -C=CC#CC=C-, one possible amon
            # is >C-C-C-C< with dvs = [1,2,2,1] ==> >C=C=C=C<, but
            # apparently this is not acceptable!! We use `obsr to
            # kick out these fragments if `use_bosr is set to .true.
            #ipass = True
            if self.use_bosr:
                #print ' -- bos = ', bos
                if np.any(bos[zs>1] != bosr):
                    #print ' bosr = ', bosr, ', bos = ', bos[zs>1]
                    #ipass = False
                    continue

            t1 = time.time()

            # handle multivalent cases
            #    struct                obabel_amons
            # 1) R-N(=O)=O,            O=[SH2]=O
            # 2) R1-P(=O)(R2)(R3)
            # 3) R-S(=O)-R,
            # 4) R-S(=O)(=O)-R
            # 5) R-Cl(=O)(=O)(=O), one possible amon is
            # "O=[SH2]=O", however,
            # openbabel cannot succeed to add 2 extra H's. We can circumvent this
            # by using isotopes of H's
            isotopes = []
            zsmv = [7,15,16,17]
            vsn = [3,3,2,1]
            zsc = np.intersect1d(zs, zsmv)
            if zsc.shape[0] > 0:
                nheav = (zs > 1).sum()
                ias = np.arange(len(zs))
                for ia in range(nheav):
                    if (zs[ia] in zsmv) and (vs[ia]>vsn[ zsmv.index(zs[ia]) ]):
                        jas = ias[bom_U[ia] > 0]
                        for ja in jas:
                            if zs[ja] == 1:
                                isotopes.append(ja)
            if na <= 100:
                blk = write_ctab(zs, chgs, bom_U, self.coords, isotopes=isotopes, sdf=None)
                m = obconv(blk)
            else:
                blk_pdb = crp.write_pdb( (zs,self.coords,chgs,bom_U) )
                m = obconv(blk_pdb,'pdb')

            #t2 = time.time()
            #print '                |_ dt1 = ', t2-t1

            can_i = pb.Molecule(m).write('can').split('\t')[0]
            #if not ipass: print ' ++ can_i = ', can_i
            #if np.any(bos[zs>1] != bosr):
            #    print '##### ', can_i, ', ', bos[zs>1], ', ', bosr
            #    continue

            # remove isotopes
            sp = r"\[[1-3]H\]"
            sr = "[H]"
            _atom_name_pat = re.compile(sp)
            can_i = _atom_name_pat.sub(sr, can_i)

            #print ' ++ zs, can, isotopes = ', zs, can_i, isotopes
            #t3 = time.time()
            #print '                |_ dt2 = ', t3-t2
            #print '             __ can = ', can_i
            if can_i not in cans:
                cans.append(can_i)
                ms.append(m)
        #if 'CC(C)C' in cans: print ' Alert!!!'
        return cans, ms

def get_coords(m):
    coords = [] # np.array([ ai.coords for ai in pb.Molecule(m).atoms ])
    na = m.NumAtoms()
    for i in range(na):
        ai = m.GetAtomById(i)
        coords.append( [ ai.GetX(), ai.GetY(), ai.GetZ() ] )
    return np.array(coords)

def get_bom(m):
    """
    get connectivity table
    """
    na = m.NumAtoms()
    bom = np.zeros((na,na), np.int)
    for i in range(na):
        ai = m.GetAtomById(i)
        for bond in ob.OBAtomBondIter(ai):
            ia1 = bond.GetBeginAtomIdx()-1; ia2 = bond.GetEndAtomIdx()-1
            bo = bond.GetBO()
            bom[ia1,ia2] = bo; bom[ia2,ia1] = bo
    return bom

def clone(m):
    m2 = pb.Molecule(m).clone
    return m2.OBMol

def check_hydrogens(m):
    mu = pb.Molecule(m).clone # a copy
    mu.addh()
    m2 = mu.OBMol
    return m.NumAtoms() == m2.NumAtoms()

def obconv(s,fmt='sdf'):
    """ convert string(s) to molecule given a format
    e.g, 'CCO','smi'
         or sdf_file_content,'sdf' """
    conv = ob.OBConversion()
    m = ob.OBMol()
    #assert type(s) is str
    conv.SetInFormat(fmt)
    conv.ReadString(m,s)
    return m


def get_bos(bom):
    na = bom.shape[0]
    bosr = []
    for i in range(na):
        bosi = bom[i]
        t = bosi[ bosi > 0 ]; t.sort()
        n = len(t)
        v = 0
        for j in range(n):
            v += t[j]*10**j
        bosr.append( v )
    return np.array(bosr,np.int)


class mol(object):

    def __init__(self, m0):
        na = m0.NumAtoms()
        m1 = clone(m0); m1.DeleteHydrogens()
        self.m0 = m0
        #print 'self.m = ', m1
        self.m = m1
        chgs = []; zs = []
        for i in range(na):
            ai = m0.GetAtomById(i)
            zi = ai.GetAtomicNum(); zs.append( zi )
            chgi = ai.GetFormalCharge(); chgs.append( chgi )

        self.zs = np.array(zs)
        self.bom = get_bom(m0)
        self.nheav =  (self.zs > 1).sum()


        self.ias = np.arange( len(self.zs) )
        self.ias_heav = self.ias[ self.zs > 1 ]
        try:
            self.coords = get_coords(m0)
        except:
            self.coords = np.zeros((na,3))

        self.chgs = np.array(chgs, np.int)
        #if 1 in zs:
        #    idxh = zs.index( 1 )
        #    if np.any(self.zs[idxh+1:] != 1):
        #        # not all H apprear appear at the end, u have to sort it
        #        self.sort()

        # check if there is any XH bond appear before XY bond
        ihsmi = False
        obsolete = """nb = m0.NumBonds(); ibs = []
        for ib in range(nb):
            bi=  m0.GetBondById(ib)
            j,k = [ bi.GetBeginAtomIdx(), bi.GetEndAtomIdx() ] # starts from 1
            if j == 1 or k == 1:
                ibs.append(ib) #[zs[j-1],zs[k-1]])
        ibs = np.array(ibs,np.int)
        if not np.all( ibs[1:]-ibs[:-1] == 1 ): ihsmi = True"""

        # a even simpler way to tell if H atom/bond appears before X
        nb = m1.NumBonds()
        for ib in range(nb):
            bi = m1.GetBondById(ib)
            if bi == None:
                ihsmi = True; break

        # sort atoms & bonds so that H atom or HX bond always appear at the end
        if ihsmi: self.sort()


        vs = self.bom.sum(axis=0)
        #print '  * vs = ', vs
        self.vs = vs
        if np.any(self.chgs != 0):
            #print ' ** update bom due to charges'
            self.eliminate_charges()
        else:
            # figure out charges for some special cases like
            # R-N(=O)=O, O=N(=C)C=C, R-C=N#N, etc as Openbabel
            # is not intelligent enough; for packages like
            # RDKit or OEChem, you don't have to do this
            self.recover_charges()
#       print ' -- chgs = ', self.chgs
        #print ' ** vs = ', self.vs

        bom_heav = self.bom[ self.ias_heav, : ][ :, self.ias_heav ]
#       print 'bom_heav = ', bom_heav
        self.vs_heav = bom_heav.sum(axis=0)
        self.cns_heav = ( bom_heav > 0 ).sum(axis=0)
        # get formal charges
        self.cns = ( self.bom > 0).sum(axis=0)
        self.nhs = self.vs[:self.nheav] - self.vs_heav #- self.chgs[:self.nheav]
        self.dvs = self.vs_heav - self.cns_heav

        # get bosr, i.e., bond order (reference data) array
        # concatenated into a integer
        self.bosr = get_bos(self.bom)
        self.dbnsr = (self.bom==2).sum(axis=0)
        #print ' -- bosr = ', self.bosr
        self.na = na


    def sort(self):
        """ sort atoms so that H's appear at the end
        """
        nheav = self.nheav
        ias_heav = list(self.ias_heav)
        g = np.zeros((nheav, nheav))
        xhs = [] # X-H bonds
        ih = nheav
        coords = []; coords_H = []
        chgs = []; chgs_H = []
        dic = dict( zip(ias_heav, range(nheav)) )
#       print ' *** dic = ', dic
        for i, ia in enumerate( ias_heav ):
            coords.append( self.coords[ia] )
            chgs.append( self.chgs[ia] )
            jas = self.ias[ self.bom[ia,:] > 0 ]
            for ja in jas:
                if self.zs[ja] == 1:
                    coords_H.append( self.coords[ja] )
                    chgs_H.append( self.chgs[ja] )
                    xhs.append([i,ih]); ih += 1
                else:
                    g[i,dic[ja]] = g[dic[ja],i] = self.bom[ia,ja]
        coords_U = np.concatenate( (coords, coords_H) )
        self.coords = coords_U
        chgs_U = np.concatenate( (chgs, chgs_H) )
        self.chgs = chgs_U
        g2 = np.zeros((ih,ih))
        g2[:nheav, :nheav] = g
        for xh in xhs:
            i,j = xh
            g2[i,j] = g2[j,i] = 1
        self.bom = g2
        nh = ih - nheav
        zsU = np.array( list(self.zs[ias_heav]) + [1,]*nh )
        self.zs = zsU
        self.ias_heav = self.ias[ self.zs > 1 ]
        blk = write_ctab(zsU, chgs_U, g2, coords_U, sdf=None)
        m0 = obconv(blk)
        m1 = clone(m0)
#       print ' *** ', Chem.MolToSmiles(m1)
        m1.DeleteHydrogens()
        self.m0 = m0; self.m = m1


    def eliminate_charges(self):
        """update bom based on `chgs
        e.g., bom of C=[N+]=[N-] will be converted to bom of C=N#N
        based on `chgs = [0,+1,-1]
        Note that only bom and the resulting `vs will be updated, no
        changes regarding the SMILES string (i.e., we still prefer
        a SMILES string like C=[N+]=[N-] instead of C=N#N"""

        bom2 = copy.copy(self.bom)
        vs2 = self.vs
        ias1 = self.ias[self.chgs == 1]
        for i in ias1:
            iasc = self.ias[ np.logical_and(self.chgs==-1, self.bom[i]>0) ]
            nac = len(iasc)
            if nac > 0:
                #print ' __ yeah'
                #assert nac == 1
                j = iasc[0]
                bij = self.bom[i,j] + 1
                bom2[i,j] = bij
                bom2[j,i] = bij
                vs2[i] = vs2[i]+1; vs2[j] = vs2[j]+1
        self.bom = bom2
        #print ' __ bom2 = ', bom2
        self.vs = vs2 #bom2.sum(axis=0)  #vs2


    def recover_charges(self):
        """figure out the charges of N atoms contraining that
        all have a valence of 3. E.g., for "CC=CC=N#N", the final
        charges of atoms is [0,0,0,0,1,-1], corresponding to the
        SMILES string of "CC=CC=[N+]=[N-]". It's similar for "CCN(=O)=O".
        """
        bom2 = copy.copy(self.bom)
        vs2 = self.vs
        ias1 = self.ias[ np.logical_and(vs2 == 5, self.zs == 7) ]
        chgs = self.chgs
        for ia in ias1:
            bom_ia = bom2[ia]
            jas = self.ias[ bom_ia >=2 ]
            bosj = bom_ia[ bom_ia >= 2 ]
            if len(jas) == 2:
                zsj = self.zs[ jas ]
                if set(bosj) == set([2])         or set(bosj) == set([2,3]):
                    # e.g., O=N(=C)C=C, O=N(=O)C        CC=CC=N#N
                    for ja in jas:
                        if (bom2[ja] > 0).sum() == 1:
                            chgs[ia] = 1; chgs[ja] = -1
                            break
                else:
                    raise '#ERROR: wierd case!'
        self.chgs = chgs


    def get_ab(self):
        """
        For heav atoms only

        get atoms and bonds info
        a2b: bond idxs associated to each atom
        b2a: atom idxs associated to each bond
        """
        # it's not necessary to exclude H's here as H's apprear at the end
        b2a = [] #np.zeros((self.nb,2), np.int)
        ibs = []
        nb = self.m.NumBonds()
        for ib in range(nb):
            bi = self.m.GetBondById(ib)
            i, j = bi.GetBeginAtomIdx()-1, bi.GetEndAtomIdx()-1
            if self.zs[i] > 1 and self.zs[j] > 1:
                ib_heav = bi.GetIdx()
                b2a.append( [i,j] )
        #assert len(b2a) == ib_heav+1, '#ERROR: not all H apprear at the end?'
        b2a = np.array(b2a, np.int)

        # assume at most 7 bonds for an atom (i.e., IF7 molecule)
        a2b = -np.ones((self.nheav, 7), np.int) # -1 means no corresponding bond
        for ia in self.ias_heav:
            ai = self.m.GetAtomById(ia)
            icnt = 0
            for bi in ob.OBAtomBondIter(ai):
                ib = bi.GetId()
                if ib <= ib_heav: #np.all( self.zs[b2a[ib]] > 1 ):
                    a2b[ia, icnt] = ib
                    icnt += 1
        return a2b, b2a


def remove_charge(m):
    # obabel molecule as input
    dic = {}
    for ai in ob.OBMolAtomIter(m):
        idx = ai.GetId()
        vi = ai.GetImplicitValence()
        chgi = ai.GetFormalCharge()
        assert abs(chgi) <= 1
        dic[ idx ] = chgi
        if chgi in [1,-1]:
            chgs = []
            for aj in ob.OBAtomAtomIter(ai):
                jdx = aj.GetId()
                chgj = aj.GetFormalCharge()
                dic[ jdx ] = chgj
                chgs.append( chgj )
            if len(chgs) > 0 and np.all(np.array(chgs,np.int) == 0):
                ai.SetFormalCharge( 0 )
                # reset valence for positively charged atom
                ai.SetImplicitValence( vi-chgi )

                # to continue, you need to remove one H atom
                # and reassign the values of atom indices;
                # Alternatively, simply return an updated SMILES
                #if chgi == 1:
                #    # remove one hydrogen atom from, say [NH3+]
    pym = pb.Molecule(m)
    su = pym.write('can')
    #print ' ++ ', su
    return su


def check_elements(zs):
    # metals are all excluded, including
    # Li,Ba,Mg,K,Ca,Rb,Sr,Cs,Ra and
    # Sc-Zn
    # Y-Cd
    # La-Lu, Hf-Hg
    zsa = [3,11,12,19,20,37,38,55,56] + \
          range(21,31) + \
          range(39,49) + \
          range(57,81) + \
          range(89,113) # Ac-Lr, Rf-Cn
    return np.all([ zi not in zsa for zi in zs ])


class amon(object):

    """
    use openbabel only
    """

    def __init__(self, s, k, k2=None, wg=False, ligand=None, \
                 fixGeom=False, ikeepRing=True, \
                 allow_isotope=False, allow_charge=False, \
                 allow_radical=False):
        """
        ligand -- defaulted to None; otherwise a canonical SMILES
        has to be specified

        vars
        ===============
        s -- input string, be it either a SMILES string or sdf file
        k -- limitation imposed on the number of heav atoms in amon
        """

        if k2 is None: k2 = k
        self.k = k
        self.k2 = k2
        self.wg = wg
        self.fixGeom = fixGeom
        self.ikeepRing = ikeepRing

        iok = True # shall we proceed?
        if os.path.exists(s):
            m0 = obconv(s,s[-3:])

            # set isotope to 0
            # otherwise, we'll encounter SMILES like 'C[2H]',
            # and error correspondently.
            # In deciding which atoms should be have spin multiplicity
            # assigned, hydrogen atoms which have an isotope specification
            # (D,T or even 1H) do not count. So SMILES N[2H] is NH2D (spin
            # multiplicity left at 0, so with a full content of implicit
            # hydrogens), whereas N[H] is NH (spin multiplicity=3). A
            # deuterated radical like NHD is represented by [NH][2H].
            na = m0.NumAtoms()
            if not allow_isotope:
                for i in range(na):
                    ai = m0.GetAtomById(i); ai.SetIsotope(0)

            # add lines below to tell if HX bond appears before some heav atom bonds
            # _____________
            #
            #
            assert check_hydrogens(m0), '#ERROR: some hydrogens are missing'
            coords0 = get_coords(m0)
            pym = pb.Molecule(m0).clone
            # check consistency
            if pym.charge != 0 and (not allow_charge): iok = False
            if pym.spin > 1 and (not allow_radical): iok = False

            m = pym.OBMol; m.DeleteHydrogens()
        else:
            if not allow_isotope:
                # remove isotopes
                patts = [r"\[[1-3]H\]", r"\[[1-9]*[1-9]+"]
                # e.g., C1=C(C(=O)NC(=O)N1[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)F)[124I]
                #       [3H]C
                #       CN([11CH3])CC1=CC=CC=C1SC2=C(C=C(C=C2)C#N)N
                subs = ["", "["]
                for ir in range(2):
                    sp = patts[ir]
                    sr = subs[ir]
                    _atom_name_pat = re.compile(sp)
                    s = _atom_name_pat.sub(sr,s)

            # There exists one anoying bug of `openbabel, i.e.,
            # for some SMILES string, the program halts when trying to convert
            # from SMILES to Mol. E.g., "CCCC[C@@H](C(=O)N[C@@H](C(C)CC)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](C(C)CC)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NC1CCC(=O)NCCCC[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC1=O)CC(=O)N)CCCN=C(N)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)NC(CC(=O)O)C(=O)N[C@](C)(CC(C)C)C(=O)N[C@H](C(C)CC)C(=O)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@](C)(CC(C)C)NC(=O)[C@H](CC2=CNC=N2)NC(=O)[C@@H](CC3=CC=CC=C3)NC(=O)[C@H](CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)C"
            # To circumvent this, we have to remove all stereochemistry first
            pat = re.compile(r"\[(\w+?)@@?\w?\]")
            matches = list( set( pat.findall(s) ) )
            for match in matches:
                _pat = re.compile(r"\[%s@@?\w?\]"%match)
                s = _pat.sub(match, s)

            m = obconv(s,'smi')
            pym = pb.Molecule(m).clone
            if not allow_radical:
                if pym.spin > 1: iok = False

#            print ' ++ 3'
            if not allow_charge:
                # now remove charge
                su = remove_charge(m)
                m = obconv(su,'smi')
            m0 = clone(m)
            m0.AddHydrogens()

#        print ' ++ 5'
        if iok:
            zs = [ ai.atomicnum for ai in pym.atoms ]
            if not check_elements(zs):
                iok = False

        self.iok = iok
        if iok: self.objQ = mol(m0)

        self.m0 = m0
        self.m = m


    def get_subm(self, las, lbs, sg):
        """
        add hydrogens & retrieve coords
        """
        #sets = [ set(self.objQ.bs[ib]) for ib in lbs ] # bond sets for this frag
        nheav = len(las)
        dic = dict( zip(las, range(nheav)) )
        ih = nheav;
        xhs = [] # X-H bonds
        if self.wg:
            coords = []; coords_H = []
            for i,ia in enumerate(las):
                coords.append( self.objQ.coords[ia] )
                jas = self.objQ.ias[ self.objQ.bom[ia,:] > 0 ]
                for ja in jas:
                    if self.objQ.zs[ja] == 1:
                        coords_H.append( self.objQ.coords[ja] )
                        xhs.append([i,ih]); ih += 1
                    else:
                        #if (ja not in las) or ( (ja in las) and (set(ia,ja) not in sets) ):
                        if (ja not in las) or ( (ja in las) and (sg[i,dic[ja]] > 0) ):
                            v = self.objQ.coords[ja] - coords_i
                            coords_H.append( coord + dsHX[z] * v/np.linalg.norm(v) )
                            xhs.append([i,ih]); ih += 1
            coords_U = np.concatenate( (coords, coords_H) )
        else:
            for i,ia in enumerate(las):
                jas = self.objQ.ias[ self.objQ.bom[ia,:] > 0 ]
                for ja in jas:
                    if self.objQ.zs[ja] == 1:
                        xhs.append([i,ih]); ih += 1
                    else:
                        if (ja not in las) or ( (ja in las) and (sg[i,dic[ja]] == 0) ):
                            xhs.append([i,ih]); ih += 1
            coords_U = np.zeros((ih,3))
        sg_U = np.zeros((ih,ih))
        sg_U[:nheav, :nheav] = sg
        for xh in xhs:
            i,j = xh
            sg_U[i,j] = sg_U[j,i] = 1

        nh = ih - nheav
        bosr1 = self.objQ.bosr[las] # for heav atoms only
        zs1 = np.array( list(self.objQ.zs[las]) + [1,]*nh )
        chgs1 = np.array( list(self.objQ.chgs[las]) + [0,]*nh )
        tvs1 = np.array( list(self.objQ.vs[las]) + [1,]*nh )
        vars1 = vars(bosr1, zs1, chgs1, tvs1, sg_U, coords_U)
        self.vars = vars1

    def get_amons(self):
        """
        tell if a given frag is a valid amon
        """

        objQ = self.objQ

        amons = []
        smiles = []

        # get amon-2 to amon-k
        g0 = ( objQ.bom > 0 ).astype(np.int)

        amons = []
        cans = []; ms = []
        a2b, b2a = objQ.get_ab()
        bs = [ set(jas) for jas in b2a ]
        for seed in generate_subgraphs(b2a, a2b, self.k):
            # lasi (lbsi) -- the i-th list of atoms (bonds)
            lasi, lbsi = list(seed.atoms), list(seed.bonds)
            _lasi = np.array(lasi).astype(np.int)
            #lasi.sort()

            #can = Chem.MolFragmentToSmiles(objQ.m, atomsToUse=lasi, kekuleSmiles=False, \
            #                               bondsToUse=lbsi, canonical=True)

            #print ''
            #print ' zs = ', objQ.zs[lasi]
            #print 'tvs = ', objQ.vs[lasi]
            #                               bondsToUse=lbsi, canonical=True)

            iprt = False
            bs = []
            for ibx in lbsi:
                bs.append( set(b2a[ibx]) )
                #if iprt:
                #    print '  -- ibx, ias2 = ', ibx, tuple(b2a[ibx])

            na = len(lasi)
            if na == 1:
                ia = lasi[0]; zi = objQ.zs[ ia ]
                iok1 = (zi in [9, 17, 35, 53])
                iok2 = ( np.any(objQ.bom[ia] >= 2) ) # -S(=O)-, -P(=O)(O)-, -S(=O)(=O)- and #N
                if np.any([iok1, iok2]):
                    continue
                can = chemical_symbols[ zi ]
                if can not in cans:
                    cans.append( can )
                #    if wg:
                #        if not self.fixGeom:
                #            ms.append( ms0[can] )
                #        else:
                #            raise '#ERROR: not implemented yet'
                #else:
                #    if wg and self.fixGeom:
                continue

            sg0_heav = g0[lasi,:][:,lasi]
            nr0 = cg.get_number_of_rings(sg0_heav)

            # property of atom in the query mol
            nhs_sg0 = objQ.nhs[lasi]
#           print ' cns_heav = ', objQ.cns_heav
            cns_sg0_heav = objQ.cns_heav[lasi]

            zs_sg = objQ.zs[ lasi ]
            sg_heav = np.zeros((na,na))
            for i in range(na-1):
                for j in range(i+1,na):
                    bij = set([ lasi[i], lasi[j] ])
                    if bij in bs:
                        sg_heav[i,j] = sg_heav[j,i] = 1
            nr = cg.get_number_of_rings(sg_heav)
            ir = True
            if self.ikeepRing:
                if nr != nr0:
                    ir = False

            cns_sg_heav = sg_heav.sum(axis=0)

#           if iprt:
#               print '  -- cns_sg0_heav, cns_sg_heav = ', cns_sg0_heav, cns_sg_heav
#               print '  -- dvs_sg_heavy = ', objQ.dvs[lasi]
#               print '  -- nhs = ', objQ.nhs[lasi]

#           print zs_sg, cns_sg0_heav, cns_sg_heav #
            dcns = cns_sg0_heav - cns_sg_heav # difference in coordination numbers
            assert np.all( dcns >= 0 )
            num_h_add = dcns.sum()
#           if iprt: print '  -- dcns = ', dcns, ' nhs_sg0 = ', nhs_sg0
            ztot = num_h_add + nhs_sg0.sum() + zs_sg.sum()
#           if iprt: print '  -- ztot = ', ztot
            chg0 = objQ.chgs[lasi].sum()

            # test
            #_cns2 = list(objQ.cns[lasi]); _cns2.sort()
            icon = False
            #if na == 7 and np.all(np.unique(zs_sg)==np.array([6,16])) and np.all(np.array(_cns2) == np.array([2,2,2,2,3,3,4])):
            #    icon = True; print ' ***** '

            if ir and ztot%2 == 0 and chg0 == 0:
                # ztot%2 == 1 implies a radical, not a valid amon for neutral query
                # this requirement kills a lot of fragments
                # e.g., CH3[N+](=O)[O-] --> CH3[N+](=O)H & CH3[N+](H)[O-] are not valid
                #       CH3C(=O)O (CCC#N) --> CH3C(H)O (CCC(H)) won't survive either
                #   while for C=C[N+](=O)[O-], with ztot%2 == 0, [CH2][N+](=O) may survive,
                #       by imposing chg0 = 0 solve the problem!

                tvsi0 = objQ.vs[lasi] # for N in '-[N+](=O)[O-]', tvi=4 (rdkit)
                bom0_heav = objQ.bom[lasi,:][:,lasi]
                dbnsi = (bom0_heav==2).sum(axis=0) #np.array([ (bom0_heav[i]==2).sum() for i in range(na) ], np.int)
                zsi = zs_sg
                ias = np.arange(na)

                ## 0) check if envs like '>S=O', '-S(=O)(=O)-', '-P(=O)<',
                ## '-[N+](=O)[O-]' (it's already converted to '-N(=O)(=O)', so `ndb=2)
                ## 'R-S(=S(=O)=O)(=S(=O)(=O))-R', '-C=[N+]=[N-]' or '-N=[N+]=[N-]'
                ## ( however, '-Cl(=O)(=O)(=O)' cannot be
                ## recognized by rdkit )
                ## are retained if they are part of the query molecule
##### lines below are not necessary as `bosr will be used to assess
##### if the local envs have been kept!

## actually, the role of the few lines below is indispensible.
## E.g., for a mol c1ccccc1-S(=O)(=O)C, an amon like C=[SH2]=O
## has bos='2211', exactly the same as the S atom in query. But
## it's not a valid amon here as it's very different compared
## to O=[SH2]=O...
## Another example is C=CS(=O)(=O)S(=O)(=O)C=C, an amon like
## [SH2](=O)=[SH2](=O) has bos='2211' for both S atoms, but are
## not valid amons
                tvs1  = [   4,      6,   5,   5 ]
                zs1   = [  16,     16,  15,   7]
                _dbns = [ [1], [2, 3], [1], [2] ] # number of double bonds
                #               |  |
                #               |  |___  'R-S(=S(=O)=O)(=S(=O)(=O))-R',
                #               |
                #               |___ "R-S(=O)(=O)-R"

                #_zsi = [ _zi for _zi in zsi ]
                #_zsi.sort()
                #if np.all(_zsi == np.array([8,8,8,8,16,16,16]) ):
                #    print '##'
                #    icon=True

                #print ' __ zsi = ', zsi

                istop = False
                # now gather all atomic indices need to be compared
                jas = np.array([], np.int)
                for j,tvj in enumerate(tvs1):
                    filt = np.logical_and(tvsi0 == tvj, zsi == zs1[j])
                    _jas = ias[filt].astype(np.int)
                    jas = np.concatenate( (jas,_jas) )
                # now compare the num_double_bonds
                if len(jas) > 0:
                    dbnsj = dbnsi[jas]
                    dbnsrj = objQ.dbnsr[ _lasi[jas] ]
                    if np.any(dbnsj != dbnsrj):
                        istop = True; continue #break
                        #print 'tvj, zs1[j], dbnsj, dbns1[j] = ', tvj, zs1[j], dbnsj, dbns1[j]
                        #print ' __ zsi = ', zsi, ', istop = ', istop
                #if istop: continue #"""
                #print ' __ zsi = ', zsi


                self.get_subm(lasi, lbsi, sg_heav)
                vr = self.vars
## added on Aug 13, 2018
#                # constraint that coordination numbers being the same
#                cnsi = (vr.g > 0).sum(axis=0)[:na]
#                cnsri = self.objQ.cns[lasi]
#                if np.any( cnsi - cnsri != 0 ):
#                    continue
#                else:
#                    print '## CN ok! ', cnsi
# added on Aug 13, 2018
                so = ''
                for i in range(na):
                    for j in range(i+1,na):
                        if vr.g[i,j] > 0: so += '[%d,%d],'%(i+1,j+1)
                #print so

                cmg = MG( vr.bosr, vr.zs, vr.chgs, vr.tvs, vr.g, vr.coords )

                # test
                #if icon: print ' ************* '

                # for diagnosis
                gr = []
                nat = len(vr.zs); ic = 0
                for i in range(nat-1):
                    for j in range(i+1,nat):
                        gr.append( vr.g[i,j] ); ic += 1
                test = """
                s = ' ########## %d'%nat
                for i in range(nat): s += ' %d'%vr.zs[i]
                for i in range(nat): s += ' %d'%vr.tvs[i]
                for i in range(ic): s += ' %d'%gr[i]
                print s
                #"""

                #if so == '[1,2],[1,6],[2,3],[3,4],[4,5],[4,7],[5,6],':
                #    icon = True

                #if len(objQ.zs[lasi])==3:
                #    if np.all(objQ.zs[lasi] == np.array([7,7,7])): print '## we r here'

                cans_i = []
                cans_i, ms_i = cmg.update_m(debug=True,icon=icon)
                #if icon: print ' -- cans = ', cans_i
                for can_i in cans_i:
                    if can_i not in cans:
                        cans.append( can_i )
                #if icon: print ''
                if icon:
                    print '###############\n', cans_i, '############\n'
        return cans



class ParentMols(object):

    def __init__(self, strings, fixGeom, iat=None, wg=True, k=7,\
                 nmaxcomb=3,icc=None, substring=None, rc=6.4, \
                 isort=False, k2=7, opr='.le.', wsmi=True, irc=True, \
                 iters=[30,90], dminVDW= 1.2, \
                 idiff=0, thresh=0.2, \
                 keepHalogen=False, debug=False, ncore=1, \
                 forcefield='mmff94', do_ob_ff=True, \
                 ivdw=False, covPLmin=5, prefix=''):
        """
        prefix -- a string added to the beginning of the name of a
                  folder, where all sdf files will be written to.
                  It should be ended with '_' if it's not empty
        irc    -- T/F: relax w/wo dihedral constraints

        substring -- SMILES of a ligand.
                  Typically in a protein-ligand complex, we need
                  to identify the ligand first and then retrieve
                  all the local atoms that bind to the ligand via
                  vdW interaction as amons for training in ML. The
                  thus obtained fragment is dubbed `centre.

                  If `substring is assigned a string,
                  we will generated only amons that are
                  a) molecular complex; b) any atom in the centre
                  must be involved.
        rc     -- cutoff radius centered on each atom of the central
                  component. It's used when `icc is not None.
        """

        def check_ncbs(a, b, c):
            iok = False
            for si in itl.product(a,b):
                if set(si) in c:
                    iok = True; break
            return iok

        param = Parameters(wg, fixGeom, k, k2, ivdw, dminVDW, \
                           forcefield, thresh, do_ob_ff, idiff, iters)

        ncpu = multiprocessing.cpu_count()
        if ncore > ncpu:
            ncore = ncpu

        # temparary folder
        #tdirs = ['/scratch', '/tmp']
        #for tdir in tdirs:
        #    if os.path.exists(tdir):
        #        break

        # num_molecule_total
        assert type(strings) is list, '#ERROR: `strings must be a list'
        nmt = len(strings)
        if iat != None:
            assert nmt == 1, '#ERROR: if u wanna specify the atomic idx, 1 input molecule at most is allowed'

        cans = []; nhas = []; es = []; maps = []
        ms = []; ms0 = []

        # initialize `Sets
        seta = Sets(param)
        for ir in range(nmt):
            print ' -- Mid %d'%(ir+1)
            string = strings[ir]
            obj = ParentMol(string, isort=isort, iat=iat, wg=wg, k=k, k2=k2, \
                            opr=opr, fixGeom=fixGeom, covPLmin=covPLmin, \
                            ivdw=ivdw, dminVDW=dminVDW, \
                            keepHalogen=keepHalogen, debug=debug)
            ncbs = obj.ncbs
            Mlis, iass, cans = [], [], []
            # we needs all fragments in the first place; later we'll
            # remove redundencies when merging molecules to obtain
            # valid vdw complexes
            nas = []; nasv = []; pss = []
            iass = []; iassU = []
            for Mli, ias, can in obj.generate_amons():
                iasU = ias + [-1,]*(k-len(ias)); nasv.append( len(ias) )
                Mlis.append( Mli ); iass.append( ias ); cans.append( can )
                iassU.append( iasU ); pss += list(Mli[1])
                nas.append( len(Mli[0]) )
            nmi = len(cans)
            print ' -- nmi = ', nmi

            nas = np.array(nas, np.int)
            nasv = np.array(nasv, np.int)
            pss = np.array(pss)
            iassU = np.array(iassU, np.int)
            ncbsU = np.array(ncbs, np.int)

            # now combine amons to get amons complex to account for
            # long-ranged interaction
            if wg and ivdw:
                if substring != None:
                    cliques_c = set( oe.is_subg(obj.oem, substring, iop=1)[1][0] )
                    #print ' -- cliques_c = ', cliques_c
                    cliques = oe.find_cliques(obj.g0)
                    Mlis_centre = []; iass_centre = []; cans_centre = []
                    Mlis_others = []; iass_others = []; cans_others = []
                    for i in range(nmi):
                        #print ' %d/%d done'%(i+1, nmi)
                        if set(iass[i]) <= cliques_c:
                            Mlis_centre.append( Mlis[i] )
                            iass_centre.append( iass[i] )
                            cans_centre.append( cans[i] )
                        else:
                            Mlis_others.append( Mlis[i] )
                            iass_others.append( iass[i] )
                            cans_others.append( cans[i] )
                    nmi_c = len(Mlis_centre)
                    nmi_o = nmi - nmi_c
                    print ' -- nmi_centre, nmi_others = ', nmi_c, nmi_o
                    Mlis_U = []; cans_U = []
                    for i0 in range(nmi_c):
                        ias1 = iass_centre[i0]
                        t1 = Mlis_centre[i0]; nha1 = (np.array(t1[0]) > 1).sum()
                        for j0 in range(nmi_o):
                            ias2 = iass_others[j0]
                            t2 = Mlis_others[j0]; nha2 = np.array((t2[0]) > 1).sum()
                            if nha1 + nha2 <= k2 and check_ncbs(ias1, ias2, ncbs):
                                dmin = ssd.cdist(t1[1], t2[1]).min()
                                if dmin >= dminVDW:
                                    cansij = [cans_centre[i0], cans_others[j0]]
                                    cansij.sort()
                                    cans_U.append( '.'.join(cansij) )
                                    Mlis_U.append( merge(t1, t2) )
                    Mlis = Mlis_U; cans = cans_U
                    print ' -- nmi_U = ', len(Mlis)
                else:
                    print 'dminVDW = ', dminVDW
                    gv,gc = fa.get_amon_adjacency(k2,nas,nasv,iassU.T,pss.T,ncbsU.T,dminVDW)
                    print 'amon connectivity done'
                    #print 'gv=',gv # 'np.any(gv > 0) = ', np.any(gv > 0)
                    ims = np.arange(nmi)
                    combs = []
                    for im in range(nmi):
                        nv1 = nasv[im]
                        jms = ims[ gv[im] > 0 ]
                        nj = len(jms)
                        if nj == 1:
                            # in this case, nmaxcomb = 2
                            jm = jms[0]
                            if nmaxcomb == 2:
                                # setting `nmaxcomb = 2 means to include
                                # all possible combinations consisting of
                                # two standalone molecules
                                comb = [im,jms[0]]; comb.sort()
                                if comb not in combs:
                                    combs += [comb]
                            else:
                                # if we are not imposed with `nmaxcomb = 2,
                                # we remove any complex corresponding to 2) below
                                #
                                # 1)    1 --- 2  (no other frag is connected to `1 or `2)
                                #
                                # 2)    1 --- 2
                                #              \
                                #               \
                                #                3
                                if len(gv[jm]) == 1:
                                    comb = [im,jm]; comb.sort()
                                    if comb not in combs:
                                        combs += [comb]
                        else:
                            if nmaxcomb == 2:
                                for jm in jms:
                                    comb = [im,jm]; comb.sort()
                                    if comb not in combs:
                                        combs += [comb]
                            elif nmaxcomb == 3:
                                #for jm in jms:
                                #    comb = [im,jm]; comb.sort()
                                #    if comb not in combs:
                                #        combs += [comb]

                                # this is the default choice and is more reasonable
                                # as only the most relevant local frags are included.
                                # Here we don't consider frags like [im,p],[im,q] as
                                # 1) the local envs are covered by [im,p,q]; 2) it's less
                                # relevant to [im,p,q]
                                for (p,q) in itl.combinations(jms,2):
                                    nv2 = nasv[p]; nv3 = nasv[q]
                                    if nv1+nv2+nv3 <= k2 and gc[p,q] == 0:
                                        comb = [im,p,q]; comb.sort()
                                        if comb not in combs:
                                            combs += [comb]
                    print 'atom indices of all amons done'
                    for comb in combs:
                        #print comb
                        cans_i = [ cans[ic] for ic in comb ]; cans_i.sort()
                        cans.append('.'.join(cans_i))
                        ts_i = [ Mlis[ic] for ic in comb ]
                        Mlis.append( merge(ts_i) )
                    print 'amons now ready for filtering'
            #else:
            #    #


            ncan = len(cans)
            # now remove redundancy
            if wg:
                #print ' cans = ', cans
                for i in range(ncan):
                    #print '** ', cans[i], (np.array(Mlis[i][0]) > 1).sum(),\
                    #                       len(Mlis[i][0]), Mlis[i][0]
                    seta.update(ir, cans[i], Mlis[i])
                seta._sort()
            else:
                for i in range(ncan):
                    #print ' ++ i, cans[i] = ', i,cans[i]
                    seta.update2(ir, cans[i], Mlis[i])
                seta._sort2()
            print 'amons are sorted and regrouped'

        cans = seta.cans; ncs = seta.ncs; nhas = seta.nhas

        ncan = len(cans)
        self.cans = cans
        if not wsmi: return
        nd = len(str(ncan))

        s1 = 'EQ' if opr == '.eq.' else ''
        svdw = '_vdw%d'%k2 if ivdw else ''
        scomb = '_comb2' if nmaxcomb == 2 else ''
        sthresh = '_dE%.2f'%thresh if thresh > 0 else ''
        if prefix == '':
            fdn = 'g%s%d%s%s_covL%d%s'%(s1,k,svdw,sthresh,covPLmin,scomb)
        else:
            fdn = prefix

        if not os.path.exists(fdn): os.system('mkdir -p %s'%fdn)
        self.fd = fdn

        if iat is not None:
            fdn += '_iat%d'%iat # absolute idx
        if wg and (not os.path.exists(fdn+'/raw')): os.system('mkdir -p %s/raw'%fdn)
        with open(fdn + '/' + fdn+'.smi', 'w') as fid:
            fid.write('\n'.join( [ '%s %d'%(cans[i],ncs[i]) for i in range(ncan) ] ) )
        dd.io.save('%s/maps.h5'%fdn, {'maps': maps} )

        if wg:
            ms = seta.ms; ms0 = seta.ms0;
            for i in range(ncan):
                ms_i = ms[i]; ms0_i = ms0[i]
                nci = ncs[i]
                labi = '0'*(nd - len(str(i+1))) + str(i+1)
                print ' ++ %d %06d/%06d %60s %3d'%(nhas[i], i+1, ncan, cans[i], nci)
                for j in range(nci):
                    f_j = fdn + '/frag_%s_c%05d'%(labi, j+1) + '.sdf'
                    f0_j = fdn + '/raw/frag_%s_c%05d_raw'%(labi, j+1) + '.sdf'
                    m_j = ms_i[j]; m0_j = ms0_i[j]
                    Chem.MolToMolFile(m_j, f_j)
                    Chem.MolToMolFile(m0_j, f0_j)
            print ' -- nmi_u = ', sum(ncs)
            print ' -- ncan = ', len(np.unique(cans))
        else:
            if wsmi:
                with open(fdn + '/' + fdn+'.smi', 'w') as fid:
                    fid.write('\n'.join( [ '%s'%(cans[i]) for i in range(ncan) ] ) )



"""
Codes below were borrowed from Andrew Dalke and some changes were made to
be independent of any cheminfomatics software!

For an explanation of the algorithm see
  http://dalkescientific.com/writings/diary/archive/2011/01/10/subgraph_enumeration.html
"""

#=========================================================================

class Subgraph(object):
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

def get_nbr(ia, b):
    ia1, ia2 = b
    if ia == ia1:
        return ia2
    else:
        return ia1

def find_extensions(considered, new_atoms, b2a, a2b):
    # Find the extensions from the atoms in 'new_atoms'.
    # There are two types of extensions:
    #
    #  1. an "internal extension" is a bond which is not in 'considered'
    # which links two atoms in 'new_atoms'.
    #
    #  2. an "external extension" is a (bond, to_atom) pair where the
    # bond is not in 'considered' and it connects one of the atoms in
    # 'new_atoms' to the atom 'to_atom'.
    #
    # Return the internal extensions as a list of bonds and
    # return the external extensions as a list of (bond, to_atom) 2-ples.
    internal_extensions = set()
    external_extensions = []
    #print 'type, val = ', type(new_atoms), new_atoms
    for atom in new_atoms: # atom is atom_idx
        ibsc = a2b[atom] # idxs of bond candidates
        for outgoing_bond in ibsc[ ibsc >= 0 ]: #atom.GetBonds():
            if outgoing_bond in considered:
                continue
            other_atom = get_nbr(atom, b2a[outgoing_bond]) #outgoing_bond.GetNbr(atom)
            if other_atom in new_atoms:
                # This this is an unconsidered bond going to
                # another atom in the same subgraph. This will
                # come up twice, so prevent duplicates.
                internal_extensions.add(outgoing_bond)
            else:
                external_extensions.append( (outgoing_bond, other_atom) )

    return list(internal_extensions), external_extensions



def all_combinations(container):
    "Generate all 2**len(container) combinations of elements in the container"
    # This just sets up the underlying call
    return _all_combinations(container, len(container)-1, 0)

def _all_combinations(container, last, i):
    # This does the hard work recursively
    if i == last:
        yield []
        yield [container[i]]
    else:
        for subcombinations in _all_combinations(container, last, i+1):
            yield subcombinations
            yield [container[i]] + subcombinations

## I had an optimization that if limit >= len(external_extensions) then
## use this instead of the limited_external_combinations, but my timings
## suggest the result was slower, so I went for the simpler code.

#def all_external_combinations(container):
#    "Generate all 2**len(container) combinations of external extensions"
#    for external_combination in all_combinations(container):
#        # For each combination yield 2-ples containing
#        #   {the set of atoms in the combination}, [list of external extensions]
#        yield set((ext[1] for ext in external_combination)), external_combination

def limited_external_combinations(container, limit):
    "Generate all 2**len(container) combinations which do not have more than 'limit' atoms"
    return _limited_combinations(container, len(container)-1, 0, limit)

def _limited_combinations(container, last, i, limit):
    # Keep track of the set of current atoms as well as the list of extensions.
    # (An external extension doesn't always add an atom. Think of
    #   C1CC1 where the first "CC" adds two edges, both to the same atom.)
    if i == last:
        yield set(), []
        if limit >= 1:
            ext = container[i]
            yield set([ext[1]]), [ext]
    else:
        for subatoms, subcombinations in _limited_combinations(container, last, i+1, limit):
            assert len(subatoms) <= limit
            yield subatoms, subcombinations
            new_subatoms = subatoms.copy()
            ext = container[i]
            new_subatoms.add(ext[1])
            if len(new_subatoms) <= limit:
                yield new_subatoms, [ext] + subcombinations


def all_subgraph_extensions(subgraph, internal_extensions, external_extensions, k):
    # Generate the set of all subgraphs which can extend the input subgraph and
    # which have no more than 'k' atoms.
    assert len(subgraph.atoms) <= k

    if not external_extensions:
        # Only internal extensions (test case: "C1C2CCC2C1")
        it = all_combinations(internal_extensions)
        it.next()
        for internal_ext in it:
            # Make the new subgraphs
            bonds = frozenset(chain(subgraph.bonds, internal_ext))
            yield set(), Subgraph(subgraph.atoms, bonds)
        return

    limit = k - len(subgraph.atoms)

    if not internal_extensions:
        # Only external extensions
        # If we're at the limit then it's not possible to extend
        if limit == 0:
            return
        # We can extend by at least one atom.
        it = limited_external_combinations(external_extensions, limit)
        it.next()
        for new_atoms, external_ext in it:
            # Make the new subgraphs
            atoms = frozenset(chain(subgraph.atoms, new_atoms))
            bonds = frozenset(chain(subgraph.bonds, (ext[0] for ext in external_ext)))
            yield new_atoms, Subgraph(atoms, bonds)
        return

    # Mixture of internal and external (test case: "C1C2CCC2C1")
    external_it = limited_external_combinations(external_extensions, limit)
    it = product(all_combinations(internal_extensions), external_it)
    it.next()
    for (internal_ext, external) in it:
        # Make the new subgraphs
        new_atoms = external[0]
        atoms = frozenset(chain(subgraph.atoms, new_atoms))
        bonds = frozenset(chain(subgraph.bonds, internal_ext,
                                (ext[0] for ext in external[1])))
        yield new_atoms, Subgraph(atoms, bonds)
    return

def generate_subgraphs(b2a, a2b, k=5):
    if k < 0:
        raise ValueError("k must be non-negative")

    # If you want nothing, you'll get nothing
    if k < 1:
        return

    # Generate all the subgraphs of size 1
    na = len(a2b)
    for atom in range(na): #mol.GetAtoms():
        yield Subgraph(frozenset([atom]), frozenset())

    # If that's all you want then that's all you'll get
    if k == 1:
        return

    # Generate the intial seeds. Seed_i starts with bond_i and knows
    # that bond_0 .. bond_i will not need to be considered during any
    # growth of of the seed.
    # For each seed I also keep track of the possible ways to extend the seed.
    seeds = []
    considered = set()
    nb = len(b2a)
    for bond in range(nb): #mol.GetBonds():
        considered.add(bond)
        subgraph = Subgraph(frozenset(b2a[bond]), #[bond.GetBgn(), bond.GetEnd()]),
                            frozenset([bond]))
        yield subgraph
        internal_extensions, external_extensions = find_extensions(considered,
                                                   subgraph.atoms, b2a, a2b)
        # If it can't be extended then there's no reason to keep track of it
        if internal_extensions or external_extensions:
            seeds.append( (considered.copy(), subgraph,
                           internal_extensions, external_extensions) )

    # No need to search any further
    if k == 2:
        return

    # seeds = [(considered, subgraph, internal, external), ...]
    while seeds:
        considered, subgraph, internal_extensions, external_extensions = seeds.pop()

        # I'm going to handle all 2**n-1 ways to expand using these
        # sets of bonds, so there's no need to consider them during
        # any of the future expansions.
        new_considered = considered.copy()
        new_considered.update(internal_extensions)
        new_considered.update(ext[0] for ext in external_extensions)

        for new_atoms, new_subgraph in all_subgraph_extensions(
            subgraph, internal_extensions, external_extensions, k):

            assert len(new_subgraph.atoms) <= k
            yield new_subgraph

            # If no new atoms were added, and I've already examined
            # all of the ways to expand from the old atoms, then
            # there's no other way to expand and I'm done.
            if not new_atoms:
                continue

            # Start from the new atoms to find possible extensions
            # for the next iteration.
            new_internal, new_external = find_extensions(new_considered, new_atoms, b2a, a2b)
            if new_internal or new_external:
                seeds.append( (new_considered, new_subgraph, new_internal, new_external) )


## test!

if __name__ == "__main__":
    import time, sys, gzip

    args = sys.argv[1:]
    nargs = len(args)
    if nargs == 0:
        ss = ["[NH3+]CC(=O)[O-]", "CC[N+]([O-])=O", \
             "C=C=C=CC=[N+]=[N-]", "CCS(=O)(=O)[O-]", \
             "C#CS(C)(=C=C)=C=C", "C1=CS(=S(=O)=O)(=S(=O)=O)C=C1", \
             "C#P=PP(#P)P(#P)P=P#P", \
             "c1ccccc1S(=O)(=O)S(=O)(=N)S(=O)(=O)c2ccccc2"] # test molecules
        k = 7
    elif nargs == 1:
        ss = args[0:1]
        k = 7
    elif nargs == 2:
        ss = args[1:2]
        k = int(args[0])
    else:
        raise SystemExit("""Usage: dfa_subgraph_enumeration.py <smiles> [<k>]
List all subgraphs of the given SMILES up to size k atoms (default k=5)
""")

    for s in ss:
        print '\n ## %s'%s
        if not os.path.exists(s):
            if s in ["C#P=PP(#P)P(#P)P=P#P",]:
                print '  ** Problematic!! Openbabel cannot obtain the correct valence for atom like P in C#P=PP(#P)C'
            t1 = time.time()
            obj = amon(s, k)
            cans = obj.get_amons()
            for can in cans:
                print can
            t2 = time.time()
            print ' time elapsed: ', t2-t1
        else:
            assert s[-3:] == 'smi'

            fn = s[:-4]
            ts = file(s).readlines()

            icnt = 0
            ids = []
            for i,t in enumerate(ts):
                si = t.strip()
                print i+1, icnt+1, si
                if '.' in si: continue
                obj = ciao.amon(si, k)
                if not obj.iok: print ' ** radical **'; continue
                print '  ++ '
                cansi = obj.get_amons()
                print '  +++ '
                nci = len(cansi)
                map_i = []
                for ci in cansi:
                    if ci not in cs:
                        cs.append(ci); map_i += [idxc]; idxc += 1
                    else:
                        jdxc = cs.index(ci)
                        if jdxc not in map_i: map_i += [jdxc]
                print 'nci = ', nci
                map_i += [-1,]*(nmaxc-nci)
                maps.append( map_i )
                #nmaxc = max(nmaxc, nci)
                ids.append( i+1 )
                icnt += 1

            with open(fn+'_all.smi','w') as fo: fo.write('\n'.join(cs))
            cs = np.array(cs)

            maps = np.array(maps,np.int)
            ids = np.array(ids,np.int)
            dd.io.save(fn+'.h5', {'ids':ids, 'cans':cs, 'maps':maps})
