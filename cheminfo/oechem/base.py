
from openeye import *
from openeye.oechem import *
import networkx as nx
import itertools as itl
import scipy.spatial.distance as ssd

import multiprocessing

import numpy as np
import ase.io as aio
import ase.data as ad
import os, sys, re, copy
import ase, openeye

import cheminfo.math as cim

global Rdic, Cdic, Rdic_Z, Cdic_Z, dic_fmt, cnsDic
Cdic = {'H':1, 'Be':2, 'B':3, 'C':4, 'N':5, 'O':6, 'F':7, \
        'Si':4, 'P':5, 'S':6, 'Cl':7, 'Ge':4, 'As':5, 'Se':6, \
        'Br':7, 'I':7}
Rdic = {'H':1, 'Be':2, 'B':2, 'C':2, 'N':2, 'O':2, 'F':2, \
        'Si':3, 'P':3, 'S':3, 'Cl':3, 'Ge':4, 'As':4, 'Se':4,\
        'Br':4, 'I':5}
Cdic_Z = {1:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 14:4, 15:5, 16:6,\
          17:7, 32:4, 33:5, 34:6, 35:7, 53:7}
Rdic_Z = {1:1, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 14:3, 15:3, 16:3,\
          17:3, 32:4, 33:4, 34:4, 35:4, 53:5}
dic_fmt = {'sdf': OEFormat_SDF, 'pdb': OEFormat_PDB, \
           'mol': OEFormat_MDL}
cnsDic = {5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 17:1}



class Groups(object):
    def __init__(self, ids=None):
        """
        The 5 integers correspond to

        vi-nhi  CN  pi_1  pi_2   Z
        --------------------------
        4       1   0     0      7  >NH
        5       3   0     2      7  -N(=O)=O, >N-C(=O)-
        5       2   0     1      7  =N-
        5       1   0     2      7  #N
        5       2   1     2      7  =N#
        5       2   0     3      7  -N$


        from left to right, respectively

        #id = (vi-nhi)*1000000 + CN*100000 + pi_1*10000 + pi_2*1000 + zi
        #   = 3130007
        """
        obsolete = """self.gs0 = {4103006: u'$C',
                    4211006: u'=C=',
                    4202006: u'#C-',
                    3102006: u'#CH',
                    4301006: u'=C<',
                    3201006: u'=CH-',
                    2101006: u'=CH2',
                    4400006: u'>C<',
                    3300006: u'>CH-',
                    2200006: u'>CH2',
                    1100006: u'-CH3',
                    5102007: u'#N',
                    5201007: u'=N-',
                    5212007: u'#N=',
                    5203007: u'$N-',
                    5302007: u'-N(=)=',
                    #5301007: u'>N-C(=O)-, >N-Ph, >N-CR=',
                    5300007: u'-N<',
                    4101007: u'=NH',
                    4200007: u'>NH',
                    3100007: u'-NH2',
                    5400007: u'>[N+]<',
                    4300007: u'>[NH+]-',
                    3200007: u'>[NH2+]',
                    2100007: u'-[NH3+]',
                    6101008: u'=O',
                    6100008: u'-[O-]',
                    6200008: u'>O',
                    #6201008: u'-O-C(=O)-, -O-Rh, ...',
                    6211008: u'>aO',
                    5100008: u'-OH',
                    5200008: u'-[OH+]-',
                    4100008: u'-[OH2+]',
                    6411016: u'>S(=)=',
                    6301016: u'>S=',
                    6101016: u'=S',
                    6100016: u'-[S-]',
                    6200016: u'>S',
                    6211016: u'>aS',
                    5100016: u'-SH',
                    5200016: u'-[SH+]-',
                    4100016: u'-[SH2+]',
                    7100009: u'-F',
                    7100017: u'-Cl',
                    7100035: u'-Br',
                    7100043: u'-I',
                    3300005: u'>B-',}"""

        self.gs0 = { '[4,1,0,3,4]': '$C',
                 '[4,2,1,1,4]': '=C=',
                 '[4,2,0,2,4]': '#C-',
                 '[3,1,0,2,4]': '#CH',
                 '[4,3,0,1,4]': '=C<',
                 '[4,3,0.5,0.5,4]': '=aC<',
                 '[3,2,0,1,4]': '=CH-',
                 '[3,2,0.5,0.5,4]': '=aCH-',
                 '[2,1,0,1,4]': '=CH2',
                 '[4,4,0,0,4]': '>C<',
                 '[3,3,0,0,4]': '>CH-',
                 '[2,2,0,0,4]': '>CH2',
                 '[1,1,0,0,4]': '-CH3',
                 '[5,1,0,2,5]': '#N',
                 '[5,2,0,1,5]': '=N-',
                 '[5,2,0.5,0.5,5]': '=aN-',
                 '[5,2,1,2,5]': '#N=',
                 '[5,2,0,3,5]': '$N-',
                 '[5,3,0,2,5]': '-N(=)=',
                 '[5,3,0,0,5]': '-N<',
                 '[5,3,0.5,0.5,5]': '-aN<',
                 '[4,1,0,1,5]': '=NH',
                 '[4,2,0,0,5]': '>NH',
                 '[4,2,0.5,0.5,5]': '>aNH',
                 '[3,1,0,0,5]': '-NH2',
                 '[5,4,0,0,5]': '>[N+]<',
                 '[4,3,0,0,5]': '>[NH+]-',
                 '[3,2,0,0,5]': '>[NH2+]',
                 '[2,1,0,0,5]': '-[NH3+]',
                 '[6,1,0,1,6]': '=O',
                 '[6,1,0,0,6]': '-[O-]',
                 '[6,2,0,0,6]': '>O',
                 '[6,2,1,1,6]': '>aO',
                 '[5,1,0,0,6]': '-OH',
                 '[6,4,1,1,2.67]': '>S(=)=',
                 '[6,3,0,1,2.67]': '>S=',
                 '[6,1,0,1,2.67]': '=S',
                 '[6,1,0,0,2.67]': '-[S-]',
                 '[6,2,0,0,2.67]': '>S',
                 '[6,2,1,1,2.67]': '>aS',
                 '[5,1,0,0,2.67]': '-SH',
                 '[7,1,0,0,9]': '-F',
                 '[7,1,0,0, 3.11]': '-Cl',
                 '[7,1,0,0, 1.75]': '-Br',
                 '[7,1,0,0, 1.12]': '-I',
                 '[3,3,0,0, 3]': '>B-',}

        self.gs = []
        if ids != None:
            for idx in list(ids):
                assert idx in self.gs0.keys(), '#ERROR: new group env?'
                self.gs.append( self.gs0[idx] )

class StringsM(object):
    """
    The plural form of `class StringM()
    It enables parallel processing !
    """
    def __init__(self, strings, igroup=False, iPL=False, nproc=1):
        self.n = len(strings)
        if nproc == 1:
            self.objs = []
            for i,string in enumerate(strings):
                #print(i+1, string)
                ipt = [string, igroup, iPL]
                self.objs.append( self.processInput(ipt) )
        else:
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ [string,igroup,iPL] for string in strings ]
            self.objs = pool.map(self.processInput, ipts)

    def processInput(self, ipt):
        string, igroup, iPL = ipt
        obj = StringM(string, igroup=igroup, iPL=iPL)
        return obj

    def get_ET(self, igroup2):
        """
        get electro-topological info
        Types of functional groups (if `igroup2 = True)
        or simply nuclear charges (if `igroup2 = False) would
        be part of the returned vars

        Note that `igroup2 here must be manually set, can be
        different compared to `igroup set at the very beginning
        """
        zs = []
        zsr = []
        nas = []
        nhass = []
        zsu = set([])
        coords = []
        for i in range(self.n):
            zsi = []
            obj_i = self.objs[i]
            coords.append( obj_i.coords )
            if igroup2:
                Gi = obj_i.groups
                keys = Gi.keys(); keys.sort()
                for key in keys:
                    vals = Gi[key]
                    zsi.append(vals[0])
            else:
                zsi = obj_i.zs
            nhass.append( (np.array(zsi)>1).sum() )
            nas.append( len(zsi) )
            print(' -- zsi = ', zsi)
            zsu.update( zsi )
            zsr += list(zsi)
            zs.append( zsi )
        zsu = list(zsu)
        nzu = len(zsu)
        zsu.sort()
        nzs = np.zeros((self.n, nzu), np.int)
        for i in range(self.n):
            for iz in range(nzu):
                ioks = np.array(zs[i]) == zsu[iz]
                nzs[i,iz] = np.sum(ioks)
        self.nzs = nzs
        self.zsu = zsu
        self.zs = zs
        self.zsr = np.array(zsr,np.int)
        self.nas = np.array(nas,np.int)
        self.nhass = np.array(nhass,np.int)
        self.coords = coords








    def fit_property(self, yfile, N1s, fix_seed=False):
        """
        fit property of molecules in test set trained
        on `N1 molecules randomly chosen

        A simple linear model will be used:

            y = \sum_i n_i * y_i
        """
        self.get_ET( self.igroup )
        ys = np.loadtxt(yfile)
        if fix_seed:
            np.random.seed(2) # fix the random sequence
        tidxs = np.random.permutation(self.n)
        ngs_u = self.ngs[tidxs]
        ys_u = ys[tidxs]
        for N1 in N1s:
            N2 = self.n - N1
            x1 = ngs_u[:N1]; x2 = ngs_u[N1:]
            y1 = ys_u[:N1]; y2 = ys_u[N1:]
            bases = np.linalg.lstsq(x1,y1)[0]
            errors = y2 - np.dot(x2, bases)
            mae = np.sum(np.abs(errors))/N2
            rmse = np.sqrt( np.sum( errors**2 )/N2 )
            print(' %d %.2f %.2f'%(N1, mae, rmse))

def write(m, f):
    fmt = f[-3:]
    m.SetDimension(3)
    ofs = oemolostream(f)
    ofs.SetFormat( dic_fmt[fmt] )
    OEWriteMolecule(ofs, m)

def sdf2oem(sdf):
    ifs = oemolistream()
    iok = ifs.SetFormat( dic_fmt[ sdf[-3:] ] )
    iok = ifs.open(sdf)
    for m in ifs.GetOEGraphMols():
        break
    return m

def smi2oem(smi, addh = False):
    m = OEGraphMol()
    iok = OESmilesToMol(m,smi)
    assert iok, '#ERROR: parsing SMILES failed!'
    if addh:
        OEAddExplicitHydrogens(m)
    else:
        OESuppressHydrogens(m, False, False, False)
    return m

def vang(u,v):
    cost = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
# sometimes, cost might be 1.00000000002, then np.arccos(cost)
# does not exist!
    u = cost if abs(cost) <= 1 else 1.0
    return np.arccos( u )*180/np.pi


class StringM(object):

    """
    build molecule object with a string (SMILES or sdf file) as input
    """

    def __init__(self, string, neutral=True, suppressH=False, ImpHCount=False, \
                 purge_charge=True, igroup=False, iPL=False, debug=False):
        import openeye

        self.suppressH = suppressH
        self.ImpHCount = ImpHCount
        self.purge_charge = purge_charge
        self.debug = debug

        # is the atom order in `can the same as that in sdf file?
        self.sorted = False
        self.ismi = False

        #if '/' in string:
        #    assert os.path.exists(string), '#ERROR: file does not exist'
        st = type(string)
        if st is str:
          if os.path.exists(string):
            self.ismi = True
            if string.endswith( ('sdf','mol','pdb') ):
                m = sdf2oem(string)
                if suppressH:
                    OESuppressHydrogens(m, False, False, False)
            elif string.endswith( 'xyz' ):
                obj = Mol(string)
                # note that `once=True makes sure that only the saturated
                # molecule return (e.g., for an input smiles cccc, C=CC=C
                # is the only output; with `once .eq. False, CC=CC would be
                # another output)
                obj.perceive_bond_order(neutral=neutral, once=True, debug=debug)
                m = obj.oem #; print('_____', m)
            else:
                raise '#ERROR: file type not supported'
            if suppressH:
                OESuppressHydrogens(m, False, False, False)
          else: # presumably a SMILES string
            m = smi2oem(string)
            if not suppressH:
                iok = OEAddExplicitHydrogens(m)
        elif st is ase.Atoms:
          obj = Mol( string ); #print('_______ g = ', obj.G)
          obj.perceive_bond_order(neutral=neutral, once=True, debug=debug)
          m = obj.oem
        elif st is openeye.oechem.OEGraphMol:
          m = string
        else:
          raise '#ERROR: input `string not supported'

        self.oem = m
        self.na = m.NumAtoms()
        #self.atoms = [ ak for ak in m.GetAtoms() ]
        self.zs = np.array( [ ai.GetAtomicNum() for ai in m.GetAtoms() ] ).astype(np.int)
        self.update()
        if self.purge_charge:
            self.annihilate_charges()
        self.can = to_can(self.oem, ImpHCount=ImpHCount, reconstruct=False)
        if igroup: self.to_groups()
        if iPL: self.get_PathLength()

    def get_component(self,i):
        return find_cliques(self.bom)[i]

    def copy(self):
        mc = OEGraphMol()
        OECopyMol(mc, self.oem)
        return mc

    def clone(self):
        return self.copy()

    def get_charged_pairs(self):
        """
        get pairs of atoms with opposite charges
        """
        charges = [ ai.GetFormalCharge() for ai in self.oem.GetAtoms() ]
        # search for the pairs of atoms with smarts like '[N+](=O)[O-]'
        patt = '[+1]~[-1]'
        cpairs = np.array( is_subg(self.oem, patt, iop=1)[1] ).astype(np.int)
        self.charges = charges
        self.cpairs = cpairs

    def neutralise(self):
        """
        neutralise a molecule, typically a protein
        """
        assert self.suppressH, '#ERROR: pls set `suppressH=True'
        m = self.copy()
        obsolete = """zs = self.zs
        numHs = []; tvs = []; atoms = []
        for ai in m.GetAtoms():
            atoms.append( ai )
            numHs.append( ai.GetExplicitHCount() + ai.GetImplicitHCount() )
            tvs.append( ai.GetValence() )

        self.get_charged_pairs()
        for i in range(self.na):
            ai = atoms[i]
            ci = self.charges[i]
            nhi = numHs[i]
            if ci != 0:
                if i not in self.cpairs.ravel():
                    msg = ' zi = %d, tvi = %d, ci = %d, neib = %d'%(self.zs[i], tvs[i], ci, cnsDic[zs[i]])
                    assert tvs[i] - ci == cnsDic[zs[i]], msg
                    if nhi == 0 and ci > 0:
                        # in the case of >[N+]<, i.e., N with CoordNum = 4
                        # we don't have to do anything
                        continue
                    ai.SetFormalCharge( 0 )
                    ai.SetImplicitHCount( nhi - ci )
                    print('i, zi, ci, nH = ', i, self.zs[i], ci, numHs[i])
                else:
                    print('atom %d in a bond like ~[A+]~[B-]'%i) """
        OERemoveFormalCharge(m) # note that for quaternary amines (>[N+]<), charges retain
        OEAddExplicitHydrogens(m)
        OESet3DHydrogenGeom(m)
        # note that H attached to sp2-hybridized and charged N, e.g., N in =[NH2+] won't
        # be removed by OERemoveFormalCharge(), now we do this manually
        for ai in m.GetAtoms():
            chgi = ai.GetFormalCharge()
        self.oem = m
        self.can = to_can(m)

    def update_coords_COOH(self):
        return

    def get_neighbors(self, ai, ishow=False): #iac):
        """
        To diagnosis of unreasonable local atomic env
        """
        dic0 = {1:1,6:4,7:3,8:2,16:2}
        #atoms = self.atoms
        #ai = atoms[iac]
        ia = ai.GetIdx()
        zi = ai.GetAtomicNum()
        vi = ai.GetValence()
        hi = ai.GetImplicitHCount()
        hi2 = ai.GetExplicitHCount()
        if ishow or (vi != dic0[zi]):
            print(' -center ', ia, zi, vi, hi, hi2)
            for bj in ai.GetBonds():
                aj = bj.GetNbr(ai)
                ja = aj.GetIdx(); zj = aj.GetAtomicNum(); vj = aj.GetValence(); hj = aj.GetImplicitHCount(); hj2 = aj.GetExplicitHCount()
                print(' -nbrs  ', ja,zj,vj,hj,hj2, ', BO = ', bj.GetOrder())


    def get_PathLength(self):
        """
        get graph PathLength between atoms
        """
        oem_u = copy.deepcopy(self.oem)
        OESuppressHydrogens(oem_u, False, False, False)
        nha = oem_u.NumAtoms()
        PLs = np.zeros((nha,nha))
        for ai in oem_u.GetAtoms():
            ia = ai.GetIdx()
            for aj in oem_u.GetAtoms():
                ja = aj.GetIdx()
                if ja > ia:
                    PLs[ia,ja] = PLs[ja,ia] = OEGetPathLength(ai,aj)
        self.PLs = PLs
        self.nha = nha
        self.L = np.max(PLs) # maximal length of a molecule
        self.H = nha/(1.0+self.L) # height of a molecule

    def to_groups(self):
        """
        find all possible functional groups
        """
        zs = self.zs
        na = self.na
        ias = np.arange(na)

        ars = []; hybs = []
        for ai in self.oem.GetAtoms():
            ars.append( ai.IsAromatic() )
            hybs.append( OEGetHybridization(ai) )
        groups = {}
        for ia in range(na):
            zi = zs[ia]
            if zi > 1:
                vi = ci = Cdic_Z[zi]
                ri = Rdic_Z[zi]
                bosi = self.bom[ia]
                #vi = bosi.sum() - self.charges[ia]
                cni = (bosi > 0).sum()
                boi = bosi.sum()
                ias_u = ias[ bosi > 0 ]
                zsi = self.zs[ bosi > 0 ]
                hs = ias_u[ zsi == 1 ]
                nhi = len(hs)
                vi2 = vi - nhi
                cni2 = cni - nhi # coord_num_heavy_atoms

                bosi_u = bosi[ np.logical_and(bosi > 0, self.zs > 1) ]
                pis = [ ]
                for boi in bosi_u:
                    if boi >= 2:
                        pis.append( boi-1 )
                npi = len(pis)
                if npi == 0:
                    pi1, pi2 = [0, 0]
                elif npi == 1:
                    pi1 = 0; pi2 = pis[0]
                elif npi == 2:
                    pis.sort(); pi1, pi2 = pis
                else:
                    print(' -- bosi_u, pis = ', bosi_u, pis)
                    raise "#ERROR: >=2 pi bonds connected to some atom"
                iid = [vi2, cni2, pi1, pi2, eval('%.2f'%(4.0*ci/ri**2))]
                groups[ia] = [iid, list(hs)]
        self.groups = groups

    def update(self):
        """
        sometimes, in the SDF file, H atoms may appear in between
        heavy atoms, resulting in heavy atom indices being, say,
        [0,1,2,3, 5,6,7], which would cause error later like
        `IndexError: index 11 is out of bounds for axis 0 with size 11
        when calling function `annihilate_charges(). The relevant part
        of codes is:

        for bi in self.oem.GetBonds():
            i,j,boij = bi.GetBgnIdx(), bi.GetEndIdx(), bi.GetOrder()
      --->  bom[i,j] = bom[j,i] = boij
        i.e., `i or `j may be larger than `na

        This function `update() can fix the problem.
        """
        m = self.oem
        mu = OEGraphMol()
        atoms = {}; coords = []; mapping = {}
        icnt = 0
        for ai in m.GetAtoms():
            ia = ai.GetIdx()
            zi = ai.GetAtomicNum()
            if zi > 1:
                aiu = mu.NewAtom( zi )
                aiu.SetHyb( OEGetHybridization(ai) )
                aiu.SetImplicitHCount( ai.GetImplicitHCount() )
                aiu.SetFormalCharge( ai.GetFormalCharge() )
                coords_ai = m.GetCoords(ai)
                coords.append( coords_ai )
                mu.SetCoords( aiu, coords_ai )
                atoms[ icnt ] = aiu
                mapping[ia] = icnt
                icnt += 1
        for ai in m.GetAtoms():
            ia = ai.GetIdx()
            zi = ai.GetAtomicNum()
            if zi == 1:
                aiu = mu.NewAtom( zi )
                coords_ai = m.GetCoords( ai )
                coords.append( coords_ai )
                mu.SetCoords( aiu, coords_ai )
                atoms[ icnt ] = aiu
                mapping[ia] = icnt
                icnt += 1
        for bi in m.GetBonds():
            p0, q0 = bi.GetBgnIdx(), bi.GetEndIdx()
            p, q = mapping[p0], mapping[q0]
            biu = mu.NewBond( atoms[p], atoms[q], bi.GetOrder() )
        OEFindRingAtomsAndBonds(m)
        OEAssignAromaticFlags(m, OEAroModel_OpenEye)
        #OECanonicalOrderAtoms(mu)
        #OECanonicalOrderBonds(mu)
        self.oem = mu
        self.coords = np.array( coords )

    def annihilate_charges(self):
        """
        Charges in some SMILES can be totally skipped.

        _____________________________________________
         group         SMILES          purged SMILES
        -------       --------        ---------------
        -NO2        -[N+](=O)[O-]        -N(=O)=O
        >CN2        >C=[N+]=[N-]         >C=N#N
        -NC         -[N+]#[C-]           -N$C (i.e., BO(N-C) = 4)
        _____________________________________________

        In cases like [NH3+]CCC(=O)[O-], the charges retain.
        """
        na = self.na
        charges = np.array([ai.GetFormalCharge() for ai in self.oem.GetAtoms()])
        zs = np.array( [ ai.GetAtomicNum() for ai in self.oem.GetAtoms() ] )
        bom = np.zeros(( na, na), np.int)
        for bi in self.oem.GetBonds():
            i,j,boij = bi.GetBgnIdx(), bi.GetEndIdx(), bi.GetOrder()
            bom[i,j] = bom[j,i] = boij

        if np.any( charges != 0 ):
            message = '#ERROR: some atom has a charge larger than 1??'
            #assert np.all( charges <= 1 ), message
            ias = np.arange( na )
            ias1 = ias[ charges == 1 ]
            ias2 = ias[ charges == -1 ];
            bs = {} # store the idx of bonds for update of BO
            cs = {} # store the idx of atom for update of charge
            irev = False # revise `bom and `charges?
            visited = dict( zip(range(na), [False,]*na) )
            for ia1 in ias1:
                for ia2 in ias2:
                    bo12 = bom[ia1,ia2]
                    if bo12 > 0:
                        assert not (visited[ia1] or visited[ia2]), \
                                  '#ERROR: visted before?'
                        visited[ia1] = visited[ia2] = True
                        bo12 += 1
                        irev = True
                        pair = [ia1,ia2]; pair.sort()
                        bs[ tuple(pair) ] = bo12
                        bom[ia1,ia2] = bom[ia2,ia1] = bo12
                        cs[ia1] = charges[ia1] - 1
                        cs[ia2] = charges[ia2] + 1
                        charges[ia1] = cs[ia1]
                        charges[ia2] = cs[ia2]
            if irev:
                csk = cs.keys()
                for ai in self.oem.GetAtoms():
                    idx = ai.GetIdx()
                    if idx in csk:
                        ai.SetFormalCharge( cs[idx])
                bsk = bs.keys()
                for bi in self.oem.GetBonds():
                    ias12 = [bi.GetBgnIdx(), bi.GetEndIdx()]
                    ias12.sort()
                    ias12u = tuple(ias12)
                    if ias12u in bsk:
                        bi.SetOrder( bs[ias12u] )

        self.zs = np.array(zs)
        self.bom = bom
        self.charges = np.array(charges)

    def get_graph(self):
        """
        get molecular graph with bond order as weights for edges
        """
        G = np.zeros((self.na, self.na), np.int)
        for bi in self.oem.GetBonds():
            i, j, boij = bi.GetBgnIdx(), bi.GetEndIdx(), bi.GetOrder()
            G[i,j] = G[j,i] = boij
        self.G = G

    def sort_atoms(self, q=None, hse=True):
        """
        usually the sequence of atoms in sdf file is not the same as
        the SMILES converted from this very sdf file, now fix it.

        Note that the SMILES could be converted from other source,
        not necessary the same software (such as OpenBabel or RDKit,
        other than OEChem).

        Parameters
        =========================
        hse -- all HydrogenS at the End?

        """
        G0 = self.bom
        na = self.na
        smi = self.can
        m = copy.deepcopy( self.oem )
        coords = m.GetCoords()
        zs = np.array(self.zs, np.int)


        if q is None: q = smi
        ss = OESubSearch(q)
        iok = OEPrepareSearch(m, ss)
        isg = ss.SingleMatch(m)
        msg = '#ERROR: gvien SMILES %s not consistent'%q + \
                  ' with the molecule in sdf file'
        assert isg, msg
        idxs = []; idxs_aux = []
        atoms_T = [ ait for ait in m.GetAtoms() ]
        for i,match in enumerate(ss.Match(m)):
            idxsQ_i = []; idxsT_i = []; #atomsT = []
            for ma in match.GetAtoms():
                idxsQ_i.append(  ma.pattern.GetIdx() )
                idxsT_i.append( ma.target.GetIdx() )
                #atomsT.append( ma.target )
            seqs = np.argsort(idxsQ_i)
            idxsT_iu = np.array(idxsT_i, np.int)[seqs]
            atoms = [ atoms_T[iat] for iat in idxsT_iu ]
            break

        ias_u = list(idxsT_iu)
        zs_u = list( zs[ias_u] )
        coords_u = [ coords[ia] for ia in ias_u ]
        for ai in atoms:
            for aj in ai.GetAtoms():
                j = aj.GetIdx()
                if j not in idxsT_i:
                    ias_u.append(j) # i.e., H atom
                    zs_u.append(1)
                    coords_u.append( coords[j] )

        coords_u = np.array( coords_u )
        #if self.ImpHCount:
        G = np.zeros((na, na), np.int)
        for ia in range(na):
            for ja in range(ia+1, na):
                ia2 = ias_u[ia]
                ja2 = ias_u[ja]
                boij = G0[ia2, ja2]
                #if boij > 0:
                #    print(' -- ia2,ja2, ia,ja,boij = ', ia2,ja2,ia,ja,boij)
                G[ia,ja] = G[ja,ia] = boij
        self.zs = zs_u
        self.coords = coords_u
        self.G = G
        self.bom = G
        self.sorted = True
        #return zs_u, coords_u, G
        charges = np.array(self.charges)[ias_u]
        self.charges = charges
        self.oem = self.to_oem( G, charges) #, kekulize=True )

    def to_oem(self, bom, charges, with_coords=True, canonicalize=False, \
               kekulize=False):
        """
        convert bond_order_matrix to OEChem Molecule object
        """
        oem = OEGraphMol()
        atoms = []
        for i in range(self.na):
            ai = oem.NewAtom( int(self.zs[i]) )
            ci = charges[i]
            if ci != 0: ai.SetFormalCharge( ci )
            if with_coords: oem.SetCoords( ai, self.coords[i] )
            atoms.append( ai )
        for i in range(self.na):
            for j in range(i+1, self.na):
                if self.G[i,j] > 0:
                    bi = oem.NewBond( atoms[i], atoms[j], int(bom[i,j]) )
        #OEFindRingAtomsAndBonds(oem)
        OEAssignAromaticFlags(oem, OEAroModel_OpenEye)
        if kekulize:
            OEClearAromaticFlags(oem)
            OEKekulize(oem)
        if canonicalize:
            OECanonicalOrderAtoms(oem)
            OECanonicalOrderBonds(oem)
        return oem

    def perceive_DHA(self, m, tv1=0, tv2=0):
        ## first get D and H in D-H...A
        ## D could be O, N or S, and of course should be
        ## bonded to at least one H
        dic = {7:3, 8:2, 9:1, 16:2}
        DHs = []; As = []
        for ai in m.GetAtoms():
            ia = ai.GetIdx()
            zi = ai.GetAtomicNum()
            hvi = ai.GetHvyDegree()
            if zi in [7, 8, 9, 16]:
                # always be valid for Accepting nearly naked H
                As.append([ia + tv1])
                # for donating H part, it's slightly more harse
                if hvi < dic[zi]:
                    for aj in ai.GetAtomIter():
                        zj = aj.GetAtomicNum()
                        ja = aj.GetIdx()
                        if zj == 1:
                            DHs.append([ia + tv1, ja + tv2])
        return DHs, As


    def perceive_hbs(self, dmax=2.2, angmin=100, scale=1.25):
        """
        perceive hydrogen bonds in the given molecule
        """
        # a dictionary of minimal covalent radius
        rsdic = {7:0.68, 8:0.68, 9:0.64, 16:1.02, 1:0.32}
        s1, s2 = self.can.split('.')
        m1 = smi2oem(s1, addh=False)
        m2 = smi2oem(s2, addh=False)
        nHEAV_1 = m1.NumAtoms()
        nHEAV_2 = m2.NumAtoms()
        nHEAV = nHEAV_2 + nHEAV_1
        if not self.sorted:
            self.sort_atoms(hse=True)
        zs = self.zs
        coords = self.coords
        G = self.G

        # since hydrogens appear at the very end, indices of atoms have to
        # be shifted for the hydrogens in the fragment molecule #1
        assert OEAddExplicitHydrogens(m1)
        assert OEAddExplicitHydrogens(m2)
        na1 = m1.NumAtoms()
        if self.debug: print('__ na1 = ', na1)
        DHs_1, As_1 = self.perceive_DHA(m1, tv1 = 0, tv2 = nHEAV_2)
        if self.debug: print(' * DHs_1, As_1 = ', DHs_1, As_1)
        DHs_2, As_2 = self.perceive_DHA(m2, tv1 = nHEAV_1, tv2 = na1)
        if self.debug: print(' * DHs_2, As_2 = ', DHs_2, As_2)

        hbs0 = []
        for (a,b) in itl.product(DHs_1, As_2): hbs0.append(a+b)
        for (a,b) in itl.product(DHs_2, As_1): hbs0.append(a+b)
        ds = ssd.squareform(ssd.pdist(coords))
        hbs = []
        for t in hbs0:
            i,j,k = t
            djk = ds[j,k]
            v1 = coords[i] - coords[j]; v2 = coords[k] - coords[j]
            dmin = scale*( rsdic[zs[j]] + rsdic[zs[k]] )
            iok1 = ( djk < dmax )
            iok2 = ( djk > dmin )
            #print ' ++ djk, dmin, dmax = ', djk, dmin, dmax
            iok3 = ( vang(v1,v2) > angmin )
            if iok1 and iok2 and iok3:
                hbs.append( [i,k] ) # donor and acceptor heavy atom
        return hbs

    def perceive_non_covalent_bonds(self, dminVDW=1.2, covPLmin=7, scale=1.1):
        # note that the thus-obtained bonds include hydrogen bonds
        # from the function `perceive_hbs()
        # The criteria of assigning a non covalent bond is that the
        # interatomic distance is .le. sum of vdw radii, and that there
        # are at least 5 heavy atoms connecting these two end atoms
        ncbs = []

        rsvdw = ad.vdw_radii
        m = self.oem
        #assert OEAddExplicitHydrogens(m)

        ds = ssd.squareform( ssd.pdist(self.coords) )
        bom = self.bom
        na = self.na
        ias = np.arange(na)
        for i in range(na):
            for j in range(i+1,na):
                zi = self.zs[i]; zj = self.zs[j]
                #nH = (np.array([zi,zj]) == 1).sum()
                dmaxVDW = scale*(rsvdw[zi]+rsvdw[zj])
                dij = ds[i,j]
                if bom[i,j] == 0 and dij < dmaxVDW and dij > dminVDW:
                    a1 = m.GetAtom( OEHasAtomIdx(i) )
                    a2 = m.GetAtom( OEHasAtomIdx(j) )
                    joints = [ a3 for a3 in OEShortestPath(a1,a2) ]
                    n = len(joints)
                    zs = np.array([ joints[k].GetAtomicNum() for k in range(n) ])
                    nheav = (zs > 1).sum()
                    if n == 0 or nheav >= covPLmin:
                        # nheav = 0 implies that there are two or more
                        # components in the system and these two atoms
                        # are not in the same molecule
                        iu = i
                        if zi == 1:
                            iu = ias[ bom[i] == 1 ][0]
                        ju = j
                        if zj == 1:
                            ju = ias[ bom[j] == 1 ][0]

                        if iu != ju and bom[iu,ju] == 0:
                            pair = [iu,ju]; pair.sort()
                            if pair not in ncbs: ncbs.append(pair)
        return ncbs

    def get_nodes_of_rings(self):
        """
        get extended smallest set of small rings,

        i.e., get all the lists of atom indices for
              rings with size ranging from 3 to 9
        """
        m = self.m
        namin = 3
        namax = 9
        sets = []
        for i in range(namin, namax+1):
            pat_i = '*~1' + '~*'*(i-2) + '~*1'
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                set_i = set()
                for ma in match.GetAtoms():
                    set_i.update( [ma.target.GetIdx()] )
                if set_i not in sets: sets.append( set_i )
        # now remove those rings that are union of smaller rings
        n = len(sets)
        sets_remove = []
        ijs = itl.combinations( range(n), 2 )
        sets_u = []
        for i,j in ijs:
            set_ij = sets[i].union( sets[j] )
            if set_ij in sets and (set_ij not in sets_remove):
                sets_remove.append( set_ij )
        sets_u = cim.get_compl(sets, sets_remove)
        return [ list(si) for si in sets_u ]

    def get_strained(self):
        # get a list of strains for each atom
        strains = []
        for ai in self.oem.GetAtoms():
            istrain = False
            if OEAtomIsInRingSize(ai, 3) or OEAtomIsInRingSize(ai, 4):
                istrain = True
            strains.append(istrain)
        return np.array(strains)


class GraphM(object):

    def __init__(self, g):
        self.nn = g.shape[0] # number_of_nodes
        self.n = self.nn
        g1 = (g > 0).astype(np.int)
        np.fill_diagonal(g1, 0)
        self.g1 = g1
        self.ne = g1.sum()/2 # number_of_edges
        self.edges = [ list(edge) for edge in \
             np.array( list( np.where(np.triu(g)>0) ) ).T ]

    def is_connected(self):
        return self.ne - self.n + 1 >= 0


class MolBase(object):

    def __init__(self, Obj):
        """
        Three types of `obj as input are possible:
        1) ase.Atoms
        2) openeye.oechem.OEGraphMol
        3) XYZ file (string type)
        """
        if type(Obj) is openeye.oechem.OEGraphMol:
            oem = Obj
            #OESuppressHydrogens(oem, False, False, False)
            self.G = ( oem2g(oem) > 0 ).astype(np.int)
            #self.oem = oem
            self.na = oem.NumAtoms()
            self.ias = np.arange(self.na)
            dicc = oem.GetCoords()
            self.coords = np.array([ dicc[i] for i in range(self.na) ])
            self.zs = np.array([ ai.GetAtomicNum() \
                          for ai in oem.GetAtoms() ], np.int)
        else:
            if type(Obj) == ase.Atoms:
                atoms = Obj
            elif type(Obj) is str:
                if os.path.exists(Obj):
                    atoms = aio.read(Obj)
            else:
                raise '#ERROR: input type not allowed'
            self.coords = atoms.positions
            self.zs = atoms.numbers
            self.symbols = [ ai.symbol for ai in atoms ]
            self.na = len(atoms)
            self.ias = np.arange(self.na)
            self.perceive_connectivity()

    def perceive_connectivity(self):
        """
        obtain molecular graph from geometry __ONLY__
        """

        # `covalent_radius from OEChem, for H reset to 0.32
        crs = np.array([0.0, 0.23, 0.0, 0.68, 0.35, 0.83, \
           0.68, 0.68, 0.68, 0.64, 0.0, 0.97, 1.1, 1.35, 1.2,\
            1.05, 1.02, 0.99, 0.0, 1.33, 0.99, 1.44, 1.47, \
           1.33, 1.35, 1.35, 1.34, 1.33, 1.5, 1.52, 1.45, \
           1.22, 1.17, 1.21, 1.22, 1.21, 0.0, 1.47, 1.12, \
           1.78, 1.56, 1.48, 1.47, 1.35, 1.4, 1.45, 1.5, \
           1.59, 1.69, 1.63, 1.46, 1.46, 1.47, 1.4, 0.0, \
           1.67, 1.34, 1.87, 1.83, 1.82, 1.81, 1.8, 1.8, \
           1.99, 1.79, 1.76, 1.75, 1.74, 1.73, 1.72, 1.94, \
           1.72, 1.57, 1.43, 1.37, 1.35, 1.37, 1.32, 1.5,\
           1.5, 1.7, 1.55, 1.54, 1.54, 1.68, 0.0, 0.0, 0.0, \
           1.9, 1.88, 1.79, 1.61, 1.58, 1.55, 1.53, 1.51, 0.0, \
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0] )

        ps = self.coords
        zsU = np.array( list( set(self.zs) ), np.int )
        rs = np.zeros(self.na)
        for zi in zsU:
            rs[ zi == self.zs ] = crs[ zi ]

        ratio = 1.25
        rs1, rs2 = np.meshgrid(rs,rs)
        dsmax = (rs1 + rs2)*ratio
        ds = np.sqrt((np.square(ps[:,np.newaxis]-ps).sum(axis=2)))
        G0 = np.logical_and( ds > 0., ds <= dsmax )
        G = G0.astype(np.int)
        #print ' -- G = '
        #print G

        self.G = G
        self.ds = ds
        #return G #, ds, dsmax


class Mol(MolBase):

    def __init__(self, Obj):
        MolBase.__init__(self, Obj)

    def to_oem(self, bom, charges, with_coords=True, \
               canonicalize=False, kekulize=False):
        """
        convert bond_order_matrix to OEChem Molecule object
        """
        oem = OEGraphMol()
        atoms = []
        for i in range(self.na):
            ai = oem.NewAtom( int(self.zs[i]) )
            ci = charges[i]
            if ci != 0: ai.SetFormalCharge( ci )
            if with_coords: oem.SetCoords( ai, self.coords[i] )
            atoms.append( ai )
        for i in range(self.na):
            for j in range(i+1, self.na):
                if self.G[i,j] > 0:
                    bi = oem.NewBond( atoms[i], atoms[j], int(bom[i,j]) )
        OEFindRingAtomsAndBonds(oem)
        if kekulize:
            OEClearAromaticFlags(oem)
            OEKekulize(oem)
        if canonicalize:
            OECanonicalOrderAtoms(oem)
            OECanonicalOrderBonds(oem)
        return oem


    def update_double_bond_chain(self, bom, dvs):
        """
        correct bond order values for cases like
        >C-C-C<
        >C-C-N-
        >C-C-C-C-C<
        ...

        steps
        1) find out the indices of atoms with dv .eq. 2
        2) perceive cliques of the above atoms (clique: connected subgraph)
        3) for each clique, see if the two neighboring atoms at the two ends
           (since atoms with dv .eq. 2 form a linear chain) have dv .eq. 1
        4) for every valid clique, resort the indices of atoms so that there
           is 1-to-1 correspondence between atom (in a chain) and its index
        5) reassign bond orders like 1-(2)_n-1, where `n is the number of
           atoms with `dv .eq. 2 in the clique
        """

        ias = self.ias

        filt = (dvs == 2)
        ias2 = ias[filt]
        nv = filt.sum()
        if nv >= 1:
            if nv == 1:
                cliques = [[0], ]
            else:
                g2 = bom[ias2,:][:,ias2]
                cliques = find_cliques(g2)

            for sgi in cliques:
                nni = len(sgi)
                ias2_u = ias2[sgi]
                neibs = find_neighbors(bom, ias2_u)
                nneibs = len(neibs)
                #assert nneibs == 2, '#ERROR: more than 2 neighboring atoms?'
                if nneibs == 2 and np.all( [ dvs[ia] == 1 for ia in neibs ] ):
                    ias3_u = [ neibs[0], ]
                    icnt = 0
                    while True:
                        if icnt == nni: break
                        ja = ias3_u[icnt]
                        for neib in ias[ bom[ja] > 0 ]:
                            if (neib not in ias3_u) and (neib in ias2_u):
                                ias3_u.append( neib )
                                icnt += 1
                    ias3_u.append( neibs[1] )
                    for k in range(nni+1):
                        ka1 = ias3_u[k]
                        ka2 = ias3_u[k+1]
                        bom[ka1,ka2] = bom[ka2,ka1] = 2
        return bom

    def update_charges(self, bom0, cns0, dvs0, neutral=True):
        """
        update charges for cases like amine acids
        """
        ias = self.ias
        zs = self.zs
        cns = copy.copy(cns0)
        bom = copy.copy(bom0)
        dvs = copy.copy(dvs0)
        #print ' -- dvs = ', dvs
        netc = sum(dvs)
        if netc != 0:
            # there must be a N group like -[NH3+], with N assigned a valence of 5
            iaN = ias[ np.logical_and(np.array(self.zs)==7, np.array(dvs)==1) ]
            if len(iaN) > 1:
                # case 2, CH3-N(=O)-N(=O)-CH3 with dvs = [0, 1, 1, 0]            _             _
                # case 1, C[NH3+](C(=O)[O-])-C-C[NH3+](C(=O)[O-])
                #         with dvs = [0, 1, 0,0,1,0,0, 1, 0,0,1]
                cliques = find_cliques(bom[iaN,:][:,iaN])
                for clique in cliques:
                    if len(clique) == 1: # case 1, fix `bom
                        ia = clique[0]
                        cns[ia] = 3
                    else: # case 2, fix `cns
                        ia,ja = clique
                        iau = iaN[ia]; jau = iaN[ja]
                        boij = bom[iau,jau]
                        bom[iau,jau] = bom[jau,iau] = boij + 1
            else:
                cns[iaN] = 3
            dvs = cns - bom.sum(axis=0)
            stags = ''
            for idv,dv in enumerate(dvs):
                if dv != 0: stags += ' %d'%(idv+1)
            msg = '#ERROR: sum(dvs) is %d but zero!! Relevant tags are %s'\
                         %( sum(dvs), stags )
            if neutral:
                print(' --  zs =',  zs)
                print(' -- dvs = ', dvs)
                print(' -- bom = ',)
                print(np.array(bom))
                assert sum(dvs) == 0, msg


        set0 = set([ abs(dvi) for dvi in dvs ])
        #print ' -- set0 = ', set0
        if set0 == set([0,1]): #, '#ERROR: some atom has abs(charge) > 1??'
            ias1 = ias[ np.array(dvs) == 1 ]
            ias2 = ias[ np.array(dvs) == -1 ]
            for ia1 in ias1:
                for ia2 in ias2:
                    #assert bom[ia1,ia2] == 0, \
                    #   '#ERROR: the two atoms are supposed to be not bonded!'
                    bo12 = bom[ia1,ia2]
                    if bo12 > 0: # CH3-N(=O)-N-NHCH3 --> CH3-N(=O)=N-NHCH3
                        assert self.zs[ia2] == 7 and cns[ia2] == 3, \
                                 '#ERROR: originally it is not [#7X3] atom??'
                        cns[ia2] = 5
                        bom[ia1,ia2] = bom[ia2,ia1] = bo12 + 1

            dvs = cns - bom.sum(axis=0)
            msg = '#ERROR: sum(dvs) = %d, should be zero!!'%( sum(dvs) )
            if neutral:
                assert sum(dvs) == 0, msg

        charges = -dvs

        return bom, cns, charges

    def perceive_bond_order(self, neutral=True, once=True, irad=False, \
                            user_cns0=None, debug=False):
        """
        once -- if it's True, then get __ONLY__ the saturated graph
                e.g., cccc --> C=CC=C; otherwise, u will obtain C=CC=C
                as well as [CH2]C=C[CH2]
        user_cns0 -- user specified `cns_ref, used when generating amons, i.e.,
                     all local envs have to be retained, so are reference
                     coordination_numbers !
        """
        zs = self.zs
        g = self.G
        na = self.na
        ias = self.ias
        bom = copy.deepcopy(g) # later change it to bond-order matrix
        cns = g.sum(axis=0)
        nuclear_charges =      [1,2, 3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18, 35]
        coordination_numbers = [1,0, 1,2,3,4,3,2,1,0,   1, 2, 3, 4, 3, 2, 1, 0,  1]
        if user_cns0 is None:
            cnsr = dict(zip(nuclear_charges, coordination_numbers))
            cns0 = np.array([ cnsr[zj] for zj in self.zs ])
        else:
            cns0 = user_cns0
            dN = len(cns) - len(cns0)
            if dN > 0:
                cns0 = np.r_[ cns0, [1,]*dN ]

        dvs = cns0 - cns;
        if debug:
            print('     zs = ', self.zs)
            print(' +1 dvs = ', dvs    )
            print('   cns0 = ', cns0   )
            print('    cns = ', cns    )

        # 1) for =O, =S
        ias_fringe = ias[ np.logical_and(dvs == 1, np.logical_or(self.zs == 8, self.zs == 16) ) ]
        if debug: print('ias_fringe = ', ias_fringe)
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            assert jas.shape[0] == 1
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 2
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        # 2) for #N
        ias_fringe = ias[ np.logical_and(dvs == 2, self.zs == 7) ]
        if debug: print('ias_fringe_2 = ', ias_fringe)
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            assert jas.shape[0] == 1
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 3
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        # 3) for $C (quadruple bond)
        ias_fringe = ias[ np.logical_and(dvs == 3, self.zs == 6) ]
        if debug: print('ias_fringe_3 = ', ias_fringe)
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            assert jas.shape[0] == 1
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 4
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        ## 4) fix special cases, where we consider
        #     the middle N has a valence of 5
        ##    -N-N-N  -->  -N=N#N
        ##    >C-N-N  -->  >C=N#N
        if debug:
            print(' ** bom = ')
            print(bom)
            print(' ** zs = ', self.zs)
        oem = self.to_oem( bom, np.zeros(self.na), with_coords=False )
        q = '[#6X3,#7X2]-[#7X2]#[#7X1]'
        ots = is_subg(oem, q, iop = 1)
        if ots[0]:
            #print ' * ots = ', ots
            ia,ja,ka = ots[1][0]
            bom[ia,ja] = bom[ja,ia] = 2
            cns0[ja] = 5
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        ## 5) fix special cases
        ##    -C(O)O  -->  -C(=O)O
        oem = self.to_oem( bom, np.zeros(self.na), with_coords=False )
        q = '[#6X3](=[#8X1])=[#8X1]'
        ots = is_subg(oem, q, iop = 1)
        if ots[0]:
            #print ' ** ots = ', ots
            ia,ja,ka = ots[1][0]
            bom[ia,ja] = bom[ja,ia] = 1
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        # 6) fix cases like >C-C-C< or =C-C-C< or -N-C-C< (dvs=[1,2,1])
        #                   >C-C-C-C< (dvs=[1,2,2,1])
        #                   >C-C-C-C-C< (dvs=[1,2,2,2,1])
        #    cases like >C-C(-X)-C-C-C(-X)-C< must be excluded (note that
        #                    dvs=[1,2,2,1] is a subset of `dvs of all atoms)
        filt1 = (dvs == 1); zs1 = zs[filt1]; ias1 = ias[filt1]; na1 = len(zs1)
        filt2 = (dvs == 2); zs2 = zs[filt2]; ias2 = ias[filt2]; na2 = len(zs2)
        if na2 > 0:
            g2 = bom[ias2,:][:,ias2]
            iasL = []
            for ias2c_raw in find_cliques(g2):
                ias2c = [ ias2[ja] for ja in ias2c_raw ]
                # store the index of atom with dv=1, which
                # is connected to atom with dv=2
                ias1c = []
                for ia2 in ias2c:
                    for ia1 in ias1:
                        if bom[ia1,ia2] == 1:
                            ias1c.append( ia1 )
                # now sort atoms to form a Line
                na1c = len(ias1c)
                if na1c == 2:
                    # now sort atoms to form a linear chain !
                    iasc = [ ias1c[0], ]; icnt = 0
                    ias_compl = copy.copy( ias2c )
                    while ias_compl:
                        for ial in ias_compl:
                            if bom[ial, iasc[icnt]] == 1:
                                iasc.append( ial )
                                icnt += 1
                                ias_compl.remove( ial )
                    iasc.append( ias1c[1] )
                    nac = len(iasc)
                    # now check if the two end atoms along this Line
                    # are not connected to another atom with `dv=1,
                    # e.g., >C-C(-X)-C-C-C(-X)-C<
                    icontinue = True
                    for iax in ias1:
                        if iax not in iasc:
                            if np.any([bom[iax,iac] == 1 for iac in iasc ]):
                                icontinue = False
                    if icontinue:
                        for iac in range(nac-1):
                            ka1 = iasc[iac]; ka2 = iasc[iac+1]
                            bom[ka1,ka2] = bom[ka2,ka1] = 2
        # now update `cns and `dvs
        cns = bom.sum(axis=0)
        dvs = cns0 - cns


        # now update the values of `cnsr
        # for multi-valent atom (note that the original values in
        # `cnsr are for normal valence atom
        ias_abn = ias[ dvs < 0 ] # index of atoms of abnormal nature
        cns0_u = copy.deepcopy(cns0)
        if debug: print(' * cns0   = ', cns0)
        for ia in ias_abn:
            zi = self.zs[ia]
            msg = '  -- ia = %d, zi = %d, zs = [%s], dvs = [%s]'%( ia, zi, \
                                   np.str(self.zs), np.str(dvs) )
            if zi in [7,]:
                # for R-N(=O)=O, dv = 3-5 = -2;
                # exception: R-[NH3+] & =N(=O)-, dv = 3-4 = -1,
                #            where we cannot determine the valence
                #            is 3 or 5, first we assign it to 5 and
                #            handle the rest at the part of charge
                #            perceiving later.
                cns0_u[ ia ] = {-1:5, -2:5}[ dvs[ia] ]
            elif zi in [15]:
                # R-P(=O)(O)(O), P(=O)Cl3, dv = 3-5 = -2
                # PCl5, dv = 3-5 = -2
                assert dvs[ia] in [-2,], msg
                cns0_u[ ia ] = {-2:5}[ dvs[ia] ]
            elif zi in [16,]:
                # R-S(=O)-R, dv = 2-4 = -2,           valence=4
                # R-S(=O)(=O)-R, dv = 2-6 = -4,       valence=6
                # SF6, dv = 2-6 = -4,                 valence=6
                # S(=O)(=O)=O, dv = 2-6 = -4,         valence=6
                # RC=S(=O)(=O), dv = 2-5 = -3,        valence=6
                assert dvs[ia] in [-4,-2], msg
                cns0_u[ ia ] = {-2:4, -4:6 }[ dvs[ia] ]
            else:
                print('#ERROR: do not know how to handle Exception (OEChem.py)')
                print('    ia, zi, dvs[ia] = ', ia, zi, dvs[ia])
                #raise '#ERROR' #sys.exit(2)
        if debug: print(' * cns0_u = ', cns0_u)


        #######################################################
        #######################################################
        cns0 = cns0_u


        # now update `cns and `dvs again
        cns = bom.sum(axis=0)
        dvs = cns0 - cns
        if debug:
            print(' +2 dvs = ', dvs)
            print('   cns0 = ', cns0)
            print('    cns = ', list(cns))

        # now update `cns and `dvs again
        cns = bom.sum(axis=0)
        dvs = cns0 - cns

        if debug:
            print(' +3 dvs = ', dvs)
            print('   cns0 = ', cns0)
            print('    cns = ', list(cns))

        combs = [ ]; bos = [ ]
        nclique = 0

        for dv in [2, 1]:
            # for C_sp2, dv = 4-3 = 1, N_sp2, dv = 3-2 = 1;
            # for C_sp, dv = 4-2 = 2
            # for O_sp2, dv = 2-1 = 1

            BO = {2: 3, 1: 2}[dv]
            # atoms to be re-bonded by double/triple bonds
            ias_dv = ias[ dvs == dv ]
            #print ' -- ias_dv = ', ias_dv + 1

            na_dv = len(ias_dv)
            if na_dv > 0:
                g2 = g[ias_dv,:][:,ias_dv]
                cliques_dv = find_cliques(g2)
                nclique_dv = len(cliques_dv)
                nclique += nclique_dv
                for cliques_i in cliques_dv:
                    nc = len(cliques_i)
                    # e.g., for 'cccNccncc', cliques = [ [0,1,2],
                    #                   [4,5,6,7,8], ], nc = 2
                    # for clique_1, there are C_2^1 (where 2 is
                    # the num_bonds) possibilities,
                    #     i.e., c=cc, cc=c; for clique_2, there are
                    #     2 possiblities as well.
                    # Thus, we'll have 2*2 = 4 possible combs,
                    nc_i = len(cliques_i)
                    if debug: print(' +++++ cliques_i = ', cliques_i)

                    ifound = True
                    if nc_i == 1:
                        ifound = False
                    elif nc_i == 2:
                        # use relative indices; later u'll convert to abs idx
                        ias_dv_i0 = [  [[0,1],],  ]
                    else: # nc_i >= _dv:
                        g2c = g2[cliques_i,:][:,cliques_i]
                        #print ' g2c = ', g2c
                        #ne = (g2c > 0).sum()/2
                        #nring = ne + 1 - nc_i
                        #
                        # special case: 4-1-2-3
                        # but atomic indices are 1,2,3,4
                        # consequently, g2c = [[0 1 0 1]
                        #                      [1 0 1 0]
                        #                      [0 1 0 0]
                        #                      [1 0 0 0]]
                        # while the program `find_double_bonds will
                        # try to find double bonds in a conjugate systems
                        # sequentially, i.e., the resulting `ias_dv_i0 is [[0,1], ]
                        # which should be instead [ [4,1],[2,3] ]
                        #
                        # Thus, `once should be set to False when calling `find_double_bonds
                        ias_dv_i0s = find_double_bonds(g2c, once=False, irad=irad)
                        n_ = len(ias_dv_i0s)
                        nmax_ = 2
                        for i_ in range(n_):
                            ias_ = ias_dv_i0s[i_]
                            ni_ = len(np.ravel(ias_))
                            if ni_ > nmax_:
                                nmax_ = ni_
                                i_chosen = i_
                        if once:
                            # choose only 1 from all options
                            ias_dv_i0 = [ ias_dv_i0s[ i_chosen ], ]
                        else:
                            ias_dv_i0 = ias_dv_i0s
                        if debug: print(' +++++ ias_dv_i0 = ', ias_dv_i0)

                    if ifound:
                        map_i = list( np.array(ias_dv, np.int)[cliques_i] )
                        if debug: print('----- ', map_i, cliques_i)

                        bonds_dv_i = [ ]
                        for iias in ias_dv_i0:
                            if debug: print(' -- iias = ', iias)
                            cisU = [ [ map_i[jas[0]], map_i[jas[1]] ] \
                                           for jas in  iias ]
                            bonds_dv_i.append( cisU )
                        if debug: print('----- ', bonds_dv_i)

                        combs.append( bonds_dv_i )
                        bos.append( BO )

        boms0 = []
        if nclique >= 1:
            for bs in cim.products(combs):
                bom_i = copy.copy(bom)
                if debug: print('-- bs = ', bs, ', bos = ', bos)
                for i,bsi in enumerate(bs):
                    for bi in bsi:
                        if debug: print(' ***** bi = ', bi)
                        ia1, ia2 = bi
                        bom_i[ia1,ia2] = bom_i[ia2,ia1] = bos[i]

                boms0.append(bom_i)
        else:
            boms0.append( bom )

        boms = []; charges = []
        cans = []; Mols = []
        for bom_i in boms0:

            cns_u = bom_i.sum(axis=0)
            dvs_u = cns0_u - cns_u;
            if debug: print(' ## dvs_u = ', dvs_u)

            ## fix case like >C-C-C< or >C-C-N- or >C-C-C-C<
            ##    where [1,(2)_n,1] appears as a subset of `dvs
            ##    In essense, we wanna find if graph 1-2-1 or
            ##    1-2-2-1 or 1-2-2-2-1 is a subgraph
            bom_i = self.update_double_bond_chain(bom_i, dvs_u)
            cns_u = bom_i.sum(axis=0)
            dvs_u = cns0_u - cns_u
            if debug: print(' ### dvs_u = ', dvs_u)

            # the last update
            # seperated charges !!
            # e.g., [O-]C(=O)CC[NH3+]
            bom_i_u, cns0_u2, charges_i = self.update_charges(bom_i, \
                                     cns0_u, dvs_u, neutral=neutral)
            cns_u = bom_i_u.sum(axis=0)
            dvs_u = cns0_u2 - cns_u + charges_i

            if debug: print(' #### dvs_u = ', dvs_u)
            if np.all(dvs_u == 0):
                pass
                #boms.append( bom_i_u )
                #charges.append( charges_i )
            else:
                print(' #### dvs_u = ', dvs_u)

            Mol_i = self.to_oem(bom_i_u, charges_i, with_coords=True)
            Mol_i.SetDimension(3)
            #print Mol_i.GetCoords()
            can_i = to_can(Mol_i)

            # since the input coordinates is fixed, thus the number of
            # output molecules is determined soly by number of `cans
            if can_i not in cans:
                cans.append(can_i)
                boms.append(bom_i_u)
                charges.append(charges_i)
                Mols.append(Mol_i)

        if once:
            self.oem = Mols[0]
            self.can = cans[0]
            self.bom = boms[0]
            self.charges = charges[0]
        else:
            self.oem = Mols
            self.can = cans
            self.bom = boms
            self.charges = charges


    def write(self, sdf, Tv=[0.,0.,0]):
        """
        Tv: Translation vector
            This may be useful when u need to stack molecules
            within a cube and leave molecules seperated by a
            certain distance
        """
        assert type(self.oem) is not list, '#ERROR: `once=False ?'
        coords = self.coords
        if np.linalg.norm(Tv) > 0.:
            icnt = 0
            for ai in self.oem.GetAtoms():
                coords[icnt] = np.array(coords[icnt]) + Tv
                icnt += 1
        # Don't use lines below, they will __change__ the molecule
        # to 2D structure!!!!!!!!
        obsolete = """ifs = oemolistream()
        ofs = oemolostream()
        iok = ofs.open( sdf )
        OEWriteMolecule(ofs, wm)"""
        to_sdf(self.zs, coords, self.bom, self.charges, sdf)


def to_can(m, ImpHCount=False, reconstruct=False):
    # input should be OEGraphMol object
    flavor = OESMILESFlag_Canonical
    if ImpHCount:
        # the only way to specify multiple flavors
        flavor = OESMILESFlag_ImpHCount | OESMILESFlag_Canonical
    if reconstruct:
        mu = OEGraphMol()
        atoms = {}; icnt = 0
        for ai in m.GetAtoms():
            ia = ai.GetIdx()
            zi = ai.GetAtomicNum()
            if zi > 1:
                aiu = mu.NewAtom( zi )
                aiu.SetHyb( OEGetHybridization(ai) )
                aiu.SetImplicitHCount( ai.GetImplicitHCount() )
                atoms[ icnt ] = aiu
                icnt += 1
        for bi in m.GetBonds():
            p, q = bi.GetBgnIdx(), bi.GetEndIdx()
            biu = mu.NewBond( atoms[p], atoms[q], bi.GetOrder() )
        OEFindRingAtomsAndBonds(mu)
        OEAssignAromaticFlags(mu, OEAroModel_OpenEye)
        OECanonicalOrderAtoms(mu)
        OECanonicalOrderBonds(mu)
    else:
        mu = m
        OEAssignAromaticFlags(mu, OEAroModel_OpenEye)
    can = OECreateSmiString(mu, flavor)
    return can

def is_subsub(q, ts):
    """
    check is `q is a subset of any subset in set `ts
    """
    iok = False
    for tsi in ts:
        if set(q) <= tsi:
            iok = True; break
    return iok

def is_subg(t, q, qidxs=[], iop=0, rule='none', kekulize=False):
    """
    check if `q is a subgraph of `t

    if iop .lt. 0, then the matched atom indices in `t will also
    be part of the output
    """

    if type(t) is str:
        m = OEGraphMol()
        iok = OESmilesToMol(m, t)

        # lines below does not work!!
        if kekulize:
            OEClearAromaticFlags(m)
            OEKekulize(m)
    else:
        m = t

    ss = OESubSearch(q)
    iok = OEPrepareSearch(m, ss)
    isg = ss.SingleMatch(m)

    op = [isg]; in_subm = False
    if iop > 0:
        idxs = []; idxs_aux = []
        if isg:
            for i,match in enumerate(ss.Match(m)):
                idxsQ_i = []; idxsT_i = []
                for ma in match.GetAtoms():
                    idxsQ_i.append( ma.pattern.GetIdx() )
                    idxsT_i.append( ma.target.GetIdx() );
                seqs = np.argsort(idxsQ_i);
                #print(' -- seqs = ',seqs)
                idxsT_iu = list(np.array(idxsT_i,np.int)[seqs])
                set_i = set(idxsT_iu)
                if set_i not in idxs_aux:
                    # it's always safe to append the atomic indices whose
                    # sorted ones is unique in a auxiliary set `idxs_aux
                    idxs.append( idxsT_iu )
                    idxs_aux.append( set(idxsT_iu) )
                #print('idxs = ', idxs)

            if rule in ['exclusive',]:
                # this means that the `q to be searched has unique existence
                # e.g., suppose we wanna know if the structure contains a
                # 'S=O', then 'S(=O)=O' does not count; this could be told by
                # if two matched set of idxs is joint
                if len(idxs) > 1:
                    idxsU = []
                    for idxs_i in idxs:
                        if not is_joint_vec(idxs_i, cim.get_compl(idxs, [idxs_i,])):
                            idxsU.append(idxs_i)
                    idxs = idxsU
                    if len(idxs) == 0: isg = False

            if qidxs != []:
                # check if `qidxs is a subset of `idxs
                in_subm = ( in_subm or np.all([qidx in idxs for qidx in qidxs]))

        op = [isg, idxs, in_subm ]

    return op


def find_cliques(g1):
    """
    the defintion of `clique here is not the same
    as that in graph theory, which states that
    ``a clique is a subset of vertices of an
    undirected graph such that every two distinct
    vertices in the clique are adjacent; that is,
    its induced subgraph is complete.''
    However, in our case, it's simply a connected
    subgraph, or a fragment of molecule. This is useful
    only for identifying the conjugated subset of
    atoms connected all by double bonds or bonds of
    pattern `double-single-double-single...`
    """

    n = g1.shape[0]
    G = nx.Graph(g1)
    if nx.is_connected(G):
        cliques = [ range(n), ]
    else:
        cliques = []
        sub_graphs = nx.connected_component_subgraphs(G)
        for i, sg in enumerate(sub_graphs):
            cliques.append( sg.nodes() )

    return cliques


def find_neighbors(g, ias):
    """
    get the neighbors of a list of atoms `ias
    """
    neibs = []
    na = g.shape[0]
    ias0 = np.arange(na)
    for ia in ias:
        for ja in ias0[ g[ia] > 0 ]:
            if ja not in ias:
                neibs.append( ja )
    return neibs


def is_edge_within_range(edge, edges, dsg, option='adjacent'):

    i1,i2 = edge
    iok = False
    if len(edges) == 0:
        iok = True
    else:
        jit = is_joint_vec(edge, edges) # this is a prerequisite
        #print(' -- jit = ', jit)
        if not jit:
            for edge_i in edges:
                j1,j2 = edge_i
                ds = []
                for p,q in [ [i1,j1], [i2,j2], [i1,j2], [i2,j1] ]:
                    ds.append( dsg[p,q] )
                #print(' -- ds = ', ds)
                iok1 = (option == 'adjacent') and np.any( np.array(ds) == 1 )
                iok2 = (option == 'distant') and np.all( np.array(ds) >=2 )
                if iok1 or iok2:
                    iok = True; break
    return iok

def is_adjacent_edge(edge, edges, dsg):
    return is_edge_within_range(edge, edges, dsg, option='adjacent')

def is_distant_edge(edge, edges, dsg):
    return is_edge_within_range(edge, edges, dsg, option='distant')

def get_joint(edge1, edge2):
    return np.intersect1d(edge1,edge2)

def is_joint(edge1, edge2):
    return np.intersect1d(edge1,edge2).shape[0] > 0

def is_joint_vec(edge, edges):
    return np.any([ is_joint(edge,edge_i) for edge_i in edges])

def comba(cs):
    csu = []
    for c in cs: csu += c
    return csu

def sort_edges(compl, edge_i, dsg):
    """
    sort the edges by the relative distance to nodes in `edge_i
    """
    ds = []
    for ei in compl:
        i,j = ei; k,l = edge_i
        dsi = [ dsg[r,s] for r,s in [ [i,k],[j,l],[i,l],[j,k] ] ]
        ds.append( np.min(dsi) )
    seq = np.argsort( ds )
    return [ compl[q] for q in seq ]


def edges_standalone_updated(g, irad=False):
    """
    e.g., given smiles 'cccccc', edges_1 = [(0,1),(1,2),(2,3),(3,4),(4,5)],
    possible outputs: [[(0,1),(2,3),(4,5)], i.e., 'c=cc=cc=c'

    In order to get SMILES string like 'cc=cc=cc', u have to modify this function

    Notes:
    ===================
    irad  -- If set to True, all possible combinations of double bonds
             will be output. For the above example, output will be
             'c=cc=cc=c' and 'cc=cc=cc';
             If set to False, only the formal SMILES results, i.e., the
             number of residual nodes that are not involved in double bonds
             is zero.
    """
    n = g.shape[0]
    assert n >= 3
    edges0 = [list(edge) for edge in np.array(list(np.where(np.triu(g)>0))).T];
    Gnx = nx.Graph(g)
    dsg = np.zeros((n,n), np.int)
    for i in range(n):
        for j in range(i+1,n):
            dsg[i,j] = dsg[j,i] = nx.shortest_path_length(Gnx, i,j)
    nodes0 = set( comba(edges0) )
    ess = []; nrss = []
    for edge_i in edges0:
        compl = cim.get_compl(edges0, [edge_i])
        compl_u = sort_edges(compl, edge_i, dsg)
        edges = copy.deepcopy( compl_u )
        edges_u = [edge_i, ]
        while True:
            nodes = list(set( comba(edges) ))
            nnode = len(nodes)
            if nnode == 0 or np.all(g[nodes,:][:,nodes] == 0):
                break
            else:
                edge = edges[0]
                edges_copy = copy.deepcopy( edges )
                if is_adjacent_edge(edge, edges_u, dsg):
                    edges_u.append(edge)
                    for edge_k in edges_copy[1:]:
                        if set(edge_k).intersection(set(comba(edges_u)))!=set():
                            edges.remove(edge_k)
                else:
                    edges.remove(edge)
        edges_u.sort()
        if edges_u not in ess:
            nrss_i = list(nodes0.difference(set(comba(edges_u))))
            nnr = len(nrss_i)
            if nnr > 0:
                if irad:
                    ess.append( edges_u )
                    nrss.append( nrss_i )
            else:
                ess.append( edges_u )
                nrss.append( nrss_i )
    return ess, nrss # `nrss: nodes residual, i.e., excluded in found standalone edges

def find_double_bonds(g, once=False, irad=False, debug=False):
    """
    for aromatic or conjugate system, find all the possible combs
    that saturate the valences of all atoms
    """
    ess, nrss = edges_standalone_updated(g, irad=irad)
    if once:
        edges = [ess[0], ]
    else:
        edges = ess
    return edges

def write_sdf(obj, sdf, Tv=[0,0,0]):
    # `obj is class `OEGraphMol or `StringM
    # If u try to embed this function in the StringM() or OEMol class
    # you will end up with wierd connectivies (of course Wrong)
    if obj.__class__.__name__ == 'StringM':
        M2 = obj
        zs = M2.zs
        coords = np.array(M2.coords)+Tv
        G = M2.bom
        charges = M2.charges
    elif obj.__class__.__name__ == 'OEGraphMol':
        m = obj
        G = oem2g(m)
        na = m.NumAtoms()
        dic = m.GetCoords()
        coords = np.array([ dic[i] for i in range(na) ])
        zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ], np.int)
        charges = np.array( [ ai.GetFormalCharge() for ai in m.GetAtoms() ] )
    elif type(obj) is list:
        zs, coords, G, charges = obj
    write_sdf_raw(zs, np.array(coords)+Tv, G, charges, sdf=sdf)

def write_sdf_raw(zs, coords, bom, charges, sdf=None):
    """
     cyclobutane
         RDKit          3D

      4  4  0  0  0  0  0  0  0  0999 V2000
       -0.8321    0.5405   -0.1981 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.3467   -0.8825   -0.2651 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.7190   -0.5613    0.7314 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.4599    0.9032    0.5020 C   0  0  0  0  0  0  0  0  0  0  0  0
      1  2  1  0
      2  3  1  0
      3  4  1  0
      4  1  1  0
    M  END
    """

    na = len(zs)
    nb = (np.array(bom) > 0).ravel().sum()/2

    ctab = 'none\n     RDKit          3D\n\n'
    fmt1 = '%3d'*6 + '  0  0  0  0999 V2000\n'
    ctab += fmt1%( na, nb, 0,0,0,0)

    fmt1 = '%10.4f'*3 + ' %-3s'
    fmt2 = '%2d' + '%3d'*11 + '\n'
    str2 = fmt2%(tuple([0,]+ [0,]*11))
    fmt = fmt1 + str2
    for i in range( na):
        px, py, pz = coords[i]
        zi = zs[i]
        ctab += fmt%(px, py, pz, ad.chemical_symbols[zi])

    for i in range( na):
        for j in range(i+1, na):
            boij = bom[i,j]
            if boij > 0:
                ctab += '%3d%3d%3d%3d\n'%(i+1,j+1,boij,0)

    ias = np.arange(na) + 1
    iasc = ias[ np.array(charges) != 0 ]
    nac = iasc.shape[0]
    if nac > 0:
        ctab += 'M  CHG%3d'%nac
        for iac in iasc:
            ctab += ' %3d %3d'%(iac, charges[iac-1])
        ctab += '\n'

    ctab += 'M  END'
    if sdf != None:
        with open(sdf,'w') as f: f.write(ctab)
        return
    else:
        return ctab

def oem2g(oem):
    na = oem.NumAtoms()
    G = np.zeros((na, na), np.int)
    for bi in oem.GetBonds():
        i, j, boij = bi.GetBgnIdx(), bi.GetEndIdx(), bi.GetOrder()
        G[i,j] = G[j,i] = boij
    return G #(G > 0).astype(np.int)

def read(sdf, ibom=False):
    """
    read sdf file
    Sometimes, u cannot rely on ase.io.read(), esp.
    when there are more than 99 atoms
    """
    cs = file(sdf).readlines()
    c4 = cs[3]
    na, nb = int(c4[:3]), int(c4[3:6])
    ats = cs[4:na+4]
    symbs = []; ps = []
    for at in ats:
        px,py,pz,symb = at.split()[:4]
        symbs.append(symb)
        ps.append([ eval(pj) for pj in [px,py,pz] ])
    ps = np.array(ps)
    aseobj = ase.Atoms(symbs, ps)

    if ibom:
        ctab = cs[na+4:na+nb+4]
        bom = np.zeros((na,na))
        for c in ctab:
            idx1,idx2,bo12 = int(c[:3]), int(c[3:6]), int(c[6:9])
            bom[idx1-1,idx2-1] = bom[idx2-1,idx1-1] = bo12
        return aseobj, bom
    else:
        return aseobj

def get_idx_of_molecule(smi, cans, reconstruct=False):
    can = to_can( smi2oem(smi), reconstruct=reconstruct )
    print('idx starting from 1 !')
    idxs0 = np.arange(1, len(cans)+1)
    idxs = idxs0[ can == np.array(cans) ]
    print(' '.join( [ '%d'%i for i in idxs ] ))
    return idxs

def to_ase(m):
    zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ])
    na = len(zs); dic = m.GetCoords(); coords = []
    for i in range(na): coords.append( dic[i] )
    return ase.Atoms(zs, coords)

