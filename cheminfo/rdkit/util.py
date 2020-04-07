

import networkx as nx
import itertools as itl
import scipy.spatial.distance as ssd

import multiprocessing
import copy_reg
import types as _TYPES

import numpy as np
import ase.io as aio
import ase.data as ad
import os, sys, re, copy
import ase, openeye

import aqml.cheminfo.math as cim

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


## register instance method
## otherwise, the script will stop with error:
## ``TypeError: can't pickle instancemethod objects
def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(_TYPES.MethodType, _reduce_method)



class GraphM(object):

    def __init__(self, g):
        self.nn = g.shape[0] # number_of_nodes
        self.n = self.nn
        g1 = (g > 0).astype(np.int)
        np.fill_diagonal(g1, 0)
        self.g1 = g1
        self.nb = g1.sum()/2 # num of bonds (i.e., edges)
        self.bonds = [ list(edge) for edge in \
             np.array( list( np.where(np.triu(g)>0) ) ).T ]

    def is_connected(self):
        return self.ne - self.n + 1 >= 0


class RawMol(object):

    def __init__(self, Obj):
        """
        Three types of `obj as input are possible:
        1) ase.Atoms
        2) XYZ file (string type)
        """
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

        self.G = G
        self.ds = ds


class Mol(RawMol):

    def __init__(self, Obj):
        RawMol.__init__(self, Obj)


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
                print ' --  zs =',  zs
                print ' -- dvs = ', dvs
                print ' -- bom = ',
                print np.array(bom)
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
        nuclear_charges =      [1,2, 3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18, 35, 53]
        coordination_numbers = [1,0, 1,2,3,4,3,2,1,0,   1, 2, 3, 4, 3, 2, 1, 0,  1,  1]
        z2cn = dict(zip(nuclear_charges, coordination_numbers))
        cnsr = np.array([ z2cn[zj] for zj in self.zs ]) # reference

        # for any atom i, get the num of heavy atom neighbor with degree 1
        nn1s = []
        for ia in range(na):
            nn1s.append( (cns[ias[g[i]>0]] == 1).sum() )

        # now update the values of `tvs for multi-valent atom
        for ia in range(na):
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
                if cns[ia] == 1:
                    tvs[ia] = 3; chgs[ia] = 0
                elif cns[ia] <= 2:

                else: #cns[ia] == 4:
                    tvs[ia] = 3; chgs[ia] = 1
            elif zi in [15]:
                # R-P(=O)(R)(R)
                # PCl5, dv = 3-5 = -2
                if cns[ia] <= 3:
                    # PR3, RP=R, P#R
                    tvs[ia] = 3
                else:
                    # PR5, R=PR3
                    tvs[ia] = 5

            elif zi in [16,]:
                if cns[ia] <= 2:
                    tvs[ia] = 2
                elif cns[ia] <= 3:
                    # >S=O, >S=C<, ..
                    tvs[ia] = 4
                else:
                    # >S(=R)=R, R=S(=R)=R, SF6
                    tvs[ia] = 6
            else:
                print '#ERROR: do not know how to handle Exception
                print '    ia, zi, dvs[ia] = ', ia, zi, dvs[ia]
                #raise '#ERROR' #sys.exit(2)

        nrmax = na/2
        nbmax = (g>0).sum()/2
        iok, bom = cf.update_bom(nrmax,nbmax,zs,tvs,g,icon)

        # now restore charges for case, e.g., NN bond in C=N#N, or -N(=O)=O
        iok_U, bom_U, chgs_U = accommodate_chgs(zs, chgs, bom, allow_bb=False)
        if not iok_U: continue

        return can


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

