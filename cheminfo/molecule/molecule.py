#!/usr/bin/env python

import itertools as itl
#import scipy.spatial.distance as ssd
import aqml.cheminfo.molecule.elements as cce
import numpy as np
import os, sys, re, copy
from aqml.cheminfo.rw.ctab import write_ctab
from aqml.cheminfo.rw.pdb import write_pdb
import aqml.cheminfo.math as cm
#from rdkit import Chem
#import indigo
from aqml.cheminfo.molecule.core import *
import aqml.cheminfo.oechem.core as coc
#import aqml.cheminfo.rdkit.amon_f as craf
import aqml.cheminfo.graph as cg
import tempfile as tpf
import cml.famon as cf

global T,F
T=True; F=False

#def to_indigo_can(smi):
#    obj = indigo.Indigo()
#    m = obj.loadMolecule(smi)
#    m.aromatize()
#    can = m.canonicalSmiles()
#    return can


class Mol(RawMol):

    def __init__(self, zs, coords, ican=T, allow_mc=T):
        RawMol.__init__(self, (zs,coords))
        if ican:
            if (not self.is_connected) and (not allow_mc):
                print(' ** molecule dissociated!')
                self.can = 'None'
                return
            self.can = self.perceive_bond_order()

    def check_N5_symm(self,ia,dvs):
        """
        check if a neighboring atom is also a N5 (or P5) atom with
        odd-numbered adjacent atoms having dvi=1
        """
        iok = False
        jas = self.ias[self.g[ia]>0]
        for ja in jas:
            kas = np.setdiff1d( self.ias[self.g[ja]>0], [ia] )
            for ka in kas:
                if ka in self.ias0:
                    #las = self.ias[ np.logical_and(self.g[ka]>0, dvs==self.dvi_neib) ]
                    las = self.ias[ np.logical_and(self.g[ka]>0, np.abs(dvs)==1) ]
                    nal = len(las)
                    if nal == 0: continue
                    iasm = []
                    for l in las:
                        if dvs[l] == 1:
                            for _ias in self.iass1:
                                if self.amap1[l] in _ias:
                                    iasm.append( _ias )
                        else: # -1
                            for _ias in self.iass2:
                                if self.amap2[l] in _ias:
                                    iasm.append( _ias )
                    if len(iasm)==1 and len(iasm[0])%2==1:
                        iok = True
        return iok

    def is_not_conjugated(self, _ias, _g):
        """ check if atoms form a conjugated chain or cycle
        amongs atoms specified by `_ias from a graph `g
        """
        na = len(_ias)
        iok = False
        if na%2 == 1: # imply the existence of R1=[NX]=R2
            iok = True
        else:
            if na != 2:
                sg2 = _g[_ias][:,_ias].astype(np.int32)
                nb = int( (sg2>0).sum()/2 )
                # are there even number of double bonds?
                _ipss = np.ones((2,nb,int(na/2)))
                ipss = np.array(_ipss,dtype='int32',order='F') # ipss(2,np,nr0)
                #print ' ** sg2 = ', sg2
                nr = cf.locate_double_bonds(sg2,ipss,self.debug)
                if nr == 0:
                    iok = True
        return iok


    def find_bond_to_N5(self,dvs):
        """
        strategy:
        recursively find N5 (or P5) atoms after trial & error. If things
        are ok after re-assignement of N5 (P5) env to a N3 (P3) atom, then
        it's a genuine N5 (P5) atom; otherwise, restore to N3 (P5) env.
        """
        # first identify all N3 atoms
        ft1 = np.logical_or(self.zs==7,self.zs==15)
        _filts = [ft1, dvs==0, self.cns==3, self.cns_heav>=2 ]
        filt = np.array([True,]*self.na, np.bool)
        for _filt in _filts:
            filt = np.logical_and(filt,_filt)
        ias0 = self.ias[filt]
        self.ias0 = ias0
        pairs = []
        if len(ias0) == 0:
            return pairs

        #for dvi_neib in [1,-1]:
        # dvi_neib = 1, e.g., [NH][NH][NH]
        # dvi_neib = -1, e.g., OP(=O)=P(=[PH2]C)C
        # an exotic case: O[N+]([O-])=[N+]([N-]C)O, or ON(=O)=N(=NC)O,
        #                N_4 (dvi=0) has two neighbors with dvsi=[1,-1]
        ias1 = self.ias[dvs==1] #dvi_neib]
        ias2 = self.ias[dvs==-1]
        n1 = len(ias1)
        n2 = len(ias2)
        #self.dvi_neib = dvi_neib
        if n1>0 or n2>0:
            amap1 = dict(list(zip(ias1, list(range(len(ias1))))))
            amap2 = dict(list(zip(ias2, list(range(len(ias2))))))
            amaps = [amap1,amap2];
            self.amap1 = amap1
            self.amap2 = amap2
            sg1 = self.g[ias1][:,ias1]
            # nodes are returned (relative idx)
            iass1 = cg.Graph(sg1).find_cliques() if n1>0 else [np.array([],np.int)]
            sg2 = self.g[ias2][:,ias2]
            iass2 = cg.Graph(sg2).find_cliques() if n2>0 else [np.array([],np.int)]
            sgs = [sg1,sg2]
            tas = [iass1,iass2]
            self.iass1 = iass1
            self.iass2 = iass2
            ng1 = len(iass1); ng2 = len(iass2)

            #print ' * iass1 = ', iass1, ', iass2 = ', iass2
            #print ' * zs = ', self.zs
            #print ' * dvs = ', dvs

            # try to find N5 and any neighbor, for which a BO=2 could be
            # potentially assigned to.
            istop = False
            for i in ias0:
                # now check if there is any neighbor with dv=1, e.g., [NH][NH][NH]
                jas1 = self.ias[ np.logical_and(self.g[i]>0, dvs==1) ]
                jas2 = self.ias[ np.logical_and(self.g[i]>0, dvs==-1) ]
                naj1 = len(jas1); naj2 = len(jas2)
                if naj1==0 and naj2==0: continue
                #
                # naj1=1, e.g., N_2 in [NH][NH][NH][NH]
                # naj1=2, e.g., N_2 in [NH][NH][NH]
                # naj1=3, e.g., N_2 in [NH][N]([CH][CH2])[NH]
                jss = []
                tasu = []
                jass = [jas1,jas2]; najs = [naj1,naj2]
                for i2 in range(2):
                    tasu_i = []; js = []
                    if najs[i2] > 0:
                        jas = jass[i2]
                        amap = amaps[i2]
                        iass = tas[i2]
                        #print 'jas,amap=',jas,amap
                        for j in jas:
                            for _ias in iass:
                                if amap[j] in _ias:
                                    js.append(j)
                                    tasu_i.append( _ias )
                    tasu.append(tasu_i)
                    jss.append(js)
                js1,js2 = jss
                iassu1,iassu2 = tasu

                nsg1 = len(iassu1); nsg2 = len(iassu2); nsgs = [nsg1,nsg2]
                nsg = nsg1+nsg2
                if nsg==1:
                    if nsg1==1: # N_2 (or N_3) in "[NH-][NH+]=[NH+][NH-]"
                        iassu = iassu1; js = js1
                    else: # OP(=O)=P(O)=P(O)=P(=O)O
                        iassu = iassu2; js = js2
                    _ias = iassu[0]
                    j = js[0]
                    # check if there exists a neighboring N5 atom with nsg==1
                    if self.check_N5_symm(j,dvs):
                        pair = [i,j]; pair.sort()
                        if pair not in pairs: pairs.append(pair)
                        istop = True
                else:

                    for q in range(2):
                        nsg = nsgs[q]
                        sg = sgs[q]
                        iassu = tasu[q]
                        js = jss[q]
                        for isg in range(nsg):
                            _ias = iassu[isg]
                            j = js[isg]
                            iok = False
                            na = len(_ias)
                            iok = self.is_not_conjugated(_ias,sg)
                            #print "* i, j, _ias, iok = ", i, j, _ias, iok
                            if iok:
                                pair = [i,j]; pair.sort()
                                if pair not in pairs: pairs.append( pair )

                    # Note that if a N3 is destinined to be a generic N5 env,
                    # there shoud be at least two neighboring atoms with dvi=1 (or -1),
                    # and each of this neighbor has to be involved in a subg
                    # satisfying `nr=0, i.e., a conjugated structure cannot
                    # be found for all the atoms in the subg
                    #
                    # E.g., if smiles is [CH2][N][C]([N]<)[NH][NH], for N_4, only
                    # one neighbor is involved in the aforementioned `subg, thus
                    # is not a N5 atom
                    lp = len(pairs)
                    if lp == 2:
                        istop = True
                        break
                    elif lp == 3:
                        # return this N3 and any of its neighbors; then we see what happens
                        pairs = pairs[:1]
                        istop = True
                        break
                    else:
                        pass
                    if istop: break
        #print ' -- pairs = ', pairs
        return pairs

    def find_basic_standalone_bonds(self, dvi, dvs):
        """ find standalone valency-unsaturated bonds """
        ias1 = self.ias[ dvs==dvi ]
        #cns_dvi = np.zeros(self.na)
        #filt = (np.abs(dvs) >= dvi)
        #cns_dvi[filt] = np.sum(self.g[filt][:,filt],axis=0)
        #ias1 = self.ias[ np.logical_and(cns_dvi==1, dvs==dvi) ] # standalone implies cn=1

        _pairs = [] # store all pairs

        for i in ias1:
            # In the following line, abs(dvj) is used to include
            # cases involing multivalent elements, e.g., S in
            # >S([O])[O] for which a reference valency of 2 is used
            # ======================================
            # note that there is some code in famon.f90
            # of the same functionality, but the main
            # difference is: here we used `abs(dvs[j])
            # instead of dvs[j] alone (in famon.f90)
            # due to the reason that in famon.f90, all
            # reference total valencies are exact, i.e.,
            # they are derived from query molecule, which
            # have well-defined valency.
            jas0 = self.ias[ self.g[i]>0 ]
            jas = jas0[ np.abs(dvs[jas0])>=dvi ]
            naj = len(jas)
            #print ' ** i,jas0,dvsj,jas=',i,jas0,dvs[jas0],jas
            if naj == 1:
                j = jas[0]
                _pair = [i,j]; _pair.sort()
                _pairs.append( _pair )
                #break # return one pair each time
        return _pairs

    def find_next_standalone_bonds(self, dvi, dvs):
        """ get one remaining valency-unsaturated bond with
        the constituting atom satisfying abs(dv) >= dvi
        """
        if dvi == 0:
            _pairs = self.find_bond_to_N5(dvs)
            bo = 2
        else:
            _pairs = self.find_basic_standalone_bonds(dvi, dvs)
            bo = dvi + 1
        if self.debug: print((' ##### when dvi=0, pairs = ', _pairs))
        pairs = []
        # check if some common atoms are shared in the pairs
        # Exceptions include e.g., RN(=O)=O
        #_pairs = np.array(_pairs, np.int)
        #n = len(_pairs)
        #iok = True
        #for i in range(n):
        #    for j in range(i+1,n):
        #        iac = set(_pairs[i]).intersection(set(_pairs[j]))
        #        if iac != set([]) and self.zs[list(iac)[0]] not in [7,15,16,17]:
        #            # e.g., for HN-[CH]-[N]-NH-[CH]-[CH]-[CH2] or HN-[C]1-[N]-NH-[CH]-[CH]-[CH]1
        #            # two pairs share the same atom N_3, i.e., (C_2,N_3) and (N_3,N_4)
        #            iok = False
        #            break
        #    if not iok: break

        # remove pair with opposite charges, e.g., >[C-][N+]R3
        for pair in _pairs:
            i,j = pair
            if dvs[i]+dvs[j] != 0: pairs.append( pair )
        #print ' * pairs_u = ', pairs_u
        return bo, pairs

    def recursive_update_dvi(self, tvs, bom):
        for dvi in [0,1,2]:
            # Below a "while" clause is used due to the fact that
            # a saturation of valence in one cycle may result
            # in a new set of atoms with the same value of `dvi
            # E.g., for >C-C-C-C<, in cycle 1, atom C1 & C4 have
            # dvi=1, after saturation, the SMILES becomes
            # >C=C-C=C<; in cycle 2, atom C2 & C3 have dvi = 1,
            # and the SMILES is now modified to >C=C=C=C<. Done!
            while 1:
                dvs = tvs - bom.sum(axis=0)
                bo,pairs = self.find_next_standalone_bonds(dvi, dvs)
                if len(pairs) == 0:
                    break
                for pair in pairs:
                    i, j = pair
                    bom[i,j] = bo # dvi+1
                    bom[j,i] = bo # dvi+1
        return bom


    def update_charges(self,tvs,bom,chgs):
        """update charge of
        1) standalone valency-unsaturated atom
        2) -C(=O)=O -> -C(=O)[O-]
        3) all substructure containing v_N = 5
           e.g., -N$C -> -[N+]#[C-]
                 -N(=O)=O, ...
           Note that only -N(=O)=O, =N#N can be
           converted automatically to charged form by
           RDKit itself; all other structures, e.g.,
           C=[NH]=O,C=[NH]=NH, ... cannot be recognized!
        """
        na = self.na
        dvs = tvs - bom.sum(axis=0)
        #bom_aux = copy.deepcopy(bom)
        tvs_aux = copy.deepcopy(tvs)
        pairs1 = [] # oppositely charged atomic pairs with bo > 1
        for i in range(na):
            dvi = dvs[i]
            zi = self.zs[i]
            jas = self.ias[self.g[i]>0]
            if dvi != 0:
                dvsj = dvs[jas]
                if np.all(dvsj==0):
                    #assert abs(dvi) == 1
                    filt6 = np.logical_or(self.zs[jas]==8, self.zs[jas]==16)
                    nn6 = filt6.sum()
                    #ncn1 = (self.dvs0[jas] == 1).sum()
                    if zi == 6:
                        if dvi==-1:
                            if nn6>=1:
                                # for R=O,S
                                #     -C(=O)=R -> -C(=O)[R-] (
                                #     [C](R3)=R -> C(R3)[R-]
                                jas1 = jas[filt6]
                                j = jas1[0]
                                chgs[j] = -1
                                bom[i,j] = bom[j,i] = 1
                                #bom_aux[i,j] = bom_aux[j,i] = 1
                                tvs_aux[j] += -1
                            else:
                                chgs[i] = -1 # >[C-]-
                                tvs_aux[i] += -1
                        elif dvi==2:
                            # -[C]-OR -> -[C-]=[O+]R
                            jas1 = jas[ self.zs[jas] == 8 ]
                            assert len(jas1) == 1
                            j = jas1[0]
                            chgs[i] = -1
                            chgs[j] = 1
                            bom[i,j] = bom[j,i] = 2
                            #bom_aux[i,j] = bom_aux[j,i] = 2
                            tvs_aux[i] += -1
                            tvs_aux[j] += 1
                        else:
                            print((' ** i, zi, dvi, dvs = ', i, zi, dvi, dvs))
                            print((' ** tvs = ', tvs))
                            raise Exception('#ERROR')
                    elif zi == 7: # >[N+]<
                        if dvi == 1: # -1:
                            chgs[i] = 1
                            tvs_aux[i] -= 1 # += 1
                        else:
                            print((' ** i, zi, dvi, dvs = ', i, zi, dvi, dvs))
                            raise #'#ERROR'
                    elif zi == 8: # [CH]OC -> [CH]#OC -> [CH-]=[O+]C
                        if dvi == -2:
                            jas1 = jas[self.dvs0[jas]>1] # np.logical_and(self.zs[jas]==6,self. ]
                            assert len(jas1) == 1
                            j = jas1[0]
                            chgs[i] = 1
                            chgs[j] = -1
                            bom[i,j] = bom[j,i] = 2
                            #bom_aux[i,j] = bom_aux[j,i] = 2
                            tvs_aux[i] += 1
                            tvs_aux[j] += -1
                    else:
                        print((' ** i, zi, dvi, dvs = ', i, zi, dvi, dvs))
                        raise #'#unknow case'
                else:
                    jas1 = jas[dvsj==-1*dvi]
                    if len(jas1) == 1: # >[C-][N+]R3
                        j = jas1[0]
                        pair = [i,j]; pair.sort()
                        if pair not in pairs1:
                            chgs[i] = -1 * dvi
                            chgs[j] = dvi
                            tvs_aux[i] += -1 * dvi
                            tvs_aux[j] += dvi
                            pairs1.append( pair )
        return bom, chgs, tvs_aux


    def get_pairs(self, tvs, bom):
        """ get pairs of atoms that have opposite charges """
        na = self.na
        dvs = tvs - bom.sum(axis=0)
        #bom_aux = copy.deepcopy(bom)
        chgs_aux = np.array([0]*na, np.int)
        pairs = []; bo_pairs = [] # oppositely charged atomic pairs with bo > 1
        for i in range(na):
            dvi = dvs[i]
            zi = self.zs[i]
            jas = self.ias[self.g[i]>0]
            zsj = self.zs[jas]
            if dvi == 0:
                if zi==7:
                    iok = False
                    #filt1 = np.logical_and(np.logical_or(zsj==6,zsj==8), self.cns[jas]==1)
                    if self.cns[i] == 2: # identify -N$C and -->  -[N+]#[C-]
                        jas1 = jas[ np.logical_and(zsj==6,self.cns[jas]==1) ]
                        if len(jas1)== 1:
                            iok = True
                    #elif self.cns[i] == 4: # identify >[C-][N+]R3
                    #    jas1 = jas[self.dvs0[jas]==1]
                    #    #print ' <<<<< ', i, filt1.sum()
                    #    if len(jas1) == 1:
                    #        iok = True
                    #else:
                    #    pass
                    if iok:
                        j = jas1[0]
                        boij = bom[i,j]
                        if boij > 1:
                            chgs_aux[i] = 1
                            chgs_aux[j] = -1
                            pair = [i,j]; pair.sort()
                            if pair not in pairs:
                                pairs.append(pair)
                                bo_pairs.append(boij-1)
        #print ' ***** ', chgs_aux
        return pairs, bo_pairs, chgs_aux


    def tocan(self, tvs, chgs, bom): #, c='indigo'):
        """
        """
        #print('bom=',bom)
        pairs, bo_pairs, chgs_aux = self.get_pairs(tvs, bom)
        for i,chg in enumerate(chgs):
            if chg != 0:
                assert chgs_aux[i] == 0
        if pairs is not None:
            for ipair,pair in enumerate(pairs):
                i,j = pair
                bom[i,j] = bom[j,i] = bo_pairs[ipair]
        can = 'None'
        chgs += chgs_aux

        #fmt = 'sdf'
        #if self.na < 100:
        #    blk = write_ctab(self.zs, chgs, bom, self.coords, isotopes=[], sdf=None)
        #    c = 'indigo'
        #else:
        blk = (self.zs, self.coords, chgs, bom)
        #    blk = write_pdb( (self.zs,self.coords,chgs,bom) )
        #    fmt = 'pdb'
        #    c = 'rdkit' # indigo cannot parse pdb file!!
        self.blk = blk
        #if c in ['rdkit']:
        #    fun = Chem.MolFromMolBlock if fmt in ['sdf'] else Chem.MolFromPDBBlock
        #    sanitize = True
        #    _tvs = bom.sum(axis=0)
        #    for ia in range(self.na):
        #        if _tvs[ia] == 5 and chgs[ia] == 0:
        #            sanitize = False
        #            break
        #    m = fun(blk,sanitize=sanitize)
        #    try:
        #        _can = Chem.MolToSmiles(m)
        #        # convert back to Indigo canonical smiles
        #        can = to_indigo_can(_can)
        #    except:
        #        raise '#ERROR: rdkit failed to convert?'
        #elif c in ['indigo']:
        #    obj = indigo.Indigo()
        #    m = obj.loadMolecule(blk)
        #    m.clearStereocenters()
        #    can = m.canonicalSmiles()
        #else:
        #    raise '#ERROR: unkown converter!'
        trial_and_error = F
        if trial_and_error:
            try:
                m = coc.newmol(self.zs, chgs, bom, self.coords) #rawmol_indigo(blk)
                can = m.can
            except:
                print('#ERROR: conversion to canonical smiles from sdf/pdb file failed')
                raise
        else:
            m = coc.newmol(self.zs, chgs, bom, self.coords)  #rawmol_indigo(blk)
            can = m.can #tocan()
        return can


    def reconnect(self,tvs,bom):
        """ re-connect bond for which bo was set to 0 when perceiving g
        E.g., for a SMILES string OC12C3C4C1N4C32, angle(N_6,C_4,C_5) ~ 48 degree,
              so the longest bond (N_6,C_4) was revered when calling connect() in
              class RawMol. The consequence is that N_6 & C_4 have an valency of 1
              deficient. Now reconnect these two atoms
        """
        dvs = tvs - bom.sum(axis=0)
        for i in range(self.na):
            dvi = dvs[i]
            zi = self.zs[i]
            if (zi in [6,7]) and dvi==1:
                jas0 = self.ias[ np.logical_and(self.g0[i]>0, dvs==1) ]
                if len(jas0)==1:
                    j = jas0[0]
                    if bom[i,jas0[0]] == 0:
                        bom[i,j] = bom[j,i] = 1
                        dvs = tvs - bom.sum(axis=0)
        return bom


    def perceive_bond_order(self):
        """ bond order """
        zs = self.zs
        g = self.g
        na = self.na
        ias = self.ias
        vsr = self.pt.vsr
        cns = g.sum(axis=0)
        cns_heav = np.array([np.logical_and(g[i]>0,zs>1).sum() for i in range(na)], np.int)
        self.cns = cns
        self.cns_heav = cns_heav

        # now initialize `bom and then update it
        bom1 = copy.deepcopy(g) # later change it to bond-order matrix
        tvs1 = copy.deepcopy(vsr)

        can = 'None'
        dvs = tvs1 - bom1.sum(axis=0)
        self.dvs0 = copy.deepcopy(dvs)

        chgs = np.array([0,]*na, np.int)
        if np.all(dvs==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        # locate pair of atoms with opposite charges
        # of type [A;dvi=1]~[B;dvi=-1], e.g., >[C-][N+]R3
        obsolete = """pairs11 = []
        for i in range(na):
            dvi = dvs[i]
            if dvi in [1,-1]:
                jas = ias[self.g[i]>0]
                filt = (dvs[jas] == -1 * dvi)
                if np.any(filt):
                    if filt.sum() != 1:
                        print ' dvs = ', dvs
                        print '#ERROR: more than one pair of (+1,-1)??', i,jas[filt]
                        sys.exit(2)
                    j = jas[filt][0]
                    bom1[i,j] += 1 # later update charge
                    pair = [i,j]; pair.sort()
                    pairs11.append(pair) """

        # fix two cases:
        # 1) ~N-C, ~[X]-O, ~[X]-N, ~[X]-S, ~[X]-P
        # 2) ~[XH_n], e.g., =[NH],=[CH2], =[SiH2], =[PH], ...
        # they all have cn_heav = 1
        visited = []
        for i in range(na):
            zi = zs[i]
            dvi = dvs[i]
            if zi > 1:
                if cns_heav[i]==1 and dvi>0:
                    j = ias[np.logical_and(zs>1,g[i]>0)][0]
                    pair = [i,j]; pair.sort()
                    if pair not in visited:
                        boij = bom1[i,j]
                        bom1[i,j] = bom1[j,i] = boij + dvi
                        visited.append(pair)

        debug = F
        self.debug = debug
        if debug:
            print((' *1 zs = ', zs))
            print((' *1 tvs1 = ', tvs1))
            print((' *1 bom1.sum(axis=0) = ', bom1.sum(axis=0)))
            print((' *1 g = ', g, ', nb=', (g>0).sum()/2))
            print((' *1 dvs = ', tvs1-bom1.sum(axis=0)))

        if np.all(tvs1-bom1.sum(axis=0)==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        # fix all valencies of atoms that have dvi == 1 or 2
        _bom = copy.deepcopy(bom1)
        bom1 = self.recursive_update_dvi(tvs1, _bom)
        dvs = tvs1 - bom1.sum(axis=0)

        if debug:
            print((' *2 zs = ', zs))
            print((' *2 tvs1 = ', tvs1))
            print((' *2 bom1.sum(axis=0) = ', bom1.sum(axis=0)))

        if np.all(tvs1-_bom.sum(axis=0)==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        _tvs = [ [vi] for vi in tvs1 ]
        # now update the values of `_tvs1 for multi-valent atom
        for ia in range(na):
            zi = self.zs[ia]
            dvi = dvs[ia]
            msg = 'unknown valence:  ia=%d, zi=%d, dvi=%s'%(ia,zi,dvi)
            if zi in [7,]:
                # dv = 3-4 = -1
                #    -[N*]-[N] (new: -[N*]#N)
                #    -NX3
                # dv = 3-5 = -2
                #    -N([O])[O] (new: R-N(=O)=O)
                #    -N([C])  (new: R-N$C)
                #    >C-[N*]-[N] (new: >C=N#N)
                if dvi < 0:
                    if dvi == -1:
                        if cns[ia] in [3,2]: # cn=3: >N[O] (new: >N=O; eventually: =[NR]=O)
                            _tvs[ia] = [5]
                        elif cns[ia] == 4:
                            #if np.any(ia==np.array(pairs11,np.int)):
                            _tvs[ia] = [5]
                            #else:
                            #    pass # >[N+]< (correct charge later) #pass #_tvs[ia] = [3]
                        else:
                            print(msg)
                            raise #'#ERROR'
                    elif dvi == -2:
                        _tvs[ia] = [5]
                    else:
                        print(msg)
                        raise #'#ERROR'
                else: #if dvi > 0:
                    pass
            elif zi in [15]:
                # dv = 3-5 = -2
                #     PR5
                #     >P(=O)-
                # dv = 3-4 = -1
                #     P-[P*]#P
                if dvi < 0:
                    if dvi in [-1,-2]:
                        _tvs[ia] = [5]
                    else:
                        print(msg)
                        raise #'#ERROR'
                elif dvi > 0:
                    print(msg)
                    raise #'#ERROR'
            elif zi in [16,]:
                # dvi = 2-3= -1
                #    >S[C]<
                #    >[C]S([C]<)[C]<, =[C](-R)S([C](R)(=R'))=C(R)R'
                # dvi = 2-4 = -2
                #    >S[O] (new: >S=O)
                #    >S([C]<)[C]<
                # dvi = 2-5 = -3
                #    -S([O])[O] (new: -S(=O)=O)
                #    >S([C]<)[O] (new: >S([C]<)=O)
                # dvi = 2-6 = -4
                #    >S([O])[O] (new: >S(=O)=O)
                #    [O]S([O])[O] (new: O=S(=O)=O)
                if dvi <= -3:
                    _tvs[ia] = [6]
                elif dvi == -2:
                    if cns[ia] == 3:
                        # >S=O, >S=C<, ..
                        _tvs[ia] = [4]
                    elif cns[ia] == 4:
                        _tvs[ia] = [6]
                    else:
                        print(msg)
                        raise #'#ERROR'
                elif dvi == -1:
                    # =========================
                    _tvs[ia] = [6,4]
                    # =========================
                elif dvi == 0:
                    pass # =S
                else:
                    print(msg)
                    raise #'#ERROR'
            else:
                pass # normal cases

        #print zs
        #print dvs
        #print _tvs

        nrmax = na/2
        nbmax = (g>0).sum()/2
        icon = False
        # if any of tvs[i] is a list, multiple choices of
        # `tvs exist, we have to try each of them
        tvs = cm.products(_tvs)
        istat = False
        for tvsi in tvs:
            _bom1 = copy.deepcopy(bom1)
            _bom2 = self.reconnect(tvsi,_bom1)
            _bom,_chgs,tvsi_aux = self.update_charges(tvsi,_bom2,chgs)

            if debug:
                print(_bom)
                print((' * tvsi = ', tvsi))
                print((' * tvsi_aux = ', tvsi_aux))
                print((' * _bom.sum() = ', _bom.sum(axis=0)))
                print((' * _chgs = ', _chgs))
            #if np.all(tvsi+_chgs-_bom.sum(axis=0)==0):
            if np.all(tvsi_aux-_bom.sum(axis=0)==0):
                can = self.tocan(tvsi, _chgs, _bom)
                return can

            # update bom
            iok, bom = cf.update_bom(nrmax,nbmax,zs,tvsi_aux,_bom,icon)
            if iok: # and np.all(tvsi+chgs-bom.sum(axis=0)==0):
                istat = True
                break

        if istat:
            can = self.tocan(tvsi_aux,_chgs,bom)
        else:
            print(' [failure]')
        return can

def xyz2sdf(zs, coords, fsdf=None):
    """ xyz to sdf """
    m = Mol(zs, coords, ican=True)
    zs, coords, chgs, bom = m.blk
    if fsdf is None:
        fsdf = tpf.NamedTemporaryFile(dir='/tmp').name + '.sdf'
    write_ctab(zs, chgs, bom, coords, sdf=fsdf)
    return fsdf

## test!

if __name__ == "__main__":
    import ase, sys
    import ase.io as aio

    import aqml.cheminfo.rdkit.core as cir
    from aqml.cheminfo.core import *

    args1 = sys.argv[1:]
    idx = 0

    if '-h' in args1:
        sys.exit( 'tosmiles [-oechem] *.xyz' )

    can_fmt = 'oechem'
    if '-indigo' in args1:
        can_fmt = 'indigo'
        idx += 1

    trial = True
    if '-d' in args1 or '-debug' in args1:
        trial = False
        idx += 1

    # write sdf file?
    sdf = False
    if '-sdf' in args1:
        sdf = True
        idx += 1

    args = args1[idx:]
    n = len(args)
    if n == 0:
        objs = ["C=C=S(=C=C)=[N+]=[N-]", \
                "S1(=C)(C)=CC=CC=C1", \
                "[N+]1([O-])=CC=C[NH]1", \
                "C[N+](=O)[O-]", \
                "C=[NH+][O-]", \
                "C[N+]#[C-]", \
                "C[NH2+][C-](C)C", \
                "[NH3+]CC(=O)[O-]", \
                "C[O-]",\
                "C[NH3+]",\
                "[CH-]=[O+]C", \
                "N=[NH+][NH-]", \
                "[NH-][NH+]=C1C=C(C=C)C=C1", \
                "OP(=S)=P(=[PH2]C)C", \
                "O[N+]([O-])=[N+]([N-]C)O", \
                "OC12C3C4C1N4C32"] # the last one is highly strained, may have problem in acquring g0
    elif n == 1:
        f = args[0]
        if f[-3:] in ['smi','can']:
            objs = [ si.strip() for si in file(f).readlines() ]
        else:  # either an xyz file or a SMILES string
            objs = args
    else:
        objs = args

    isf = False
    nobj = len(objs)
    for i,obj in enumerate(objs):
        if not os.path.isfile(obj):
            f = tpf.NamedTemporaryFile(dir='/tmp').name + '.xyz'
            m0 = cir.RDMol(obj, doff=True)
            m0.write_xyz(f)
        else:
            isf = True
            f = obj
        o = cc.molecule(f, isimple=T)
        can = 'None'
        iok = T
        if nobj > 1:
            if trial:
                try:
                    m = Mol(o.zs, o.coords, ican=True)
                    can = m.can
                except:
                    iok = F #print(' conversion failed!')#pass
            else:
                m = Mol(o.zs, o.coords, ican=True)
                can = m.can
        else:
            print(f)
            m = Mol(o.zs, o.coords, ican=True)
            can = m.can

        if (can != 'None') and (can_fmt in ['oechem',]):
            from openeye import oechem
            oem = oechem.OEGraphMol()
            assert oechem.OESmilesToMol(oem, can)
            can = oechem.OECreateSmiString(oem, oechem.OESMILESFlag_Canonical)

        s1 = '' if iok else ' [ conversion failed ]'
        if isf:
            print( i+1, f, can, s1 )
        else:
            print( i+1, f, obj, can, s1 )

        if sdf:
            zs, coords, chgs, bom = m.blk
            if m.na < 100:
                sdf = f[:-4]+'.sdf'
                write_ctab(zs, chgs, bom, coords, sdf=sdf)
            else:
                pdb = f[:-4]+'.pdb'
                write_pdb(m.blk, pdb)


