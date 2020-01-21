#!/usr/bin/env python

"""
Enumerate subgraphs & get amons
"""

import cheminfo as ci
import cheminfo.math as cim
import cheminfo.graph as cg
import cheminfo.openbabel.amon_f as cioaf
from cheminfo.rw.ctab import write_ctab
import networkx as nx
#import cheminfo.fortran.famon as fm
from itertools import chain, product
import numpy as np
import os, re, copy
#from rdkit import Chem
import openbabel as ob
import pybel as pb

global dic_smiles
dic_smiles = {6:'C', 7:'N', 8:'O', 14:'Si', 15:'P', 16:'S'}


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
        vars.__init__(self, bosr, zs, chgs, tvs, g, coords)
        self.use_bosr = use_bosr

    def update_m(self, once=True, debug=False):
        g = self.g
        chgs = self.chgs
        bosr = self.bosr
        tvs = self.tvs # `tvs has been modified according to `chgs
        zs = self.zs
        na = len(zs)
        ias = np.arange(na)

        bom = copy.deepcopy(g) # set bom as `g
        cns = g.sum(axis=0)
        vs = bom.sum(axis=0)
        dvs = tvs - vs
#        print '     vs = ', vs
#        print '    tvs = ', tvs
#        print '    cns = ', cns

        # 1) for =O, =S
        ias_fringe = ias[ np.logical_and(dvs == 1, \
                 np.logical_or(self.zs == 8, self.zs == 16) ) ]
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 2
        vs = bom.sum(axis=0)
        dvs = tvs - vs
#       if iprt: print ' *** dvs1 = ', dvs

        # 2) for #N
        ias_fringe = ias[ np.logical_and(dvs == 2, self.zs == 7) ]
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 3
        vs = bom.sum(axis=0)
        dvs = tvs - vs

        # 3) for $C (quadruple bond)
        ias_fringe = ias[ np.logical_and(dvs == 3, self.zs == 6) ]
        for ia in ias_fringe:
            jas = ias[ g[ia] > 0 ]
            ja = jas[0]
            bom[ia, ja] = bom[ja,ia] = 4
        vs = bom.sum(axis=0)
        dvs = tvs - vs
#       print ' -- dvs = ', dvs

        ## 4) fix special cases, where we consider
        #     the middle N has a valence of 5
        ##    -N-N-N  -->  -N=N#N
        ##    >C-N-N  -->  >C=N#N
        # ==============================
        # now it's not necessary to do this
        # as step 2) and step below suffice

        ## 5) fix special cases
        ##    -C(O)O  -->  -C(=O)O
        # ==============================
        # necessary only for charged species

        # 6) fix cases like >C-C-C< or =C-C-C< or -N-C-C< (dvs=[1,2,1])
        #                   >C-C-C-C< (dvs=[1,2,2,1])
        #                   >C-C-C-C-C< (dvs=[1,2,2,2,1])
        #    cases like >C-C(-X)-C-C-C(-X)-C< must be excluded (note
        #    that [1,2,2,1] is a subset of `dvs for all atoms)
        filt1 = (dvs == 1)
        zs1 = zs[filt1]
        ias1 = ias[filt1]; na1 = len(zs1)
        filt2 = (dvs == 2)
        zs2 = zs[filt2]
        ias2 = ias[filt2]; na2 = len(zs2)
        if na2 > 0:
            g2 = bom[ias2,:][:,ias2]
            for ias2c_raw in cg.find_cliques(g2):
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
        # now update `dvs
        vs = bom.sum(axis=0)
        dvs = tvs - vs

#       if iprt: print ' ***** dvs = ', dvs

        combs = [ ]; bos = [ ]
        nclique = 0
        iok = True # assume valences are all ok for atoms
        for dv in [2, 1]:
            # for C_sp2, dv = 4-3 = 1, N_sp2, dv = 3-2 = 1;
            # for C_sp, dv = 4-2 = 2
            # for O_sp2, dv = 2-1 = 1
            BO = {2: 3, 1: 2}[dv]
            # atoms to be re-bonded by double/triple bonds
            ias_dv = ias[ dvs == dv ]
            na_dv = len(ias_dv)
            if na_dv > 0:
                g2 = g[ias_dv,:][:,ias_dv]
                cliques_dv = cg.find_cliques(g2)
#               print ' ***** dv, cliques_dv = ', dv, cliques_dv
#               nclique_dv = len(cliques_dv)
#               nclique += nclique_dv
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
                    if nc_i%2 == 1: iok = False; continue
#                   ifound = True
#                   if nc_i == 1:
#                       ifound = False
                    if nc_i == 2:
                        # use relative indices; later I will convert
                        # them to absolute ones
                        ipss_dv_0 = [  [[0,1],],  ]
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
                        # sequentially, i.e., the resulting `ipss_dv_0 is [[0,1], ]
                        # which should be instead [ [4,1],[2,3] ]
                        #
                        # Thus, `once should be set to False when calling
                        #  `find_double_bonds
                        ipss_dv = find_double_bonds(g2c, once=False, irad=False)
                        n_ = len(ipss_dv)
                        if n_ == 0:
                             # e.g., edges = [[3,1],[3,2],[3,4]] and dvs = [1,1,1,1]
                            iok = False; continue
                        nmax_ = 2
#                       print '\n *** dv, cliques_i = ', dv, cliques_i
#                       print ' *** g2c = ', g2c
#                       print ' *** nc_i, n_ = ', nc_i, n_
                        for i_ in range(n_):
                            ias_ = ipss_dv[i_]
                            ni_ = len(np.ravel(ias_))
#                           print ' *** i_, ni_, ias_ = ', i_, ni_, ias_
                            if ni_ >= nmax_:
                                nmax_ = ni_
                                i_chosen = i_
#                               print ' *** ', i_chosen, ipss_dv[ i_chosen ]
                        if once:
                            # choose only 1 from all options
                            ipss_dv_0 = [ ipss_dv[ i_chosen ], ]
                        else:
                            ipss_dv_0 = ipss_dv
                    #if ifound:
                    map_i = list( np.array(ias_dv, np.int)[cliques_i] )
                    bonds_dv_i = [ ]
                    for iips in ipss_dv_0:
                        cisU = [ [ map_i[jas[0]], map_i[jas[1]] ] \
                                       for jas in  iips ]
                        bonds_dv_i.append( cisU )

                    combs.append( bonds_dv_i )
                    bos.append( BO )
        _boms = []
        if iok:
            if len(combs) > 0: # nclique >= 1:
                for bs in cim.products(combs):
                    bom_i = copy.copy(bom)
                    for i,bsi in enumerate(bs):
                        for bi in bsi:
                            ia1, ia2 = bi
                            bom_i[ia1,ia2] = bom_i[ia2,ia1] = bos[i]
                    _boms.append(bom_i)
            else:
                _boms.append( bom )

        cans = []; ms = []
        for bom in _boms:

            # note that the order of calling `get_bos() and `accommodate_chgs()
            #  matters as `bosr was obtained based on modified `bom, i.e., all
            # pairs of charges (the relevant two atoms are bonded) were eliminated
            bos = cioaf.get_bos(bom)


            # now restore charges for case, e.g., NN bond in C=N#N
            bom_U = cioaf.accommodate_chgs(chgs, bom)


            # for query molecule like -C=CC#CC=C-, one possible amon
            # is >C-C-C-C< with dvs = [1,2,2,1] ==> >C=C=C=C<, but
            # apparently this is not acceptable!!
            if self.use_bosr:
                if np.any(bos[zs>1] != bosr):
                    #print ' bosr = ', bosr, ', bosr0 = ', bos[zs>1]
                    continue

            isotopes = []
            zsmv = [7,15,16,17]
            vsn = [3,3,2,1]
            print ' -- zs = ', zs
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
            blk = write_ctab(zs, chgs, bom_U, self.coords, isotopes=isotopes, sdf=None)
            m = cioaf.obconv(blk)
            self.blk = blk
            can_i = pb.Molecule(m).write('can').split('\t')[0]

            # remove isotopes
            sp = r"\[[1-3]H\]"
            sr = "[H]"
            _atom_name_pat = re.compile(sp)
            can_i = _atom_name_pat.sub(sr, can_i)

            if can_i not in cans:
                cans.append(can_i)
                ms.append(m)
        return cans, ms



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
            m0 = cioaf.obconv(s,s[-3:])

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
            assert cioaf.check_hydrogens(m0), '#ERROR: some hydrogens are missing'
            coords0 = cioaf.get_coords(m0)
            pym = pb.Molecule(m0).clone
            # check consistency
            if pym.charge != 0 and (not allow_charge): iok = False
            if pym.spin > 1 and (not allow_radical): iok = False

            m = pym.OBMol; m.DeleteHydrogens()
        else:
            # assume SMILES string
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

            m = cioaf.obconv(s,'smi')
            mt = pb.Molecule(m)
            #print ' -- type(mt) = ', type(mt)
            pym = mt.clone
            if not allow_radical:
                if pym.spin > 1: iok = False

            if not allow_charge:
                #if pym.charge != 0:

                # now remove charge
                su = cioaf.remove_charge(m)
                m = cioaf.obconv(su,'smi')
            m0 = cioaf.clone(m)
            m0.AddHydrogens()

        if iok:
            zs = [ ai.atomicnum for ai in pym.atoms ]
            if not cioaf.check_elements(zs): iok = False

        self.iok = iok
        if iok: self.objQ = cioaf.mol(m0)

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
        for seed in cioaf.generate_subgraphs(b2a, a2b, self.k):
            # lasi (lbsi) -- the i-th list of atoms (bonds)
            lasi, lbsi = list(seed.atoms), list(seed.bonds)
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
                can = cioaf.chemical_symbols[ zi ]
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
                ## '-C=[N+]=[N-]' or '-N=[N+]=[N-]'
                ## ( however, '-Cl(=O)(=O)(=O)' cannot be
                ## recognized by rdkit )
                ## are retained if they are part of the query molecule
##### lines below are not necessary as `bosr will be used to assess
##### if the local envs have been kept!
                obsolete = """tvs1 = [4,6,5,5]
                dbns1 = [1,2,1,2] # number of double bonds
                zs1 = [16,16,15,7]
                istop = False
                for j,tvj in enumerate(tvs1):
                    filt = np.logical_and(tvsi0 == tvj, zsi == zs1[j])
                    jas = ias[filt].astype(np.int)
                    if len(jas) > 0:
                        dbnsj = dbnsi[jas]
                        if np.any(dbnsj != dbns1[j]):
                            istop = True; break
#                       print 'tvj, zs1[j], dbnsj, dbns1[j] = ', tvj, zs1[j], dbnsj, dbns1[j]
                if istop: continue"""


                self.get_subm(lasi, lbsi, sg_heav)
                vr = self.vars
                cmg = MG( vr.bosr, vr.zs, vr.chgs, vr.tvs, vr.g, vr.coords )

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


                #if len(objQ.zs[lasi])==3:
                #    if np.all(objQ.zs[lasi] == np.array([7,7,7])): print '## we r here'




                cans_i = []
                cans_i, ms_i = cmg.update_m(debug=True)
                for can_i in cans_i:
                    if can_i not in cans:
                        cans.append( can_i )
        return cans




def is_edge_within_range(edge, edges, dsg, option='adjacent'):

    i1,i2 = edge
    iok = False
    if len(edges) == 0:
        iok = True
    else:
        jit = is_joint_vec(edge, edges) # this is a prerequisite
        #print ' -- jit = ', jit
        if not jit:
            for edge_i in edges:
                j1,j2 = edge_i
                ds = []
                for p,q in [ [i1,j1], [i2,j2], [i1,j2], [i2,j1] ]:
                    ds.append( dsg[p,q] )
                #print ' -- ds = ', ds
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
    dsg = np.zeros((n,n), np.int)
    Gnx = nx.Graph(g)
    for i in range(n):
        for j in range(i+1,n):
            dsg[i,j] = dsg[j,i] = nx.shortest_path_length(Gnx, i,j)
#   dsg = fm.dijkstra(g)

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



## test!

if __name__ == "__main__":
    import sys, gzip

    args = sys.argv[1:]
    nargs = len(args)
    if nargs == 0:
        ss = ["[NH3+]CC(=O)[O-]", "CC[N+]([O-])=O", \
             "C=C=C=CC=[N+]=[N-]", "CCS(=O)(=O)[O-]"] # test molecules
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
        print ' ## %s'%s
        if not os.path.exists(s):
            obj = amon(s, k)
            cans = obj.get_amons()
            for can in cans:
                print can
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
