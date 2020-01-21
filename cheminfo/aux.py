
"""
auxiliary tools for creating conjugated polymers, etc
So far, we support three variants:
1) benzene drived, e.g., c6h6, two c6h6 sharing 2 carbon atoms (i.e., c10h8),
                         Three Ph rings (any of the two neighboring rings share two C atoms),
                         ......
                         c6h5n, ......
2) c4h5n and its derivation
   c4h4o, ...
   c4h4s, ...
3) (-PhNH-)n

"""

import os,sys,ase
import ase.io as aio
import cheminfo.OEChem as cio
import cheminfo.RDKit as cir
from openeye.oechem import *
import numpy as np
from rdkit import Chem

class Polymer(object):

    def __init__(self, n, base, fname, za0=6, ff='mmff94'):
        """
        n       -- numumber of new rings (or conjugate unit)
        tv      -- translation vector, can be a genuine vector (e.g., [1,1,1])
                   or defined by two atoms (atom indices must be given)
        tias    -- idx of atoms to be translated by `tv
        replace -- [ idx, zi_new, num_implicit_h_new ]
        """

        self.base = base
        self.n = n
        self.fname = fname
        home = os.environ['HOME']
        bsdf = '%s/Dropbox/workspace/python/cheminfo/data/molecules/%s.sdf'%(home,base)
        obj = cio.StringM(bsdf, suppressH=True) # Base Geometry
        coords = obj.coords
        zs = obj.zs
        bom = obj.bom # bond order matrix
        bm = obj.oem

        #if len(tv) == 3:
        #    tv = np.array(tv)
        #else:
        if base in ['c6h6', 'benzene']:
             # -------------------------
             # only one monomer in one repeating Unit
             # -------------------------
            i1, i2 = [0,4]
            tv = coords[i2] - coords[i1]
            replace = [5, za0, 0] # [idx, zi_new, nih]
            # u might be interested in replacing some atoms
            nar = len(replace)
            if nar > 0:
                # note that atomic index in `replace starts from 1
                assert nar == 3
                iar, za, nih = replace
                for ai in bm.GetAtoms():
                    ia = ai.GetIdx()
                    if ia == iar:
                        break
                zs[ia] = za
                ai.SetAtomicNum( za )
                ai.SetImplicitHCount( nih )
        elif base in ['c4h5n', 'pyrrole', 'thiophene','PEDOT', 'c4h4s']:
             # -------------------------
             # two monomers in one repeating Unit
             # -------------------------
            i1,i2 = [3,13]
            tv = coords[i2] - coords[i1]

        elif base in ['aniline', 'c6h7n']:
            i1,i2 = [0,14]
            tv = coords[i2] - coords[i1]
        else:
            raise '#ERROR: no such `base molecule'


        vs0 = [ ai.GetDegree() for ai in bm.GetAtoms() ]
        vs = [ vi for vi in vs0 ]
        m = OEGraphMol()
        atoms = []

        if base in ['c6h6','benzene']:
            for i,zi in enumerate(zs):
                new_atom = m.NewAtom( zi )
                m.SetCoords( new_atom, coords[i] )
                atoms.append( new_atom )
            es = [ [0,1],[2,3],[4,5], [1,2],[3,4],[5,0] ]
            bos =   [  2,    2,    2,     1,    1,    1    ]
            for i,ei in enumerate(es):
                m.NewBond( atoms[ei[0]], atoms[ei[1]], bos[i] )

            assert n >= 2
            tias = [2,3,4,5] # idx starting from 0
            p,q,r,s = tias
            nau = len(tias) # num_atoms_per_repeating_unit
            for i in range(1,n):
                ias_i = []
                for ia0 in tias:
                    ia = ia0 + i*nau
                    ias_i.append( ia )
                    new_atom = m.NewAtom( zs[ia0] )
                    vs.append( vs0[ia0] )
                    coords_i = coords[ia0] + i*np.array(tv)
                    m.SetCoords( new_atom, coords_i )
                    atoms.append( new_atom )
                p2,q2,r2,s2 = ias_i
                es_i = [ [q,p2],[r,s2],  [q2,r2], [p2,q2],[r2,s2] ]
                bos_i = [  1,     1,       1,       2,      2      ]
                for j,ej in enumerate(es_i):
                    m.NewBond( atoms[ej[0]], atoms[ej[1]], bos_i[j] )
                p,q,r,s = p2,q2,r2,s2
        else:
            if base in ['c4h5n', 'pyrrole', 'thiophene','PEDOT', 'c4h4s']:
                case = 1
                es1 = [ [0,1],[1,2],[2,3],[3,4],[4,0] ] # unit 1 in PUC
                bos1 = [ 2,    1,    2,    1,    1    ]
                es2 = [ [5,6],[6,7],[7,8],[8,9],[9,5] ] # unit 2 in PUC
                bos2 = [ 2,    1,    2,    1,    1    ]
                es3 = [ [0,5] ] # bond linking the two units in PUC
                bos3 = [ 1,   ]
                tias1 = [0,1,2,3,4]
                tias2 = [5,6,7,8,9]
            elif base in ['aniline', 'c6h7n']:
                case = 2
                es1 = [ [0,1],[1,2],[2,3],[3,4],[4,5],[5,0], [3,6] ] # unit 1 in PUC
                bos1 = [ 2,    1,    2,    1,    2,   1,     1    ]
                es2 = [ [7,8],[8,9],[9,10],[10,11],[11,12],[12,7], [10,13] ] # unit 2 in PUC
                bos2 = [ 2,    1,    2,    1,      2,      1,      1   ]
                es3 = [ [6,7] ] # bond linking the two units in PUC
                bos3 = [ 1,   ]
                tias1 = [0,1,2,3,4,5,6]
                tias2 = [7,8,9,10,11,12,13]

            tias = tias1 + tias2
            for ia in tias:
               new_atom = m.NewAtom( zs[ia] )
               m.SetCoords( new_atom, coords[ia] )
               atoms.append( new_atom )

            es = es1 + es2 + es3; bos = bos1 + bos2 + bos3
            for i,ei in enumerate(es):
                m.NewBond( atoms[ei[0]], atoms[ei[1]], bos[i] )

            # note that there are 3 units in file `c4h5n.sdf
            # and the third unit is only used for calc `tv
            assert n >= 3

            if case == 1:
                p1,q1,r1,s1,t1 = tias1
                p2,q2,r2,s2,t2 = tias2
            else: # == 2
                p1,q1,r1,s1,t1,u1,v1 = tias1
                p2,q2,r2,s2,t2,u2,v2 = tias2
            nau = len(tias1) + len(tias2) # num_atoms_per_repeating_unit
            n1 = (n-1)/2 + 1# n = 3 or 4, n1 -> 2; ...

            for i in range(1,n1):
                ias1 = []
                for ia0 in tias1:
                    ia = ia0 + i*nau
                    ias1.append( ia )
                    new_atom = m.NewAtom( zs[ia0] )
                    vs.append( vs0[ia0] )
                    coords_i = coords[ia0] + i*np.array(tv)
                    m.SetCoords( new_atom, coords_i )
                    atoms.append( new_atom )

                # if number of monomers is odd, remove the last half of atoms appended
                if not (n%2 == 1 and i == n1-1):
                  ias2 = []
                  for ia0 in tias2:
                    ia = ia0 + i*nau
                    ias2.append( ia )
                    new_atom = m.NewAtom( zs[ia0] )
                    vs.append( vs0[ia0] )
                    coords_i = coords[ia0] + i*np.array(tv)
                    m.SetCoords( new_atom, coords_i )
                    atoms.append( new_atom )

                if case == 1:
                    p1u,q1u,r1u,s1u,t1u = ias1
                    es1 = [ [p1u,q1u],[q1u,r1u],[r1u,s1u],[s1u,t1u],[t1u,p1u],  [s2,s1u], ]
                    bos1 = [ 2,       1,         2,       1,        1,          1,     ]
                    for j,ej in enumerate(es1):
                        m.NewBond( atoms[ej[0]], atoms[ej[1]], bos1[j] )
                    p1,q1,r1,s1,t1 = p1u,q1u,r1u,s1u,t1u
                    #print '   -- unit 1 done'

                    # if number of monomers is odd, remove the last half of atoms appended
                    if not (n%2 == 1 and i == n1-1):
                        p2u,q2u,r2u,s2u,t2u = ias2
                        es2 =  [ [p2u,q2u],[q2u,r2u],[r2u,s2u],[s2u,t2u],[t2u,p2u],  [p1u,p2u], ]
                        bos2 = [ 2,       1,         2,       1,        1,          1,     ]
                        for j,ej in enumerate(es2):
                            m.NewBond( atoms[ej[0]], atoms[ej[1]], bos1[j] )
                        p2,q2,r2,s2,t2 = p2u,q2u,r2u,s2u,t2u
                        #print '   -- unit 2 done'
                else: # 2
                    p1u,q1u,r1u,s1u,t1u,u1u,v1u = ias1
                    es1 = [ [p1u,q1u],[q1u,r1u],[r1u,s1u],[s1u,t1u],[t1u,u1u],[u1u,p1u],  [v1u,s1u],    [v2,p1u] ]
                    bos1 = [ 2,       1,         2,       1,        2,          1,        1,            1        ]
                    for j,ej in enumerate(es1):
                        m.NewBond( atoms[ej[0]], atoms[ej[1]], bos1[j] )
                    p1,q1,r1,s1,t1,u1,v1 = p1u,q1u,r1u,s1u,t1u,u1u,v1u

                    # number of monomers is odd, remove the last half of atoms appended
                    if not (n%2 == 1 and i == n1-1):
                        p2u,q2u,r2u,s2u,t2u,u2u,v2u = ias2
                        es2 =  [ [p2u,q2u],[q2u,r2u],[r2u,s2u],[s2u,t2u],[t2u,u2u],[u2u,p2u],  [v2u,s2u],   [v1u,p2u]]
                        bos2 = [ 2,       1,         2,       1,        2,          1,        1,            1        ]
                        for j,ej in enumerate(es2):
                            m.NewBond( atoms[ej[0]], atoms[ej[1]], bos1[j] )
                        p2,q2,r2,s2,t2,u2,v2 = p2u,q2u,r2u,s2u,t2u,u2u,v2u


        OEFindRingAtomsAndBonds(m)

        # update nih of two C atoms to which new unit is to be appened
        #print ' n, vs = ', len(vs), vs
        for ai in m.GetAtoms():
            i = ai.GetIdx()
            di = ai.GetDegree()
            #print i, vs[i], di
            ai.SetImplicitHCount( vs[i]-di )

        OEAssignAromaticFlags(m, OEAroModel_OpenEye)
        smarts = OECreateSmiString(m, OESMILESFlag_Canonical)
        print smarts

        #OEAddExplicitHydrogens( m )

        self.m = m

        sdf = fname + '_n%02d_old.sdf'%n
        zso = []; pso = []
        for ai in m.GetAtoms():
            zso.append( ai.GetAtomicNum() )
            pso.append( m.GetCoords(ai) )

        #aseobj = ase.Atoms(zso, pso)
        #aio.write(fn, aseobj)
        na = len(zso)
        bom_u = cio.oem2g(m)
        cio.write_sdf_raw(zso, pso, bom_u, np.zeros(na), sdf)


        # further add H's using RDKit
        mr = Chem.MolFromMolFile(sdf)
        mr = Chem.AddHs(mr)
        obj = cir.RDMol(mr, forcefield=ff, doff=True)
        sdf2 = fname + '_n%02d.sdf'%n
        obj.write_sdf( sdf2 )

