#!/usr/bin/env python

import numpy as np
import openbabel as ob
import pybel as pb
import os, sys, copy
from aqml.cheminfo.rw.ctab import write_ctab
import scipy.spatial.distance as ssd
import tempfile as tpf
from aqml.cheminfo.core import *
from aqml.cheminfo.molecule.elements import Elements

T,F = True,False

elements_mmff94 = [ 'H','C','N','O','F','Si','P','S','Cl', 'Br','I' ]
zs_mmff94 = [ chemical_symbols.index(el) for el in elements_mmff94 ]

elements_c_uff = ['B','Al','Ga','In','Tl', \
                      'Si','Ge','Sn', \
                      'P', 'As','Sb','Bi', \
                           'Se','Te', ]
zsc_uff = [ chemical_symbols.index(el) for el in elements_mmff94 ]

element_cans = copy.copy(elements_c_uff)

nves = {}
nves = dict(list(zip(['B','Al','Ga','In','Tl'],[3]*5)))
nves.update( dict(list(zip(['Si','Ge','Sn'], [4]*3))) )
nves.update( dict(list(zip(['P','As','Sb','Bi'], [3]*4))) )
nves.update( dict(list(zip(['Se','Te'], [2]*2))) )

def get_coords(m):
    assert type(m) is ob.OBMol
    na = m.NumAtoms()
    coords = []
    for i in range(1,na+1): # atom idx starts from 1 in openbabel
        ai = m.GetAtom(i)
        coords.append( [ ai.GetX(), ai.GetY(), ai.GetZ() ] )
    return np.array(coords)

def get_bom(m):
    na = m.NumAtoms()
    bom = np.zeros((na,na),dtype=int)
    for i in range(na):
        ai = m.GetAtomById(i)
        for bond in ob.OBAtomBondIter(ai):
            ab, ae = bond.GetBeginAtom(), bond.GetEndAtom()
            ib1 = ab.GetIdx()-1; ie1 = ae.GetIdx()-1
            #ib2 = ab.GetId(); ie2 = ae.GetId()
            #print i, ' -- ', ib1,ie1, ib2,ie2
            bo = bond.GetBO()
            bom[ib1,ie1] = bo; bom[ie1,ib1] = bo
    return bom

def to_oechem_can(smi):#,hack_uff=F):
    """ Note that some SMILES such as C=N=O cannot be recognized
    correctly by openbabel. With OEChem, it is interpretted as
    C=[NH]=O, as was desired """

    from openeye import oechem

    if smi in element_cans:
        return '['+smi+'H%d]'%( nves[smi] )
    m = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(m,smi)
    assert oechem.OEAddExplicitHydrogens(m)
    #atyps = []
    #for ai in m.GetAtoms():
    #    zi = ai.GetAtomicNum()
    #    vi = ai.GetValence()
    #    nhi = ai.GetTotalHCount()
    #    if zi in [7,15] and vi==5 and nhi>0:
    #        for aj in ai.GetAtomIter():
    #            zj = aj.GetAtomicNum()
    #            if zj==1: aj.SetIsotope(T)
    #    elif zi in [5] and vi==3 and nhi>0:
    #        for aj in ai.GetAtomIter():
    #            zj = aj.GetAtomicNum()
    #            if zj==1: aj.SetIsotope(T)
    #
    # "OESMILESFlag_ImpHCount" is indispensible!!
    # Otherwise, B=C won't be processed correctly by openbabel, i.e., somehow obabel
    # tries to add two H's (instead of one) to B. While things are ok with [BH]=C
    flavor = oechem.OESMILESFlag_Isotopes | oechem.OESMILESFlag_Canonical | oechem.OESMILESFlag_ImpHCount
    smi = oechem.OECreateSmiString(m, flavor)
    # OECreateIsoSmiString() # ISOMERIC .eq. Isotopes | AtomStereo | BondStereo | Canonical | AtomMaps | RGroups
    #fout = tpf.NamedTemporaryFile(dir='/tmp/').name + '.sdf'
    #ofs = oemolostream(fout)
    #ofs.SetFormat( OEFormat_SDF )
    #OEWriteMolecule(ofs, m)
    #_m = pb.readstring( 'sdf', open(fout,'r').read() )
    #m = _m.OBMol
    #os.system('rm %s'%fout)
    return smi # m


class Mol(object):

    def __init__(self, obj, hack_smi=F, hack_env=F, fmt=None, addh=T,  \
                 make3d=F, ichg=F, irad=F, ff='mmff94', isotope=F):
        """
        openbabel has problem intepreting smiles like O=N=C (in reality, it
        should be O=[NH]=C). We can hack this by calling `to_oechem_can() first.
        (if hack_smi=T)

        Another issue is the lack of uff parameters for envs like As=, As#,
        Sb=, et al. We can hack this by replacing these elements by the element
        of the same group in periodic table. (hack_env=T)
        """
        conv = ob.OBConversion()
        m = ob.OBMol()
        typ = type(obj)
        self.istat = T
        self.bom = None
        repl = []
        if typ is str:
            ismi = F
            if obj[-3:] in ['sdf','mol','pdb']:
                fmt = obj[-3:]
                #conversion from file to OBMol object
                conv.SetInFormat(fmt)
                conv.ReadFile(m, obj)
            else:
                ismi = T
                smi = obj
                if hack_smi:
                    smi = to_oechem_can(obj)
                if fmt is None: fmt = 'smi'
                conv.SetInFormat(fmt)
                conv.ReadString(m, smi)
                m.AddHydrogens()
        else:
            print('#ERROR: input not supported yet!')
            raise

        #print '#1'

        na = m.NumAtoms()
        self.na = na
        if self.bom is None:
            self.bom = get_bom(m)

        # set isotope to 0
        # otherwise, we'll encounter SMILES like 'C[2H]',
        # and error correspondently.
        # In deciding which atoms should be have spin multiplicity
        # assigned, hydrogen atoms which have an isotope specification
        # (D,T or even 1H) do not count. So SMILES N[2H] is NH2D (spin
        # multiplicity left at 0, so with a full content of implicit
        # hydrogens), whereas N[H] is NH (spin multiplicity=3). A
        # deuterated radical like NHD is represented by [NH][2H].
        if not isotope:
            for i in range(na):
                ai = m.GetAtomById(i); ai.SetIsotope(0)

        zs = []; chgs = []
        for i in range(1,na+1):
            ai = m.GetAtom(i)
            zi = ai.GetAtomicNum()
            zs.append( zi )
            chgs.append( ai.GetFormalCharge() )
        self.zs = np.array(zs, np.int)
        self.chgs = np.array(chgs,dtype=int)
        if not ismi:
            coords = get_coords(m)

        istat_rad = T
        if not irad:
            if sum(zs)%2 > 0:
                istat_rad = F

        istat_chg = T
        if not ichg:
            if sum(chgs) != 0:
                istat_chg = F

        if (not istat_rad) or (not istat_chg):
            self.istat = F
            print( '    [ERROR] Dected radical/charged mol' )
            return

        if ff in ['mmff94','mmff94s']:
            compl = np.setdiff1d(zs, zs_mmff94)
            if len(compl) > 0:
                print(( '    [ERROR] Detected symbols not supported by MMFF94: ', chemical_symbols[compl] ))
                self.istat = F
                return
        #print '#2'

        # get coarse geometry
        # Note: openbabel fails to process SMILES like "Br[pH]1n[pH2]n[p]n1", i.e.,
        #       after calling `make3d(), the program adds one more H atom to the
        #       last P atom. To circumvent this,
        repls = []
        nr = 0
        if ismi and make3d:
            # find atom whose type is not covered by UFF
            if hack_env:
                m2, repls = self.get_repl(m)
                nr = len(repls)
                if nr > 0: print(' *info: detected envs not supported by UFF/MMFF94')
            else:
                m2 = m
            m3 = pb.Molecule(m2)
            m3.make3D(forcefield=ff, steps=50)
            assert len(m3.atoms) == na, '#ERROR: make3d() added extra H atoms!'
            if hack_env and nr > 0:
                m = self.restore(m3.OBMol)
                print(' *info: now envs are restored')
            else:
                m = m3.OBMol

        M = pb.Molecule(m)
        coords = np.array([ ai.coords for ai in M.atoms ])
        #print '#3'
        # check coords
        if np.any(coords.astype(np.str) == 'nan'):
            print('    [ERROR] %s cannot process some envs, e.g., -As='%ff)
            self.istat = F
            return
        self.m = m
        self.M = M
        self.coords = coords
        self.repls = repls

    def get_repl(self,m):
        """ get envs to be replaced
        This has to be done in case the env is not supported in
        UFF/MMFF94. E.g., [As]# in [As]#C"""
        m2 = copy.copy(m)
        assert np.all(self.bom==self.bom.astype(np.int)), '#ERROR: BO != integer?'
        repls = []
        for ia in range(self.na):
            ai = m2.GetAtomById(ia)
            zi = self.zs[ia]
            _bosi = self.bom[ia]
            bosi = _bosi[_bosi>0]; bosi.sort()
            atyp = ''.join(['%d'%boi for boi in bosi])
            if zi in zsc_uff:
                if atyp != '1'*nves[ chemical_symbols[zi] ]:
                    zj = subs[zi]
                    ai.SetAtomicNum(zj)
                    repls.append( (ia,zi,zj) )
        return m2, repls

    def restore(self,m):
        m2 = copy.copy(m)
        return m2

    def is_overcrowded(self):
        ds = ssd.squareform( ssd.pdist(self.coords) )
        non_bonds = np.where( np.logical_and(self.bom==0, ds!=0.) )
        rcs = Elements().rcs[ self.zs ]
        dsmin = rcs[..., np.newaxis] + [rcs]
        return np.any(ds[non_bonds]<dsmin[non_bonds])

    def clone(self):
        m2 = pb.Molecule(self.m).clone
        return m2.OBMol

    def Atoms(self):
        """
        convert to ase object
        """
        import ase
        mu = pb.Molecule( self.m )
        m2 = ase.Atoms([])
        for ai in mu.atoms:
            m2.append(ase.Atom(ai.atomicnum, ai.coords))
        return m2

    def to_RDKit(self):
        """
        convert OBMol to RDKit_Mol
        """
        from rdkit import Chem

        c = ob.OBConversion()
        c.SetOutFormat('sdf')
        ctab = c.WriteString(self.m)
        mu = Chem.MolFromMolBlock( ctab, removeHs=False ) # plz keep H's
        return mu

    def write(self, fout):
        """ this will keep stereochem """
        c = ob.OBConversion()
        c.SetOutFormat(fout[-3:])
        iok = c.WriteFile(self.m, fout)

    def write_nostereo(self, fout):
        write_ctab(self.zs, self.chgs, self.bom, self.coords, sdf=fout)

    def optg_c2(self, smartss, iass, ff='MMFF94', step=500):
        """
        optg with constraints specified by `smartss

        vars
        ================
        smartss -- ['[#1]', 'C(=O)[OH]']
        iass    -- [[0,], [1,2,3] ]
        """
        # Define constraints
        c = ob.OBFFConstraints()

        iast = list(range(self.na))
        iasf = []
        for i, smarts_i in enumerate(smartss):
            #if 'H' in smarts_i: assert self.addh
            q = pb.Smarts( smarts_i )
            assert q.obsmarts.Match(self.m), '#ERROR: no match?'
            for match in q.obsmarts.GetMapList():
                idxs = [ self.m.GetAtom(ia).GetIdx() for ia in match ]
                for j in iass[i]:
                    iasf.append( idxs[j] )

        for ia in list( set(iast)^set(iasf) ):
            c.AddAtomConstraint(idxs[j])

        # Setup the force field with the constraints
        f = ob.OBForceField.FindForceField(ff)
        assert f.Setup(self.m, c), '#ERROR: ForceFiled setup failure [contains non-mmff94 elements]?'
        f.SetConstraints(c)
        #if optimizer in ['cg','CG','ConjugateGradients']:
        f.ConjugateGradients(step)
        #elif optimizer in ['sd','SD','SteepestDescent']:
        #    f.SteepestDescent(step)
        f.GetCoordinates(self.m)
        self.M = pb.Molecule(m)
        self.coords = get_coords(m)

    def optg_c(self, steps=60, ff="mmff94", optimizer='cg'):
        """
        Opt geometry by ff with constrained torsions
        ==============================================
        defaut ff: mmff94, mmff94s seems to be
                   problematic, sometimes it
                   leads to weird geometries
        steps: ff steps for constraint opt
        """
        m = self.m #copy.copy( self.m )
        # Define constraints
        c = ob.OBFFConstraints()
        #c.AddDistanceConstraint(1, 8, 10)        # Angstroms
        #c.AddAngleConstraint(1, 2, 3, 120.0)     # Degrees
        # constrain torsion only
        torsions = self.get_all_torsions()
        for torsion in torsions:
            i,j,k,l = torsion
            ang = torsions[torsion]
            c.AddTorsionConstraint(i,j,k,l, ang) # Degrees
        # Setup the force field with the constraints
        obff = ob.OBForceField.FindForceField(ff)
        obff.Setup(m, c)
        obff.SetConstraints(c)
        obff.ConjugateGradients(steps)
        obff.GetCoordinates(m)
        self.m = m
        self.M = pb.Molecule(m)
        self.coords = get_coords(m)

    def optg(self, cparam=[False,'slow'], ff="MMFF94", steps=2500):
        """ geoemtry optimization
        ================
        cparam: conformer parameters to be used
                cparam = [False] -- no conformer generation
                cparam = [True, algo, num_conf, N] -- gen confs and `N ff steps for each conf
        """
        # ff opt
        m = self.m
        obff = ob.OBForceField.FindForceField(ff)
        obff.Setup(m)
        #obff.SteepestDescent(1200, 1.0e-4) # a quick cleanup before searching
        obff.ConjugateGradients(600, 1.0e-3) # fast & coarse
        if cparam[0]:
            if cparam[1] in ['slow']:
                obff.FastRotorSearch(True)
            elif algo in ['slowest']:
                nc, n = cparam[1:]
                obff.WeightedRotorSearch(nc, n)
            else:
                print('unknow algo')
                raise
        obff.ConjugateGradients(steps, 1.0e-6) # fast & coarse
        #obff.SteepestDescent(500, 1.0e-6) # slower &
        obff.GetCoordinates(m)
        self.m = m
        self.M = pb.Molecule(m)
        self.coords = get_coords(m)

    def get_all_torsions(self):
        """
        enumerate Torsions in a molecule
        """

        m = self.m
        #m.DeleteHydrogens()

        torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        q = pb.Smarts( torsionSmarts )
        iok = q.obsmarts.Match(m)

        torsions = {}; tts = []
        for match in q.obsmarts.GetMapList():
            ia2,ia3 = match
            a2, a3 = m.GetAtom(ia2), m.GetAtom(ia3)
            hyb2, hyb3 = a2.GetHyb(), a3.GetHyb()

            if (hyb2 not in [2,3]) or (hyb3 not in [2,3]):
                # no formal torsion here !
                continue

            for a1 in ob.OBAtomAtomIter( a2 ):
                ia1 = a1.GetIdx()
                if ia1 == ia3:
                    continue

                for a4 in ob.OBAtomAtomIter( a3 ):
                    ia4 = a4.GetIdx()
                    if ia4 in [ia2, ia1]: # when ia4 == ia1, it means 3-membered ring
                        continue
                    else:
                        torsion_1 = (ia1,ia2,ia3,ia4)
                        torsion_2 = (ia4,ia3,ia2,ia1)
                        if torsion_1 not in tts:
                            torsions[ torsion_1 ] = m.GetTorsion(ia1,ia2,ia3,ia4)
                            tts.append(torsion_1); tts.append(torsion_2)
        return torsions

    def get_bom(self):
        """
        get connectivity table
        """
        bom = np.zeros((self.na,self.na), np.int)
        for i in range(1,self.na+1):
            ai = self.m.GetAtom(i)
            for bond in ob.OBAtomBondIter(ai):
                ia1 = bond.GetBeginAtomIdx()-1; ia2 = bond.GetEndAtomIdx()-1
                bo = bond.GetBO()
                bom[ia1,ia2] = bo; bom[ia2,ia1] = bo
        return bom



if __name__ == '__main__':

    import os, sys

    # Note that file in `fs represent
    fs = sys.argv[1:]
    for f in fs:
        #print ' -- f = ', f
        assert f[-3:] == 'sdf'
        si = f.split('/'); nsi = len(si)
        if nsi == 2:
            f2 = '%s/raw/%s_raw.sdf'%(si[0], si[-1][:-4])
        elif nsi == 1:
            f2 = 'raw/%s_raw.sdf'%( si[-1][:-4] )
        else:
            print('#ERROR:')
            raise

        iok = True
        try:
            obj = Mol( f )
            g = perceive_g(obj.zs,obj.coords) # graph from geometry
            bom = obj.get_bom() # bond orders in SDF file
            g0 = (bom > 0).astype(np.int)
            if not np.all(g == g0):
                iok = False
        except:
            print(' -- Parsing failed for %s'%f)
            iok = False # fails to retrieve coordinates!

        if not iok:
            obj2 = Mol( f2 )
            obj2.optg_c(ff="MMFF94", steps=120)
            obj2.write( f )


