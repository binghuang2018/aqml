
import os,sys
import numpy as np
import indigo
from cheminfo import *

__all__ = [ '_indigo' ]

T,F = True,False

class _indigo(object):

    def __init__(self, _obj, addh=True, kekulize=False):
        hadd_status = T
        parse_status = T
        kekulize_status = T
        if isinstance(_obj, indigo.IndigoObject):
            m = _obj
        elif isinstance(_obj, str):
            obj = indigo.Indigo()
            if os.path.exists(_obj):
                m = obj.loadMoleculeFromFile(_obj)
            else:
                try:
                    m = obj.loadMolecule(_obj)
                except:
                    print ' ** failed to parse SMILES'
                    parse_status = F
            if parse_status:
                if addh:
                    try:
                        m.unfoldHydrogens()
                    except:
                        hadd_status = F
                        print ' ** failed to add H'
        elif isinstance(_obj, (tuple,list)):
            # unfortunately, Indigo cannot load file formats other than sdf/mol
            # so we manually construct molecule here
            assert len(_obj) == 4
            zs, coords, chgs, bom = _obj
            na = len(zs)
            ias = np.arange(na).astype(np.int)
            newobj = indigo.Indigo()
            m = newobj.createMolecule()
            ats = []
            for ia in range(na):
                ai = m.addAtom(chemical_symbols[zs[ia]])
                ai.setXYZ( coords[ia,0],coords[ia,1],coords[ia,2] )
                ai.setCharge(chgs[ia])
                ats.append(ai)
            for ia in range(na):
                ai = ats[ia]
                jas = ias[ np.logical_and(bom[ia]>0, ias>ia) ]
                for ja in jas:
                    bi = ai.addBond(ats[ja], bom[ia,ja])
        else:
            raise '#unknown instance'

        if parse_status and hadd_status:
            if kekulize:
                kekulize_status = m.dearomatize()
                if not kekulize_status: print ' ** failed to kekulize'

        self.na = m.countAtoms()
        self.zs = np.array([m.getAtom(i).atomicNumber() for i in range(self.na)],np.int)
        self.ias = np.arange(self.na)
        self.ias_heav = self.ias[self.zs>1]
        self.nheav = len(self.ias_heav)
        self.m = m

        self.parse_status = parse_status
        self.hadd_status = hadd_status
        self.kekulize_status = kekulize_status


    def get_coords(self):
        coords = []; zs = []
        for i in xrange(self.na):
            ai = self.m.getAtom(i)
            zs.append( ai.atomicNumber() )
            coords_i = ai.xyz()
            coords.append( coords_i )
        return np.array(coords)

    def kekulize(self):
        iar = self.m.dearomatize()
        self.iar = iar

    def get_bom(self):
        #m2 = self.copy()
        #iar = m2.dearomatize()
        bom = np.zeros((self.na, self.na)).astype(np.int)
        nb = self.m.countBonds()
        for ib in range(nb):
            bi = self.m.getBond(ib)
            i, j = bi.source().index(), bi.destination().index()
            bo = bi.bondOrder()
            bom[i,j] = bom[j,i] = bo
        return bom

    def get_states(self):
        """ valency states, charges and positions of atoms """
        self.coords = self.get_coords()
        self.bom = self.get_bom()
        self.cns = (self.bom>0).sum(axis=0)
        self.cns_heav = (self.bom[self.ias_heav][:,self.ias_heav]>0).sum(axis=0)
        self.nhs = self.cns[self.ias_heav]-self.cns_heav
        self.chgs = self.get_charges()
        self.vs = self.get_valences()

    def update_states(self,coords,bom,chgs,vs):
        self.coords = coords
        self.bom = bom
        self.cns = (self.bom>0).sum(axis=0)
        self.cns_heav = (self.bom[self.ias_heav][:,self.ias_heav]>0).sum(axis=0)
        self.nhs = self.cns[self.ias_heav]-self.cns_heav
        self.chgs = chgs
        self.vs = vs

    def check_states(self,bom,chgs,tvsr,cnsr):
        tvs = bom.sum(axis=0) - chgs
        cns = (bom>0).sum(axis=0)
        return np.all([ (tvs[i] in tvsr[zi]) and (cns[i]<=cnsr[zi]) for i,zi in enumerate(self.zs) ])

    def check_detailed_states(self,bom,chgs,tvsr,cnsr):
        tvs = bom.sum(axis=0) - chgs
        cns = (bom>0).sum(axis=0)
        for i,zi in enumerate(self.zs):
            if not ( (tvs[i] in tvsr[zi]) and (cns[i]<=cnsr[zi]) ):
                print 'i,zi,vi,vr, cn,cnr = ', i,zi,tvs[i],tvsr[zi],cns[i],cnsr[zi]

    def copy(self):
        return self.m.clone()

    def get_charges(self):
        return np.array([self.m.getAtom(i).charge() for i in range(self.na)],np.int)

    def get_raw_valences(self):
        """ original valences associated to SMILES """
        return np.array([self.m.getAtom(i).valence() for i in range(self.na)],np.int)

    def get_valences(self):
        """
        indigo gives rather different valences compared to rdkit (desired)

        [NH+]#[C-]     4,4 -> 4,3
        N=[NH+][NH-] 3,4,3 -> 3,4,2
        O=[NH]=O     2,4,2 -> 2,5,2
        """
        vs = []
        for i in range(self.na):
            vi = self.m.getAtom(i).valence()
            ci = self.chgs[i]
            if ci == 0:
                if self.zs[i]==7 and self.bom[i].sum()==5:
                    vi = 5 # Indigo somehow give a valence of 4 for N_2 in N=[NH]=N
            elif ci < 0:
                vi += ci
            vs.append(vi)
        return np.array(vs,np.int)

    def remove_isotope(self):
        for i in range(self.na):
            self.m.getAtom(i).setIsotope(0)

    def has_isotope(self):
        return np.any([self.m.getAtom(i).isotope() for i in range(self.na)])

    def has_standalone_charge(self):
        iok = False
        cas = self.ias[self.chgs!=0]
        for ca in cas:
            if (self.chgs[self.bom[ca]>0]!=0).sum()==0:
                iok = True
                break
        return iok

    def is_radical(self):
        return np.any([self.m.getAtom(i).radical() for i in range(self.na)])

    def tocan(self, nostereo=True, aromatize=True):
        """
        one may often encounter cases that different SMILES
        correspond to the same graph. By setting aromatize=True
        aliviates such problem
        E.g., C1(C)=C(C)C=CC=C1 .eqv. C1=C(C)C(C)=CC=C1
        """
        m2 = self.copy()
        if nostereo: m2.clearStereocenters()
        if aromatize: m2.aromatize()
        can = m2.canonicalSmiles()
        return can

