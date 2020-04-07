
import io2, os, sys
import numpy as np
import aqml.cheminfo.rdkit.core as crk
import rdkit.Chem.rdmolfiles as rcr
from rdkit import Chem


class db(object):

    def __init__(self, obj, addh=True):
        """
        process a list of SMILES objects  
        """
        smis = []
        typ = type(obj)
        if typ is str:
            if os.path.exists(obj):
                smis += [ si.strip() for si in file(obj).readlines() ]
            else:
                smis += [ obj, ]
        elif typ is list:
            for obj_i in obj:
                if os.path.exists(obj_i):
                    smis += [ si.strip() for si in file(obj).readlines() ]
                else:
                    smis += [obj_i, ]
        self.nm = len(smis)
        self.objs = [ crk.RDMol(si) for si in smis ]
        #self.imst = np.arange(self.nm)

    def get_atom_types(self,icn=True):
        ats = []
        uats = []
        for obj in self.objs:
            atsi = []
            cns = (obj.bom>0).sum(axis=0)
            for i,zi in enumerate(obj.zs):
                ati = (zi,cns[i]) if icn else (zi)
                atsi.append(ati)
                if ati not in uats: uats.append(ati)
            ats.append( atsi )
        n = len(uats)
        nats = []
        uats.sort()
        for atsi in ats:
            cnts = np.zeros(n)
            for i,uat in enumerate(uats):
                for ati in atsi:
                    if ati == uat: cnts[i] += 1
            nats.append(cnts)
        self.nats = np.array(nats,dtype=int)
        self.uats = np.array(uats,dtype=int)

    def get_bond_types(self,icn=True):
        """ bond statistics 
        vars
        ============
        icn: add atomic type to bond as well
        """
        bts = []
        ubts = []
        for obj in self.objs:
            bom = obj.bom
            zs = obj.zs
            cns = (bom>0).sum(axis=0)
            btsi = []
            bsi = np.array(np.where(np.triu(bom)>0),dtype=int)
            for i in range(obj.nb):
                ia,ja = bsi[:,i]
                bo = bom[ia,ja]
                z1, z2 = obj.zs[ia], obj.zs[ja]
                cn1, cn2 = cns[ia], cns[ja]
                if (z1>z2) or (z1==z2 and cn1>cn2): 
                    tz = z1; z1 = z2; z2 = tz
                    tn = cn1; cn1 = cn2; cn2 = tn
                bti = (z1,cn1,z2,cn2,bo) if icn else (z1,z2,bo)
                btsi.append(bti)
                if bti not in ubts: ubts.append(bti)
            bts.append(btsi)
        ubts.sort()
        nbts = []
        nubt = len(ubts)
        for btsi in bts:
            cnts = np.zeros(nubt)
            for iu,ubt in enumerate(ubts):
                for bti in btsi:
                    if ubt == bti: cnts[iu] += 1
            nbts.append( cnts )
        self.ubts = np.array(ubts)
        self.nbts = np.array(nbts,dtype=int)

    def save(self,f):
        np.savez(f, uats=self.uats, nats=self.nats, ubts=self.ubts, nbts=self.nbts)

    def get_statistics(self, igrp=False, prefix=None, iNM=0, inas=True):
        """
        get `nas for the whole db and group the SMILES
        into different subgroups according to `na if `igrp
        is True. Afterwards, write them into files.

        var's
        ================================================
        igrp -- if all smiles strings are to be grouped
        iNM  -- the number of molecules to be chosen
                randomly as a subset of GDB_n. If it's
                assigned a negative value, no subset of
                molecules will be writen to a *.smi file
        inas -- if True, then once we get `nas, stop
        """

        np.random.seed(2) # fix the random sequence

        nas = []
        zs = []; zsu = []

        #suppl = rcr.SmilesMolSupplier()
        #suppl.SetData( '\n'.join(self.smis), delimiter='\n', smilesColumn=0, titleLine=False)
        for smi in self.smis: # suppl:
            mi = Chem.MolFromSmiles(smi)
            miu = Chem.RemoveHs(mi)
            nas.append( miu.GetNumAtoms() )
            zsi = [ ai.GetAtomicNum() for ai in miu.GetAtoms() ]
            zs.append( zsi )
            zsu.append( list(set(zsi)) )

        self.zs = zs
        self.zsu = zsu
        self.nas = np.array(nas, np.int)
        if inas: return
        nasu = np.unique(self.nas)
        n = len(nasu)
        for i in range(n):
            nheav_i = nasu[i]
            filt_i = ( nheav_i == self.nas )
            ims_i = self.imst[ filt_i ]
            nm_i = ims_i.shape[0]
            print ' %4d %16d'%(nheav_i, nm_i)
            if igrp:
                smis_i = self.smis[ filt_i ]
                assert prefix != None
                with open('%s%02d.smi'%(prefix,nheav_i),'w') as fo:
                    fo.write( ''.join( smis_i ) )

                if nheav_i >= 9:
                    if iNM > 0:
                        #assert nm_i > iNM*1000
                        if nm_i > iNM:
                            ims_i_u = np.random.permutation(nm_i)
                            with open('GDB_%02d_%d.smi'%(nheav_i,iNM),'w') as fo:
                                fo.write( ''.join( smis_i[ ims_i_u[:iNM] ] ) )

        #self.nms = np.array(nms).astype(np.int)
        #self.zsu = np.array( list(zsu) )

