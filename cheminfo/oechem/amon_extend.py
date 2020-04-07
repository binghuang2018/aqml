
#from rdkit import Chem
from aqml.cheminfo.rw.ctab import write_ctab
import aqml.cheminfo.oechem.amon as coa
from aqml.cheminfo.oechem.oechem import *
import tempfile as tpf
import os, sys

T, F = True, False

class transform(StringM):
    def __init__(self, string, iprt=F):
        StringM.__init__(self, string, iPL=T, suppressH=T)
        assert np.all(self.zs > 1), '#ERROR: hydrogens were not suppressed??'
        ias = np.arange(self.na)
        self.ias = ias 
        self.string = string
        self.iprt = iprt

    #def clone(self):
    #    return self.clone()
    
    def get_newolds_i(self, li3):
        """ get new and old bonds to be operated on later"""
        newolds_i = [] # creat bonds (or bonds to be contracted)

        ## step 1: is this angle strained?
        ## To-do
        
        ## step 2: 
        i,c,j= li3
        iasb,jasb = [self.ias[ self.bom[idx]>0 ] for idx in [i,j] ]
        bosi_r, bosj_r = self.bom[i,iasb], self.bom[j,jasb]
        for ia in iasb:
            for ja in jasb:
                bo_i, bo_j = self.bom[ia,i], self.bom[ja,j]
                if bo_i == bo_j and bo_i == 1: 
                    # note that when bo_i == 2, some unsaturated 
                    # atoms would become saturated and thus it's
                    # strictly forbidden.
                    newolds_i.append( [bo_i, [i,j], [i,ia], [j,ja]] ) # entry 2: bond_to_make, entries 3-4: bonds_to_break
                    ## yet there is another possibility
                    ##newolds_i.append( [bo_i, [i,j], ])
        return newolds_i

    def get_newolds(self):
        """ get __all__ new and old bonds to be operated on later"""
        newolds = []
        keeps = []
        bom = self.bom 
        for i in range(self.na):
            for j in range(i+1,self.na):
                ti,tj = bom[i], bom[j]
                bosi, bosj = ti[ti>0], tj[tj>0]
                if self.PLs[i,j]-2==0: 
                    comm_neibs = self.ias[ np.logical_and(bom[i]>0, bom[j]>0) ]
                    for comm in comm_neibs:
                        li3 = [i,comm,j]
                        newolds_i = self.get_newolds_i(li3)
                        if len(newolds_i) > 0:
                            newolds += newolds_i 

                            #size_r = self.get_ring_size([i,ia],[j,ja])

                            keeps += [ li3 ]*len(newolds_i)
        self.newolds = newolds
        self.keeps = keeps  # used to check overlap between bonds_break and bonds_make

    def get_newm(self, newolds):
        bom = self.bom.copy()
        #print('')
        #print('newolds=',newolds)
        for newold in newolds:
            bo, newb = newold[0], newold[1]
            i,j = newb; bom[i,j] = bom[j,i] = bo
            bbs = newold[2:]
            #print('bbs=',bbs)
            for bb in bbs:
                k, l = bb
                bom[k,l] = bom[l,k] = 0
        ctab = write_ctab(self.zs, self.chgs, bom, self.coords)

        ## Don't use rdkit as it will fail to process strings like "N(=O)=O"
        #newm = Chem.MolFromMolBlock(ctab) # no H's by default 

        ## Don't try to call newmol() from oechem.py as this function
        ## is designed exclusively for the complete mol (i.e., with H's
        ## specified explicitely)
        #newm = newmol(self.zs, self.chgs, bom, self.coords)

        # the code below works robustly
        tf = tpf.NamedTemporaryFile(dir='/tmp').name + '.sdf'
        with open(tf,'w') as fid: fid.write(ctab)
        newm = sdf2oem(tf)
        os.remove(tf)
        return newm
    
    def iovlp_bonds(self, newold1, newold2, kps1, kps2):
        """check if there exists any bond in """
        i,c,j = kps1; bbs1_keep = [ set([i,c]), set([j,c]) ]
        i,c,j = kps2; bbs2_keep = [ set([i,c]), set([j,c]) ]
        bbs1_rm = [ set(bbi) for bbi in newold1[2:] ]
        bbs2_rm = [ set(bbi) for bbi in newold2[2:] ]
        iovlp = F
        for bk1 in bbs1_keep:
            if bk1 in bbs2_rm:
                iovlp = T
                break 
        for bk2 in bbs2_keep:
            if bk2 in bbs1_rm:
                iovlp = T
                break
        return iovlp
        
    def T(self, level=2):
        """
        now transform the given graph by breaking/making bonds
        """
        newms=[]
        smiles=[]
        qs = []
        # one bond contraction
        newolds = self.newolds
        keeps = self.keeps 
        #print('len of newolds, keeps = ', len(newolds),len(keeps))
        #print('newolds=',newolds)
        for c in newolds:
            m = self.get_newm([c])
            smi  = oem2smi(m) #Chem.MolToSmiles(m)
            if smi not in smiles:
                newms.append(m)
                smiles.append(smi)
                qs += smi.split('.')
        
        newms2 = []
        if level == 2:
            n = len(self.newolds)
            for i in range(n):
                for j in range(i+1,n):
                    cb1, cb2 = newolds[i], newolds[j]
                    kps1, kps2 = keeps[i], keeps[j]
                    comm = np.intersect1d(cb1[1], cb2[1])
                    iovlp = self.iovlp_bonds(cb1,cb2,kps1,kps2)
                    if comm.size == 0 and (not iovlp):
                        m = self.get_newm([cb1,cb2])
                        smi = oem2smi(m) #Chem.MolToSmiles(m)
                        if smi not in smiles:
                            newms2.append(m)
                            smiles.append(smi)
                            qs += smi.split('.')

        self.qs = list( set(qs) )
        return newms, newms2

    def get_amons(self):
        reduce_namons=T; label='temp'; imap=F; wg=F; k=7; k2=7
        a = coa.ParentMols([self.string], reduce_namons, label=label, \
                            iprt=self.iprt,imap=imap, fixGeom=F, wg=wg,\
                            k=k,k2=k2) #, forcefield=ff, thresh=0.01)
                           
        self.amons = a.cans 

    def get_amons_extended(self, k):
        # now generate amons for new queries obtained by contraction of rings
        reduce_namons=T; label='temp'; imap=F; wg=F
        a = coa.ParentMols(self.qs, reduce_namons, label=label, imap=imap, 
                           iprt=self.iprt, fixGeom=F, wg=wg, k=k,k2=k)
                           #, forcefield=ff, thresh=0.01)
        self.get_amons()
        #print('new amons:', a.cans)
        self.amons_extended = list( set(a.cans).difference( set(self.amons) ))



import multiprocessing

class amons_ext(object):
    """
    Get extended amons
    Parallel processing is possilbe!
    """
    def __init__(self, ss1, k2, nproc=1):
        nm = len(ss1)
        objs = []
        if nproc == 1:
            for i,s in enumerate(ss1):
                print('%d/%d: %s'%(i+1,nm,s))
                ipt = [s, k2, F]
                objs.append( self.get_amons(ipt) )
        else:
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ [s,k2,F] for s in ss1 ]
            objs = pool.map(self.get_amons, ipts)
        # now collect all amons_ext
        amons_ext = set()
        for amons_i in objs:
            amons_ext.update( amons_i )
        self.amons_ext = list(amons_ext)

    @staticmethod
    def get_amons(ipt):
        s, k2, iprt = ipt
        obj = transform(s, iprt=iprt)
        obj.get_newolds()
        newms, newms2 = obj.T(level=2)
        obj.get_amons_extended(k2)
        return obj.amons_extended


if __name__ == "__main__":
    
    args = sys.argv[1:]
    k2 = int(args[0])
    ss1 = args[1:]
    amons_ext = get_amons_ext(ss1,k2)
    print( 'extended_amons = ', list(amons_ext) )

