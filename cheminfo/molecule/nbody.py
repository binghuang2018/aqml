
import os, sys
import numpy as np
import aqml.cheminfo.molecule.geometry as cmg

T,F = True,False

def get_mbtypes(zs):
    """
    get many-body types
    """
    # atoms that cannot be J in angle IJK or J/K in dihedral angle IJKL
    zs1 = [1,9,17,35,53]

    zs.sort()
    nz = len(zs)

    # 1-body
    mbs1 = [ '%d'%zi for zi in zs ]

    # 2-body
    mbs2 = []
    mbs2 += [ '%d-%d'%(zi,zi) for zi in zs ]
    for i in range(nz):
        for j in range(i+1,nz):
            mbs2.append( '%d-%d'%(zs[i],zs[j]) )

    # 3-body
    mbs3 = []
    zs2 = list( set(zs).difference( set(zs1) ) )
    zs2.sort()
    nz2 = len(zs2)
    for j in range(nz2):
        for i in range(nz):
            for k in range(i,nz):
                type3 = '%d-%d-%d'%(zs[i],zs2[j],zs[k])
                if type3 not in mbs3: mbs3.append( type3 )

    # 4-body
    mbs4 = []
    for j in range(nz2):
        for k in range(j,nz2):
            for i in range(nz):
                for l in range(nz):
                    zj,zk = zs2[j],zs2[k]
                    zi,zl = zs[i],zs[l]
                    if j == k:
                        zi,zl = min(zs[i],zs[l]), max(zs[i],zs[l])
                    type4 = '%d-%d-%d-%d'%(zi,zj,zk,zl)
                    if type4 not in mbs4: mbs4.append( type4 )
    return [mbs2,mbs3,mbs4]


def copy_class(objfrom, objto, names):
    for n in names:
        if hasattr(objfrom, n):
            v = getattr(objfrom, n)
            setattr(objto, n, v);

def set_cls_attr(obj, names, vals):
    for i,n in enumerate(names):
        setattr(obj, n, vals[i])

class NBody(object):
    """
    get many body terms
    """

    # reference coordination numbers
    cnsr = { 1:1, 3:1, 4:2, 5:3, 6:4, 7:3, 8:2, 9:1, \
             11:1, 12:2, 13:3, 14:4, 15:3, 16:2, 17:1, \
             35:1}

    def __init__(self, obj, g=None, pls=None, rpad=F, rcut=12.0, unit='rad', \
                 iconn=F, icn=F, iconj=F, icnb=F, plmax4conj=3, bob=F, plcut=None, \
                 iheav=F, ivdw=F, cns=None, ctpidx=F, idic4=F):
        """
        iconj : distinguish between sigma and pi bond type in a bond with BO>1
        icnb  : allow conjugated & non-bonded atomic pair to be treated by a Morse pot
        #i2b   : allow a bond with BO>1 to be treated as a sigma and a pi bond

        ctpidx: calculate toplogical idx? T/F
        plcut : Path Length cutoff
        rpad  : if set to T, all bond distances associated with a torsion are
                also part of the list to be returned
        """
        self.plmax4conj = plmax4conj
        self.ctpidx = ctpidx
        self.plcut = plcut
        self.bob = bob
        self.rpad = rpad
        self.idic4 = idic4
        if isinstance(obj,(tuple,list)):
            assert len(obj) == 2
            zs, coords = obj
        else:
            #sa = obj.__str__() # aqml.cheminfo.core.molecules object
            #if ('atoms' in sa) or ('molecule' in sa):
            try:
                zs, coords = obj.zs, obj.coords
            except: #else:
                raise Exception('#ERROR: no attributes zs/coords exist for `obj')
        if iconj:
            iconn, icn = T, T
        if icn:
            iconn = T
        na = len(zs)
        set_cls_attr(self, ['na','zs','coords','g','pls'], \
                      [na, zs, coords, g, pls] )
        if iconn:
            assert g is not None
        if icnb:
            assert pls is not None
        set_cls_attr(self, ['rcut','unit','iheav','iconn','icn','iconj', 'icnb',], \
                            [rcut, unit, iheav, iconn, icn, iconj, icnb])
        ias = np.arange(self.na)
        self.ias = ias
        ias_heav = ias[ self.zs > 1 ]
        self.ias_heav = ias_heav
        g_heav = g[ias_heav][:,ias_heav]
        self.nb_heav = int( (g_heav > 0).sum()/2 )
        iasr1, iasr2 = np.where( np.triu(g_heav) > 0 )
        self.iasb = np.array([ias_heav[iasr1],ias_heav[iasr2]],np.int).T
        if cns is None:
            # in case that we're given as input a subgraph made up of heavy atoms only,
            # then `cns info is incomplete (due to neglect of H's), you must manually
            # specify `cns to rectify this.
            cns = g.sum(axis=0)
        self.cns = cns
        self.geom = cmg.Geometry(coords)
        self.vars2, self.vars3, self.vars4 = [], [], []
        # is atom unsaturated?
        self.ius = ( cns < np.array([self.cnsr[zi] for zi in zs]) )

    def iza8(self, zs):
        return np.all(np.array(zs)==8)

    def is_conjugated(self,ia,ja, hyperconj=T):
        """ are the i- and j-th atoms conjugated?
        criteria:
            cn_i < cn_ref, e.g., C_sp2, cn=3, cnr=4
        if hyperconj is `True, and one atom satisfying cni<cnr while
        the other being O/N-sp3, the corresponding bond is also considered
        to be conjugated
        """
        istat = F
        ius = self.ius[ [ia,ja] ]
        #print('    -- saturated? i,j = ', ius)
        zsp = self.zs[ [ia,ja] ]
        if np.all(ius): #iu1 and iu2:
            #if not self.iza8([z1,z2]):
            #    # exclude interaction between "=O" and "=O"
            istat = T
        else:
            if hyperconj and np.any(ius):
                z1 = zsp[ius][0]
                z2 = zsp[1] if z1 == zsp[0] else zsp[0]
                if z2 in [7,8]: # and (not self.iza8([z1,z2])):
                    istat = T
        return istat

    def get_atoms(self):
        mbs1 = {}
        for ia in range(self.na):
            cni = self.cns[ia]
            zi = self.zs[ia]
            type1 = '%d_%d'%(zi,cni) if self.icn else '%d'%zi
            if type1 in list(mbs1.keys()):
                mbs1[type1] += [zi]
            else:
                mbs1[type1] = [zi]
        return mbs1

    @property
    def cg(self):
        if not hasattr(self, '_cg'):
            self._cg = self.get_cg()
        return self._cg

    def get_cg(self, hyperconj=T):
        """
        get conjugation graph, i.e., cg[i,j] = T if the i- and j-th atom
        1) form a bond and 2) are in the same conjugation env
        """
        cg = np.zeros((self.na,self.na)) # conjugation graph
        for ia in range(self.na):
            for ja in range(ia+1,self.na):
                if self.g[ia,ja]:
                    cg[ia,ja] = cg[ja,ia] = self.is_conjugated(ia,ja, hyperconj=F)
        return cg


    @property
    def tpidx(self):
        """ toplogical idx calculated based on the molecular graph """
        if not hasattr(self, '_tpidx'):
            tpidx = np.zeros((self.na,self.na)) # topological index
            cg = self.get_cg(hyperconj=F)  ###### `hyperconj reset to False!!
            dgrs = np.sum(cg, axis=0)
            for ia in range(self.na):
                for ja in range(ia+1,self.na):
                    #tpidx[ia,ja] = 1./np.sqrt(dgrs[ia]*dgrs[ja]) if cg[ia,ja]>0 else 0
                    tpidx[ia,ja] = (dgrs[ia]-1)*(dgrs[ja]-1) if cg[ia,ja]>0 else 0
            self._tpidx = tpidx
        return self._tpidx


    def get_bonds(self, ias=[]):
        """
        the atomic pair contrib

        vars
        ====================
        iconn: should the pair of atoms be connected?
        """
        bs = []
        nai = len(ias)
        if nai == 0:
            ias = self.ias # allows for user-specified central atoms
            for ia in ias:
                for ja in range(ia+1, self.na):
                    bs.append([ia,ja])
        elif nai == 1:
            for ja in np.setdiff1d(self.ias, ias):
                bs.append( [ias[0],ja] )
        #vars2 = []
        mbs2 = {}
        ds = self.geom.ds
        if self.ctpidx: self.tpsidx = {}
        for ia,ja in bs: #ias: #range(self.na):
                zi, zj = self.zs[ [ia,ja] ]
                if self.iheav:
                    if (zi==1 or zj==1):
                        continue
                cni,cnj = self.cns[ [ia,ja] ]
                dij = ds[ia,ja]
                iok = True
                if dij > self.rcut:
                    continue
                icb = F # is conjugate bond?
                icnb_ij = F # allow conjugated & non-bonded atomic pair
                ib = self.g[ia,ja] # ( self.g[ia,ja] > 0 )
                if ib:
                    #if not self.iconn:
                    #    continue
                    if self.iconj:
                        #print('#################')
                        icb = cg[ia,ja]
                else:
                    if self.icnb:
                        pl = self.pls[ia,ja]
                        if pl == 0:
                            raise Exception('Todo:')
                        else:
                            #raise Exception('Todo: determine if two non-bonded atoms are conjugated, i.e., all atoms in the path connecting i,j should be non-saturated')
                            if pl > 0 and pl <= self.plmax4conj:
                                if self.is_conjugated(ia,ja):
                                    icnb_ij = T
                                else:
                                    continue
                            else:
                                continue
                    else:
                        if self.bob: # only for generating BoB repr
                            assert self.plcut is not None
                            if self.pls[ia,ja] > self.plcut:
                                continue
                if (zi>zj) or (self.icn and zi==zj and cni>cnj):
                    ia1 = ja; ja1 = ia
                else:
                    ia1 = ia; ja1 = ja
                zi1,zj1 = self.zs[ [ia1,ja1] ]
                cni,cnj = self.cns[ [ia1,ja1] ]
                type2 = '%d_%d-%d_%d'%(zi1,cni,zj1,cnj) if self.icn else '%d-%d'%(zi1,zj1)
                #print('i,j,icnb=', ia,ja,icnb_ij)
                if ib:
                    #types2 = [type2 + '-sigma', type2+'-pi'] if icb else [type2]
                    _type2 = type2
                    #print('ia,ja, types2=', ia,ja,types2)
                    #for _type2 in types2:
                    if _type2 in list(mbs2.keys()):
                        mbs2[_type2] += [dij]
                        if self.ctpidx:
                            self.tpsidx[_type2] += [self.tpidx[ia,ja]]
                    else:
                        mbs2[_type2] = [dij]
                        if self.ctpidx:
                            self.tpsidx[_type2] = [self.tpidx[ia,ja]]
                    #vars2.append( dij )
                else:
                    if icnb_ij:
                        type2 += '-pi' # '-pi-nb' ### 'nb': non-bond
                        if type2 in list(mbs2.keys()):
                            mbs2[type2] += [dij]
                        else:
                            mbs2[type2] = [dij]
                        #vars2.append( dij )
                    else:
                        if self.bob:
                            if type2 in list(mbs2.keys()):
                                mbs2[type2] += [dij]
                            else:
                                mbs2[type2] = [dij]
        #vars2 = np.array(vars2)
        return mbs2

    def get_neighbors(self,ia):
        return self.ias[ self.g[ia] > 0 ]

    def get_angles(self, jas=[]):
        """
        3-body parts: angles spanned by 3 adjacent atoms,
                      must be a valid angle in forcefield
        """
        ds = self.geom.ds
        #print('ds=',ds)
        mbs3 = {}
        if len(jas) == 0:
            jas = self.ias_heav # allows for user-specified central atoms
        for j in jas:
            zj = self.zs[j]
            neibs = self.get_neighbors(j)
            nneib = len(neibs)
            if nneib > 1:
                for i0 in range(nneib):
                    for k0 in range(i0+1,nneib):
                        i, k = neibs[i0], neibs[k0]
                        ias = [i,j,k]
                        zi,zj,zk = self.zs[ias]
                        cni,cnj,cnk = self.cns[ias]
                        if self.iheav and np.any(self.zs[ias]==1): continue
                        if (zi>zk) or (self.icn and zi==zk and cni>cnk): ias = [k,j,i]
                        zsi = [ self.zs[ia] for ia in ias ]
                        d1, d2 = ds[ias[0],ias[1]], ds[ias[1],ias[2]]
                        if self.icn:
                            tt = [ '%d_%d'%(self.zs[_],self.cns[_]) for _ in ias ]
                        else:
                            tt = [ '%d'%self.zs[_] for _ in ias ]
                        type3 = '-'.join(tt)
                        theta = self.geom.get_angle(ias, self.unit) # in degree
                        val = [d1,d2,theta] if self.rpad else theta
                        #print('a=',val)
                        if type3 in list(mbs3.keys()):
                            mbs3[type3] += [val]
                        else:
                            mbs3[type3] = [val]
                        #vars3.append( val )
        return mbs3

    def is_rotatable(self, zs, cns):
        """ check if a bond is rotatable """
        na = len(zs)
        assert na==2
        iok = T
        for ia in range(na):
            z = zs[ia]; cn = cns[ia]
            if cn == 1:
                iok = F
                break
            else:
                if z==6 and cn==2: # =C=C= is not rotatable!
                    iok = F
                    break
        return iok

    def get_dihedral_angles(self):
        """
        4-body parts: dihedral angles
        """
        mbs4 = {}
        _buffer = []
        for ib in range(self.nb_heav):
            j,k = self.iasb[ib]
            zsjk = self.zs[ [j,k] ]; zj,zk = zsjk
            cnsjk = self.cns[ [j,k] ]; cnj,cnk = cnsjk
            #print('  zj,cnj=', zj,cnj, ' zk,cnk=', zk,cnk)
            if not self.is_rotatable(zsjk,cnsjk):
                continue
            _buffer.append(ib)
            if (zj>zk) or (self.icn and zj==zk and cnj>cnk):
                t=k; k=j; j=t
            neibs1 = self.get_neighbors(j); n1 = len(neibs1);
            neibs2 = self.get_neighbors(k); n2 = len(neibs2);
            visited = []
            for i0 in range(n1):
                for l0 in range(n2):
                    i = neibs1[i0]; l = neibs2[l0]
                    pil = set([i,l])  # pair_i_l
                    if pil not in visited:
                        visited.append(pil)
                    else:
                        continue
                    ias = [i,j,k,l]
                    zi,zj,zk,zl = self.zs[ias]
                    cni,cnj,cnk,cnl = self.cns[ias]
                    if len(set(ias)) == 4: # in case of 3 membered ring
                        if self.iheav and np.any(self.zs[ias]==1):
                            continue
                        da = self.geom.get_dihedral_angle(ias, self.unit)
                        if self.idic4:
                            for keyi in [ tuple([i,j,k,l]), tuple([l,k,j,i]) ]:
                                mbs4[keyi] = da
                        else:
                            if self.icn:
                                if zj==zk and cnj==cnk:
                                    if zi>zl or (zi==zl and cni>cnl):
                                        ias = [l,k,j,i]
                            else:
                                if zj==zk and zi>zl:
                                    ias = [l,k,j,i]
                            zsi = [ self.zs[ia] for ia in ias ]
                            if self.icn:
                                tt = [ '%d_%d'%(self.zs[i],self.cns[i]) for i in ias ]
                            else:
                                tt = [ '%d'%(self.zs[i]) for i in ias ]
                            type4 = '-'.join(tt)
                            da = self.geom.get_dihedral_angle(ias, self.unit)
                            #print ' ++ ', type4, ias, [cni,cnj,cnk,cnl], tor
                            d1,d2,d3 = [ self.geom.ds[ias[a1],ias[a1+1]] for a1 in range(3) ]
                            a1,a2 = [ self.geom.get_angle(ias[a1:a1+3], unit=self.unit) for a1 in range(2) ]
                            val = [d1,d2,d3,a1,a2,da] if self.rpad else da
                            if type4 in list(mbs4.keys()):
                                mbs4[type4] += [val]
                            else:
                                mbs4[type4] = [val]
                            #vars4.append(val)
        if len(_buffer) == 0:
            print("No rotable bond found! Maybe you didn't have H's attached to heavy atoms?")
        #self.vars4 = np.array(vars4)
        return mbs4

    def get_all(self, nbody=4, isub=T):
        if nbody == 2:
            #if isub:
            self.mbs1 = self.get_atoms()
            self.mbs2 = self.get_bonds()
        elif nbody == 3:
            #if isub:
            self.mbs1 = self.get_atoms()
            self.mbs2 = self.get_bonds()
            self.mbs3 = self.get_angles()
        elif nbody == 4:
            #if isub:
            self.mbs1 = self.get_atoms()
            self.mbs2 = self.get_bonds()
            self.mbs3 = self.get_angles()
            self.mbs4 = self.get_dihedral_angles()
        else:
            raise Exception('Not implemented')

