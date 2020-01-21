
import scipy.spatial.distance as ssd
import numpy as np

T,F = True,False

class Geometry(object):

    def __init__(self, coords, ds=None):
        coords = np.array(coords)
        self.coords = coords
        #self.ds = ssd.squareform( ssd.pdist(coords) )
        if ds is None:
            ds = np.sqrt((np.square(coords[:,np.newaxis]-coords).sum(axis=2)))
        self.ds = ds

    def get_distance(self, idx):
        return self.ds[idx[0],idx[1]]

    def get_angle(self, idx, unit='rad'):
        """ angle spanned by vec_j_i, vec_j_k """
        i,j,k = idx
        vji = self.coords[i] - self.coords[j]
        vjk = self.coords[k] - self.coords[j]
        nvji = vji/self.ds[j,i]
        nvjk = vjk/self.ds[j,k]
        _ang = np.vdot(nvji,nvjk)
        if _ang < -1:
            ang = -1
        elif _ang > 1:
            ang = 1
        else:
            ang = _ang
        const = 1.0 if unit == 'rad' else 180./np.pi
        return np.arccos(ang) * const

    def get_dihedral_angle(self,idx,unit='rad'):
        """
         get absolute dihedral angle
        """
        i,j,k,l = idx
        a = self.coords[j] - self.coords[i]
        b = self.coords[k] - self.coords[j]
        c = self.coords[l] - self.coords[k]
        bxa = np.cross(b, a)
        bxa /= np.linalg.norm(bxa)
        cxb = np.cross(c, b)
        cxb /= np.linalg.norm(cxb)
        angle = np.vdot(bxa, cxb)
        # check for numerical trouble due to finite precision:
        if angle < -1:
            angle = -1
        if angle > 1:
            angle = 1
        _tor = np.arccos(angle)
        #if np.vdot(bxa, c) > 0:
        #    _tor = 2.*np.pi - _tor

        maxa = np.pi
        maxa2 = np.pi * 2.
        if _tor > maxa:
            tor = maxa2 - _tor
        elif _tor < -maxa:
            tor = maxa2 + _tor
        else:
            tor = np.abs(_tor)

        const = 1.0 if unit == 'rad' else 180./np.pi
        return _tor * const


class GraphGeometry(Geometry):

    def __init__(self, obj, g, icn=F, iheav=F):
        if isinstance(obj,(tuple,list)):
            zs, coords = obj
        else:
            zs, coords = obj.zs, obj.coords # assume cheminfo.core.atoms/molecule object
        Geometry.__init__(self, coords)
        self.g = g
        self.icn = F
        self.iheav = iheav # consider heavy atoms only?
        self.cns = g.sum(axis=0)
        self.zs = np.array(zs,dtype=int)
        self.na = len(zs)
        ias = np.arange(self.na)
        self.ias = ias
        self.ias_heav = ias[self.zs > 1]

    def get_angles(self, jas=[], unit='rad'):
        """ get all angles spanned by two cov bonds """
        mbs3 = {}
        if len(jas) == 0:
            jas = self.ias_heav # allows for user-specified central atoms
        for j in jas:
            zj = self.zs[j]
            neibs = self.ias[ self.g[j] > 0 ]
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
                        if self.icn:
                            tt = [ '%d_%d'%(self.zs[_],self.cns[_]) for _ in ias ]
                        else:
                            tt = [ '%d'%self.zs[_] for _ in ias ]
                        type3 = '-'.join(tt)
                        theta = self.get_angle(ias, unit=unit) # in degree
                        if type3 in list(mbs3.keys()):
                            mbs3[type3] += [theta]
                        else:
                            mbs3[type3] = [theta]
        return mbs3

