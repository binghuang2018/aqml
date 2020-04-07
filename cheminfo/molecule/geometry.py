
import scipy.spatial.distance as ssd
import numpy as np

T,F = True,False

# ref. http://en.wikipedia.org/wiki/Rotation_matrix
def rotation_mat(vec, theta):
    '''rotate angle theta along vec
    new(x,y,z) = R * old(x,y,z)'''
    vec = _normalize(vec)
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r


class Geometry(object):

    def __init__(self, obj, ds=None, zs=None):

        if isinstance(obj, str):
            # assume z-mat
            self.coords = self.zmat2cart(obj)
        elif isinstance(obj, (list,np.ndarray)):
            self.coords = np.array(obj)
        else:
            raise Exception('type of `obj not supported')
        self._rs = ds #ssd.squareform( ssd.pdist(coords) )
        self._zs = zs

    @property
    def ds(self):
        if not hasattr(self, '_ds'):
            if self._rs is None:
                _ds = np.sqrt((np.square(self.coords[:,np.newaxis]-self.coords).sum(axis=2)))
            else:
                _ds = self._rs
            self._ds = _ds
        return self._ds

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

    def zmat2cart(self):
        '''>>> a = """H
        H 1 2.67247631453057
        H 1 4.22555607338457 2 50.7684795164077
        H 1 2.90305235726773 2 79.3904651036893 3 6.20854462618583"""
        >>> for x in zmat2cart(a): print(x)
        ['H', array([ 0.,  0.,  0.])]
        ['H', array([ 2.67247631,  0.        ,  0.        ])]
        ['H', array([ 2.67247631,  0.        ,  3.27310166])]
        ['H', array([ 0.53449526,  0.30859098,  2.83668811])]
        '''
        from pyscf.symm import rotation_mat
        atomstr = atomstr.replace(';','\n').replace(',',' ')
        symb = []
        coord = []
        min_items_per_line = 1
        for line in atomstr.splitlines():
            line = line.strip()
            if line and line[0] != '#':
                rawd = line.split()
                assert(len(rawd) >= min_items_per_line)

                symb.append(rawd[0])
                if len(rawd) < 3:
                    coord.append(np.zeros(3))
                    min_items_per_line = 3
                elif len(rawd) == 3:
                    coord.append(np.array((float(rawd[2]), 0, 0)))
                    min_items_per_line = 5
                elif len(rawd) == 5:
                    bonda = int(rawd[1]) - 1
                    bond  = float(rawd[2])
                    anga  = int(rawd[3]) - 1
                    ang   = float(rawd[4])/180*np.pi
                    assert(ang >= 0)
                    v1 = coord[anga] - coord[bonda]
                    if not np.allclose(v1[:2], 0):
                        vecn = np.cross(v1, np.array((0.,0.,1.)))
                    else: # on z
                        vecn = np.array((0.,0.,1.))
                    rmat = rotation_mat(vecn, ang)
                    c = np.dot(rmat, v1) * (bond/np.linalg.norm(v1))
                    coord.append(coord[bonda]+c)
                    min_items_per_line = 7
                else:
                    bonda = int(rawd[1]) - 1
                    bond  = float(rawd[2])
                    anga  = int(rawd[3]) - 1
                    ang   = float(rawd[4])/180*np.pi
                    assert(ang >= 0 and ang <= np.pi)
                    v1 = coord[anga] - coord[bonda]
                    v1 /= np.linalg.norm(v1)
                    if ang < 1e-7:
                        c = v1 * bond
                    elif np.pi-ang < 1e-7:
                        c = -v1 * bond
                    else:
                        diha  = int(rawd[5]) - 1
                        dih   = float(rawd[6])/180*np.pi
                        v2 = coord[diha] - coord[anga]
                        vecn = np.cross(v2, -v1)
                        vecn_norm = np.linalg.norm(vecn)
                        if vecn_norm < 1e-7:
                            if not np.allclose(v1[:2], 0):
                                vecn = np.cross(v1, np.array((0.,0.,1.)))
                            else: # on z
                                vecn = np.array((0.,0.,1.))
                            rmat = rotation_mat(vecn, ang)
                            c = np.dot(rmat, v1) * bond
                        else:
                            rmat = rotation_mat(v1, -dih)
                            vecn = np.dot(rmat, vecn) / vecn_norm
                            rmat = rotation_mat(vecn, ang)
                            c = np.dot(rmat, v1) * bond
                    coord.append(coord[bonda]+c)
        atoms = list(zip([_atom_symbol(x) for x in symb], coord))
        return atoms

    def x2z(self, var=T):
        '''
           convert cartesian coord to z-mat
        '''
        zstr = ['']
        return '\n'.join(zstr)


    @property
    def zmat(self):
        """ cartesian to z-mat """
        if not hasattr(self, '_zmat'):
            self._zmat = self.x2z()
        return self._zmat


    def check_dihedral(self, construction_table):
        """
          Check if the dihedral defining atom is colinear.
        Checks for each index starting from the third row of the
        ``construction_table``, if the reference atoms are colinear.
        Args:
            construction_table (pd.DataFrame):
        Returns:
            list: A list of problematic indices.
        """
        c_table = construction_table
        angles = self.get_angle_degrees(c_table.iloc[3:, :].values)
        problem_index = np.nonzero((175 < angles) | (angles < 5))[0]
        rename = dict(enumerate(c_table.index[3:]))
        problem_index = [rename[i] for i in problem_index]
        return problem_index




class GraphGeometry(Geometry):

    def __init__(self, obj, g, icn=F, iheav=F):
        if isinstance(obj,(tuple,list)):
            zs, coords = obj
        else:
            zs, coords = obj.zs, obj.coords # assume aqml.cheminfo.core.atoms/molecule object
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

