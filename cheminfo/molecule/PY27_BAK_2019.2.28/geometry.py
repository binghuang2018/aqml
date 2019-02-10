
import scipy.spatial.distance as ssd
import numpy as np

class Geometry(object):

    def __init__(self, coords):
        coords = np.array(coords)
        self.coords = coords
        #self.ds = ssd.squareform( ssd.pdist(coords) )
        self.ds = np.sqrt((np.square(coords[:,np.newaxis]-coords).sum(axis=2)))

    def get_distance(self, idx):
        return self.ds[idx[0],idx[1]]

    def get_angle(self, idx):
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
        return np.arccos(ang) * 180/np.pi

    def get_dihedral_angle(self,idx):
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
        angle = np.arccos(angle) * 180 / np.pi
        if np.vdot(bxa, c) > 0:
            angle = 360 - angle
        return angle


