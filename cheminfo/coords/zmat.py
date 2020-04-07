
import numpy as np
import itertools as itl
import aqml.cheminfo as ci 
import aqml.cheminfo.molecule.core as cmc 

abc='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ss = [ si+'*' for si in abc ] + [ si[0]+si[1]+'*' for si in itl.product(abc,abc) ]
ss1 = [ si for si in abc ] + [ si[0]+si[1] for si in itl.product(abc,abc) ]

def replace_vars(vlist, variables):
    for i in range(len(vlist)):
        if vlist[i] in variables:
            value = variables[vlist[i]]
            vlist[i] = value
        else:
            try:
                value = float(vlist[i])
                vlist[i] = value
            except:
                print("Problem with entry " + str(vlist[i]))



###### partially adapted to codes of package `chemcoord`

class Cart(ci.atoms, cmc.RawMol):

    """ 
      cartesian object
    """

    def __init__(self, zs, coords):
        ci.atoms.__init__(self, zs, coords, index=None)
        cmc.RawMol.__init__(self)

        frame = pd.DataFrame(index=index,
                             columns=['atom', 'x', 'y', 'z'], dtype='f8')
        frame['atom'] = atoms
        frame.loc[:, ['x', 'y', 'z']] = coords
        self._frame = frame.copy()

    def _get_ctab(self, iatr=None, ctabr=None):
        """Create a construction table (ctab) for a Zmatrix.
        A `ctab is basically a Zmatrix without the values
        for the bond lenghts, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.
        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.
        By default the first atom is the one nearest to the centroid.
        (Compare with :meth:`~Cartesian.get_centroid()`)
        Args:
            iatr: An index for the first atom may be provided.
            ctabr (pd.DataFrame): An uncomplete construction table
                may be provided. The rest is created automatically.
        Returns:
            pd.DataFrame: Construction table
        """

        def modify_priority(bond_dict, user_defined):
            def move_to_start(dct, key):
                "Due to PY27 compatibility"
                keys = dct.keys()
                if key in keys and key != keys[0]:
                    root = dct._OrderedDict__root
                    first = root[1]
                    link = dct._OrderedDict__map[key]
                    link_prev, link_next, _ = link
                    link_prev[1] = link_next
                    link_next[0] = link_prev
                    link[0] = root
                    link[1] = first
                    root[1] = first[0] = link
                else:
                    raise KeyError

            for j in reversed(user_defined):
                try:
                    try:
                        bond_dict.move_to_end(j, last=False)
                    except AttributeError:
                        # No move_to_end method in python 2.x
                        move_to_start(bond_dict, j)
                except KeyError:
                    pass

        if (iatr is not None) and (ctabr is not None):
            raise Exception('Either `iatr or `ctabr has to be None')

        bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)

        if ctabr is not None:
            self._check_construction_table(ctabr)
            construction_table = ctabr.copy()

        if ctabr is None:
            if iatr is None:
                molecule = self.get_distance_to(self.get_centroid())
                i = molecule['distance'].idxmin()
            else:
                i = iatr
            order_of_def = [i]
            user_defined = []
            construction_table = {i: {'b': 'origin',
                                      'a': 'e_z',
                                      'd': 'e_x'}}
        else:
            i = construction_table.index[0]
            order_of_def = list(construction_table.index)
            user_defined = list(construction_table.index)
            construction_table = construction_table.to_dict(orient='index')

        visited = {i}
        if len(self) > 1:
            parent = {j: i for j in bond_dict[i]}
            work_bond_dict = OrderedDict(
                [(j, bond_dict[j] - visited) for j in bond_dict[i]])
            modify_priority(work_bond_dict, user_defined)
        else:
            parent, work_bond_dict = {}, {}

        while work_bond_dict:
            new_work_bond_dict = OrderedDict()
            for i in work_bond_dict:
                if i in visited:
                    continue
                if i not in user_defined:
                    b = parent[i]
                    if b in order_of_def[:3]:
                        if len(order_of_def) == 1:
                            construction_table[i] = {'b': b,
                                                     'a': 'e_z',
                                                     'd': 'e_x'}
                        elif len(order_of_def) == 2:
                            a = (bond_dict[b] & set(order_of_def))[0]
                            construction_table[i] = {'b': b, 'a': a,
                                                     'd': 'e_x'}
                        else:
                            try:
                                a = parent[b]
                            except KeyError:
                                a = (bond_dict[b] & set(order_of_def))[0]
                            try:
                                d = parent[a]
                                if d in set([b, a]):
                                    message = "Don't make self references"
                                    raise UndefinedCoordinateSystem(message)
                            except (KeyError, UndefinedCoordinateSystem):
                                try:
                                    d = ((bond_dict[a] & set(order_of_def))
                                         - set([b, a]))[0]
                                except IndexError:
                                    d = ((bond_dict[b] & set(order_of_def))
                                         - set([b, a]))[0]
                            construction_table[i] = {'b': b, 'a': a, 'd': d}
                    else:
                        a, d = [construction_table[b][k] for k in ['b', 'a']]
                        construction_table[i] = {'b': b, 'a': a, 'd': d}
                    order_of_def.append(i)

                visited.add(i)
                for j in work_bond_dict[i]:
                    new_work_bond_dict[j] = bond_dict[j] - visited
                    parent[j] = i

            work_bond_dict = new_work_bond_dict
            modify_priority(work_bond_dict, user_defined)
        output = pd.DataFrame.from_dict(construction_table, orient='index')
        output = output.loc[order_of_def, ['b', 'a', 'd']]
        return output



def readzmat(filename):
    zmatf = open(filename, 'r')
    atomnames = []
    rconnect = []
    rlist = []
    aconnect = []
    alist = []
    dconnect = []
    dlist = []
    variables = {}

    if not zmatf.closed:
        for line in zmatf:
            words = line.split()
            eqwords = line.split('=')

            if len(eqwords) > 1:
                varname = str(eqwords[0])
                try:
                    varval  = float(eqwords[1])
                    variables[varname] = varval
                except:
                    print("Invalid variable definition: " + line)

            else:
                if len(words) > 0:
                    atomnames.append(words[0])
                if len(words) > 1:
                    rconnect.append(int(words[1]))
                if len(words) > 2:
                    rlist.append(words[2])
                if len(words) > 3:
                    aconnect.append(int(words[3]))
                if len(words) > 4:
                    alist.append(words[4])
                if len(words) > 5:
                    dconnect.append(int(words[5]))
                if len(words) > 6:
                    dlist.append(words[6])

    replace_vars(rlist, variables)
    replace_vars(alist, variables)
    replace_vars(dlist, variables)

    return (atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist)



def angle(xyzarr, i, j, k):
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi
    return theta

def dihedral(xyzarr, i, j, k, l):
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi

def write_zmat(xyzarr, distmat, atomnames, rvar=False, avar=False, dvar=False):
    npart, ncoord = xyzarr.shape
    rlist = []
    alist = []
    dlist = []
    if npart > 0:
        # Write the first atom
        print(atomnames[0])

        if npart > 1:
            # and the second, with distance from first
            n = atomnames[1]
            rlist.append(distmat[0][1])
            if (rvar):
                r = 'R%s'%ss[0]
            else:
                r = '{:>11.5f}'.format(rlist[0])
            #print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))
            print('{:s} {:d} {:s}'.format(n, 1, r))

            if npart > 2:
                n = atomnames[2]

                rlist.append(distmat[0][2])
                if (rvar):
                    r = 'R%s'%ss[1]
                else:
                    r = '{:>11.5f}'.format(rlist[1])

                alist.append(angle(xyzarr, 2, 0, 1))
                if (avar):
                    t = 'A%s'%ss[0]
                else:
                    t = '{:>11.5f}'.format(alist[0])

                #print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))
                print('{:s} {:d} {:s} {:d} {:s}'.format(n, 1, r, 2, t))

                if npart > 3:
                    for i in range(3, npart):
                        n = atomnames[i]

                        rlist.append(distmat[i-3][i])
                        if (rvar):
                            r = 'R%s'%ss[i-1] #{:<4d}'.format(i)
                        else:
                            r = '{:>11.5f}'.format(rlist[i-1])

                        alist.append(angle(xyzarr, i, i-3, i-2))
                        if (avar):
                            t = 'A%s'%ss[i-2] #{:<4d}'.format(i-1)
                        else:
                            t = '{:>11.5f}'.format(alist[i-2])

                        dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                        if (dvar):
                            d = 'D%s'%ss[i-3] #{:<4d}'.format(i-2)
                        else:
                            d = '{:>11.5f}'.format(dlist[i-3])
                        #print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
                        print('{:s} {:d} {:s} {:d} {:s} {:d} {:s}'.format(n, i-2, r, i-1, t, i, d))
    if (rvar):
        print(" ")
        for i in range(npart-1):
            print('R%s=%.5f'%(ss1[i], rlist[i])) #{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
    if (avar):
        #print(" ")
        for i in range(npart-2):
            print('A%s=%.5f'%(ss1[i],alist[i])) #{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
    if (dvar):
        #print(" ")
        for i in range(npart-3):
            print('D%s=%.5f'%(ss1[i],dlist[i])) #{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))


def write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist):
    npart = len(atomnames)
    print(npart)
    print('INSERT TITLE CARD HERE')

    xyzarr = np.zeros([npart, 3])
    if (npart > 1):
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if (npart > 2):
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        r = rlist[n-1]
        theta = alist[n-2] * np.pi / 180.0
        phi = dlist[n-3] * np.pi / 180.0

        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta

        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]

        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)

        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]

    for i in range(npart):
        print('{:<4s}\t{:>11.5f}\t{:>11.5f}\t{:>11.5f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))


