
import numpy as np
from mayavi import mlab
import ase.io.cube as aic
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from cheminfo.molecule.core import *

dc = {'red':(1,0,0), 'green':(0,1,0), 'blue':(0,0,1), 'gray':(0.5,0.5,0.5)}

def contour3d(isovalues, colors, fid=1, atoms=None, data=None, box=None, origin=None, filename=None):
    mlab.figure(bgcolor=(0, 0, 0), size=(350, 350))
    #mlab.clf()
    if filename is not None:
        fobj = open(filename,'r')
        dt = aic.read_cube(fobj, read_data=True)
        atoms, origin, data = [ dt[key] for key in ['atoms','origin','data'] ]
        box = atoms.cell
    else:
        assert atoms is not None
        assert data is not None
        assert box is not None
        if len(box.shape) == 1:
            _box = np.eye(3)
            _box[ np.diag_indices_from(_box) ] = box
            box = _box
        if origin is None: origin = np.zeros(3)

    _zs = atoms.numbers
    na = len(_zs)
    # get bonds
    newm = RawMol(_zs, atoms.positions)
    newm.connect()
    bonds = [ list(edge) for edge in \
                np.array( list( np.where(np.triu(newm.g)>0) ) ).T ]

    box0 = np.zeros((3,3))
    box0[ np.diag_indices_from(box0) ] = np.diag(box)
    assert np.all(box0 == box), '#ERROR: not a cubic box?'

    grids = data.shape
    scales = np.array(data.shape)/np.diag(box)
    scale_av = np.mean(scales)

    # rescale the position of the atoms
    _coords = np.array( [ (atoms.positions[i]-origin) * scales for i in range(na) ] )

    nzs = []
    zsu = []
    coords = []
    for i,zi in enumerate(_zs):
        if zi in zsu:
            idx = zsu.index(zi)
            coords[idx] += [ _coords[i] ]
        else:
            zsu.append(zi)
            coords.append( [_coords[i]] )
    nzu = len(zsu)

    objs = []
    for i in range(nzu):
        Z = zsu[i]
        coords_i = coords[i]
        ox, oy, oz = list(map(np.array, zip(*coords_i)))
        ai = mlab.points3d(ox, oy, oz, 
                      scale_factor=covalent_radii[Z] * scale_av,
                      resolution=20,
                      color=tuple(cpk_colors[Z]))
        objs.append(ai)

    # The bounds between the atoms, we use the scalar information to give
    # color
    for bond in bonds:
        coords_b = _coords[bond]
        ox, oy, oz = list(map(np.array, zip(*coords_b)))
        obj = mlab.plot3d(ox, oy, oz, [1, 2],
                tube_radius=0.05*scale_av, color=(0.5,0.5,0.5))
        objs.append(obj)
    for i,val in enumerate(isovalues):
        color = dc[colors[i]]
        obj = mlab.contour3d(data, contours=[val], opacity=0.25, color=color) #(1,0,0))
        objs.append(obj)
    objs.append(obj)

    # https://gitlab.com/chaffra/ase/blob/master/ase/visualize/mlab.py
    # Do some tvtk magic in order to allow for non-orthogonal unit cells:
    #polydata = obj.actor.actors[0].mapper.input
    #pts = np.array(polydata.points) - 1
    # Transform the points to the unit cell:
    #polydata.points = np.dot(pts, box / np.array(data.shape)[:, np.newaxis])

    #mlab.axes(x_axis_visibility=True) #,y_axis_visibility=True)
    #mlab.axes(xlabel='X', ylabel='Y', zlabel='Z') #Display axis
    obj = mlab.orientation_axes()
    objs.append(obj)

    return objs[0]


