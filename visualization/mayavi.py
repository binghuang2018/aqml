
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from cheminfo.molecule.core import *
from mayavi.tools.helper_functions import *

def draw_molecule(atoms,data,cell,orig=None):
    res = np.array(data.shape) / cell
    objs = []
    # Plot the atoms as spheres:
    _zs = []
    _coords = []
    ps, zs = atoms.positions, atoms.numbers


    na = len(zs)
    for i in range(na):
        psi = ps[i]
        if orig is not None:
            psi -= orig
        ps[i] = psi * res
    for pos, Z in zip(ps,zs):
        if Z not in _zs:
            _zs.append(Z)
            _coords.append( [pos] )
        else:
            idx = _zs.index(Z)
            _coords[idx] += [pos]
    nzu = len(_zs)
    # plot atoms
    for i in range(nzu):
        Z = _zs[i]
        coords_i = _coords[i]
        ox, oy, oz = list(map(np.array, zip(*coords_i)))
        ai = points3d(ox, oy, oz,
                      scale_factor=covalent_radii[Z],
                      resolution=20,
                      color=tuple(cpk_colors[Z]))
        objs.append(ai)
    # now bonds
    rawm = RawMol(zs, ps)
    rawm.connect()
    g = rawm.g
    bonds = [ list(edge) for edge in \
             np.array( list( np.where(np.triu(g)>0) ) ).T ]
    for b in bonds:
        coords_b = ps[b]
        ox, oy, oz = list(map(np.array, zip(*coords_b)))
        bi = plot3d(ox, oy, oz, [1, 2],
                    tube_radius=0.06, color=(0.6,0.6,0.6)) #map='gray')
        objs.append(bi)
    # axes
    xx = yy = zz = np.arange(0,1,0.1) #-0.6,0.7,0.1)
    lensoffset = 0.0
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
    axis_x = plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01,color=(1,0,0))
    axis_y = plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01,color=(0,1,0))
    axis_z = plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01,color=(0,0,1))
    objs += [axis_x, axis_y, axis_z]

    # now volumetric data
    #iso1 = contour3d(data, contours=[0.07], opacity=0.5, color=(1,0,0)) #transparent=F,  colormap='hot')
    #iso2 = contour3d(data, contours=[-0.07], opacity=0.5, color=(0,1,0)) # transparent=F, colormap='hot')
    #objs += [iso1,iso2]
    return objs[0]


from ase.io.cube import read_cube_data

def read_cube(f):
    assert os.path.exists(obj), '#ERROR: file does not exist?'
    data, atoms = read_cube_data(obj)
    return

