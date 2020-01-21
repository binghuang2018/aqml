"""Reads chemical data in SDF format (wraps the molfile format).

See https://en.wikipedia.org/wiki/Chemical_table_file#SDF
"""
from ase.atoms import Atoms


def read_sdf(fileobj,iconn=False):
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    lines = fileobj.readlines()
    # first three lines header
    del lines[:3]

    sn = lines.pop(0)[:3] #.split()
    natoms = int(sn) #L1[0])
    positions = []
    symbols = []
    for line in lines[:natoms]:
        x, y, z, symbol = line.split()[:4]
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    m = Atoms(symbols=symbols, positions=positions)
    if iconn: # connectivity
        ctab = cs[na+4:na+nb+4]
        bom = np.zeros((na,na))
        for c in ctab:
            idx1,idx2,bo12 = int(c[:3]), int(c[3:6]), int(c[6:9])
            bom[idx1-1,idx2-1] = bom[idx2-1,idx1-1] = bo12
        return m, bom
    else:
        return m


