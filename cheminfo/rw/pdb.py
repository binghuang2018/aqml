"""
Module to read and write atoms in PDB file format.

+++++++++++++++++++++++++++++++++++++++++++
+++                 NOTE                +++
+++   We are intented to read/write     +++
+++   organic molecules only, for now   +++
+++   NO peptide/protein                +++
+++++++++++++++++++++++++++++++++++++++++++


PDB format wiki
http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html

1) MODEL
The MODEL record specifies the model serial number when multiple models
of the same structure are presented in a single coordinate entry, as is
often the case with structures determined by NMR.

Record Format

COLUMNS        DATA  TYPE    FIELD          DEFINITION
----------------------------------------------------------------
 1 -  6        Record name   "MODEL "
11 - 14        Integer       serial         Model serial number.
Details
This record is used only when more than one model appears in an entry.
Generally, it is employed mainly for NMR structures. The chemical
connectivity should be the same for each model. ATOM, HETATM, ANISOU,
and TER records for each model structure and are interspersed as needed
between MODEL and ENDMDL records.


2) ATOM
The ATOM records present the atomic coordinates for standard amino acids
and nucleotides. They also present the occupancy and temperature factor
for each atom. Non-polymer chemical coordinates use the HETATM record
type. The element symbol is always present on each ATOM record; charge
is optional.

Changes in ATOM/HETATM records result from the standardization atom and
residue nomenclature. This nomenclature is described in the Chemical
Component Dictionary (ftp://ftp.wwpdb.org/pub/pdb/data/monomers).

Record Format

COLUMNS        DATA  TYPE    FIELD        DEFINITION
12345678901234567890123456789012345678901234567890123456789012345678901234567890
-------------------------------------------------------------------------------------
 1 -  6        Record name   "ATOM  "
 7 - 11        Integer       serial       Atom  serial number.
13 - 16        Atom          name         Atom name.
17             Character     altLoc       Alternate location indicator.
18 - 20        Residue name  resName      Residue name.
22             Character     chainID      Chain identifier.
23 - 26        Integer       resSeq       Residue sequence number.
27             AChar         iCode        Code for insertion of residues.
31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
55 - 60        Real(6.2)     occupancy    Occupancy.
61 - 66        Real(6.2)     tempFactor   Temperature  factor.
77 - 78        LString(2)    element      Element symbol, right-justified.
79 - 80        LString(2)    charge       Charge  on the atom.


3) CONECT
The CONECT records specify connectivity between atoms for which coordinates are supplied.
The connectivity is described using the atom serial number as shown in the entry. CONECT
records are mandatory for HET groups (excluding water) and for other bonds not specified
in the standard residue connectivity table. These records are generated automatically.

Record Format

COLUMNS       DATA  TYPE      FIELD        DEFINITION
-------------------------------------------------------------------------
 1 -  6        Record name    "CONECT"
 7 - 11       Integer        serial       Atom  serial number
12 - 16        Integer        serial       Serial number of bonded atom
17 - 21        Integer        serial       Serial  number of bonded atom
22 - 26        Integer        serial       Serial number of bonded atom
27 - 31        Integer        serial       Serial number of bonded atom

"""

import numpy as np
import aqml.cheminfo as co

def write_pdb(images, fileobj=None, cell=None, prop={}, sort_atom=False):
    """Write images to PDB-file.
    images could be a tuple (symbols,coords,charges,bom) or a list
    of tuples [ (symbols_1,charges_1,bom_1,coords_1),
                (symbols_2,charges_2,bom_2,coords_2),
                ... ]
    """

    if isinstance(images, tuple):
        assert len(images) == 4
        images = [images]
    else:
        for image in images:
            assert len(image) == 4

    so = '' # str_output

    if cell != None:
        # ignoring Z-value, using P1 since we have all atoms defined explicitly
        fmt = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
        so += fmt % (cell[0], cell[1], cell[2],
                     cell[3], cell[4], cell[5])

    #     1234567 123 6789012345678901   89   67   456789012345678901234567 890
    fmt1 = ('ATOM  %5d %4s MOL     1    %8.3f%8.3f%8.3f  1.00  0.00'
              '          %2s%2s\n')

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    dic = {-1:'-', 1:'+'}
    for i, _image in enumerate(images):
        nucs, _chgs, bom, ps = _image
        bom = bom.astype(np.int)
        if not isinstance(nucs[0], str):
            symbs = [ co.chemical_symbols[_iz] for _iz in nucs ]
        else:
            symbs = nucs
        scs = []
        _achgs = np.abs(_chgs)
        _signs = np.sign(_chgs)
        na = len(_chgs)
        for ia in range(na):
            sc = ''
            if _achgs[ia] != 0:
                sc = '%d%s'%(_achgs[ia],dic[_signs[ia]])
            scs.append(sc)
        so = so + 'MODEL     ' + str(i + 1) + '\n'
        # write atoms
        for ia in range(na):
            x, y, z = ps[ia]
            so += fmt1 % ((ia+1) % MAXNUM, symbs[ia],
                          x, y, z, symbs[ia].upper(), scs[ia])
        # write bonds
        idxs = np.array( np.where( np.triu(bom) > 0 ), np.int ).T
        #idxs = np.array( np.where( bom > 0 ), np.int ).T
        nb = len(idxs)
        for ib in range(nb):
            #        1234561234512345123451234512345
            #e.g.,  'CONECT    6    7    7    8    8'
            ia1, ia2 = idxs[ib]; bo12 = bom[ia1,ia2]
            s1 = '%5d'%(ia1+1); s2 = '%5d'%(ia2+1)
            so = so + 'CONECT' + s1 + s2*bo12 + '\n'
        so += 'ENDMDL\n'

    if isinstance(fileobj, str):
        fileobj = open(fileobj, 'w')
        fileobj.write(so)
    else:
        return so


