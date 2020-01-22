
Experimental geometry of 1ubq (file raw/1ubq.pdb) was taken from Protein Data Bank (PDB) https://www.rcsb.org/structure/1ubq. 

Before carrying out ab-initio calculations, all charges (of some N and O atoms) are removed and hydrogens are added to saturate valencies of heavy atoms by OEChem. The ready-to-use pdb file is 1ubq.pdb

Experimental geometry was used throughout for electronic structure calculation. 

- for single point energy calculation, Turbomole-7.2 was used and the associated controling parameters are

```bash
$dft
   functional pbe
   gridsize   m4
```
together with a basis of `def2-svp` for all atoms.

- for NMR shift calculation, orca4 was used instead with basically the same setting: 

```bash
! pbe TIGHTSCF def2-svp NMR

%eprnmr
 Nuclei = all H { shift }
 Nuclei = all C { shift }
 Nuclei = all N { shift }
end
```

