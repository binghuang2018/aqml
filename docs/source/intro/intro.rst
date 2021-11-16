The ``aqml``
============

What it is
----------

AQML is a mixed Python/Fortran/C++ package, intends to simulate quantum
chemistry problems through the use of amons --- the fundamental building
blocks of larger systems (such as protein, solid).

Features
--------

-  Amons selection algorithm: automatic sampling of training set that are most representative for the query
-  composition and size-independent
-  currently limited to molecule/solid with explicit graph (w/wo periodic boundary condition)
-  Molecular representation
-  Parameter-free global SLATM (Spectrum of London and Axilrod-Teller-Muto potential) representations

   -  global SLATM and its local (atomic) counterpart aSLATM.
   -  A new pair-wise version of aSLATM is available, dealing favorably with dataset involving many elements.

-  Bond, Angle ML (BAML) representation, including up to 4-body potential (UFF inspired)
-  Graph-based representation (coming soon...)

-  Machine learning
-  KRR and multi-fidelity KRR
-  Deep neural network potential (under developement...)
-  Automatic workflow of generating quantum chemical reference data. The following programs are currently supported:

   -  G09
   -  ORCA4
   -  MOLPRO
   -  CASINO (a QMC package)


Purpose of ``aqml``
-------------------

-  training set selection
-  representation design
-  ML algorithm

