Amons-based QML
===============

In the section, we cover 

* basic I/O, i.e., reading/writing info from xyz files and output files from quantum chemistry programs such as orca4
* generation of molecular representations: SLATM and its local conterpart 
* run KRR training/test based on SLATM

Again, one has to ``cd`` to the main directory of ``aqml``.


Basic I/O
---------

Reading from xyz files
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import aqml.io2 as io2

.. code:: python

    # get amons data used for training, as stored in xyz files
    fs = io2.cmdout('ls demo/example/reference/g7/f*z')
    fs

::

    ['../../demo/example/reference/g7/frag_01_c00001.xyz',
     '../../demo/example/reference/g7/frag_02_c00001.xyz',
     '../../demo/example/reference/g7/frag_03_c00001.xyz',
     '../../demo/example/reference/g7/frag_04_c00001.xyz',
     '../../demo/example/reference/g7/frag_05_c00001.xyz',
     '../../demo/example/reference/g7/frag_06_c00001.xyz',
     '../../demo/example/reference/g7/frag_07_c00001.xyz',
     '../../demo/example/reference/g7/frag_07_c00002.xyz',
     '../../demo/example/reference/g7/frag_08_c00001.xyz',
     '../../demo/example/reference/g7/frag_09_c00001.xyz',
     '../../demo/example/reference/g7/frag_10_c00001.xyz',
     '../../demo/example/reference/g7/frag_11_c00001.xyz',
     '../../demo/example/reference/g7/frag_11_c00002.xyz',
     '../../demo/example/reference/g7/frag_12_c00001.xyz',
     '../../demo/example/reference/g7/frag_12_c00002.xyz',
     '../../demo/example/reference/g7/frag_13_c00001.xyz']

A typical xyz file (to be used by ``aqml``) looks like:

.. code:: bash

      !cat demo/example/reference/g7/frag_01_c00001.xyz

::

    5
    alpha=13.42 b3lypvdz=-40.480429212044 
         C     -1.54626100      0.73185600      0.67141100
         H     -1.73565800      0.47084700      1.72356700
         H     -1.64277500      1.82028800      0.53911200
         H     -0.52974600      0.41588200      0.39233400
         H     -2.27556000      0.21702800      0.02997500

where ``alpha`` and ``b3lypvdz`` are the polarizability (in
Bohr\ :math:`^3`) and total energy (in Hartree) of CH4 molecule.
``b3lypvdz`` indicates that all properties were calculated at the level
of theory ``b3lyp/cc-pvdz`` (by orca4, if not otherwise stated).

To read geometry together with all properties of one molecule (say,
CH4), do the following

.. code:: python

    import aqml.cheminfo.core as cc

    mol = cc.molecule(fs[0], ['a'])
    mol.props

::

    {'alpha': array([13.42]), 'b3lypvdz': array([-25401.85283769])}

Note that atomic units for energetic properties would be automaically
converted to units that are used more favorably by Chemists, namely,
kcal/mol.

To read geometries and all properties of multiple molecules (say, the
first 5 mols):

.. code:: python

    mols = cc.molecules(fs[:5], ['a'])
    mols.props

::

    {'alpha': array([[13.42],
            [21.68],
            [13.57],
            [33.96],
            [25.19]]), 'b3lypvdz': array([[-25401.85283769],
            [-49280.26416003],
            [-71817.46145015],
            [-73935.93419366],
            [-96479.43688458]])}

Note that the second entry of function ``cc.molecule`` and
``cc.molecules`` is the list of names of properties. If it's set to
``['a']``, then all properties would be read and it's equivalent to
specify the second entry to ``['alpha','b3lypvdz']``.

Reading from orca output file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import aqml.io2.orca as orca

    #from importlib import reload
    #reload(orca)

.. code:: python

    fs = io2.cmdout('ls ../../demo/example/reference/g7/f*out')

.. code:: python

    obj = orca.orcajob(fs[0])

Note that only serial mode is supported, i.e., one output file each
time.

Now get the method and total energy from the output file:

.. code:: python

    obj.e

::

    {'b3lypvdz': -40.480429212044}

Output file produced by other programs such as ``Molpro``,
``Gaussian 09`` are also supported and the corresponding usages will be
documented in the near future.

Generation of SLATM and ML (KRR)
--------------------------------

.. code:: python

    import cml.algo.aqml as aq
    reload(aq)

::

    <module 'cml.algo.aqml' from '/home/bing/anaconda3/lib/python3.7/site-packages/cml/algo/aqml.py'>

.. code:: python

    T, F = True, False

    # root directory storing all relevant data
    root = '../../demo/example/reference/'

    # amons data are stored in in xyz files under folder `g7`
    train = [root + 'g7/']

    # test data are stored in xyz file under folder `target/`
    test = [root + 'target/']

    # representation generation and krr can be done within one line of commmand
    obj = aq.calculator(iaml=T, 
                     train=train, test=test, lambdas=[4.0], coeffs=[1.], kernel='g', 
                     rcut=4.8, p='b3lypvdz', nprocs=1, i_disp_eaq=F, debug=F)

Now run training & test

.. code:: python

    obj.run()

