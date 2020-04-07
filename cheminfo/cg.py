




import aqml.cheminfo.RDKit as cir

###############
class cg(object):

    def __init__(self, obj):


        m = Chem.MolFromMolFile(sdf,removeHs=False,strictParsing=False)
        self.m = m

    def get_mbtypes_cg(self):

        """
        BH2
        BeH
        CH3
        NH2
        OH
        """

    def get_rigid_nodes(self):
        """
        rigid nodes include two types of nodes:
        1) some ring structure, note that not all nodes in
           a ring are rigid, esp. the rings made up of single bonds
        2) conjugate functional groups, e.g.,
            -C(=O)N-
            -C=C-C=C-
            -C=N#N
            ...
        """


    def get_rings(self, namin=3, namax=9):

        """
        get nodes of `namin- to `namax-membered ring

        We focus on those nodes which constitute the so-called
        `extended smallest set of small unbreakable fragments,
        including aromatic rings, 3- and 4-membered rings
        (accompanied with high strain)
        interactions in amons) and

        """
        import itertools as itl

        m = self.m

        # first search for rings
        sets = []
        for i in range(namin, namax+1):
            #if i in [3,4,5]:
            pat_i = '*~1' + '~*'*(i-2) + '~*1'
            #else:
            #    pat_i = '*:1' + ':*'*(i-2) + ':*1'
            Qi = Chem.MolFromSmarts( pat_i )
            for tsi in m.GetSubstructMatches(Qi):
                set_i = set(tsi)
                if set_i not in sets:
                    sets.append( set(tsi) )

        # now remove those rings that are union of smaller rings
        n = len(sets)
        sets_remove = []
        ijs = itl.combinations( range(n), 2 )
        sets_u = []
        for i,j in ijs:
            set_ij = sets[i].union( sets[j] )
            if set_ij in sets and (set_ij not in sets_remove):
                sets_remove.append( set_ij )
        sets_u = oe.get_compl(sets, sets_remove)
        sets = sets_u

    def get_rigid_neighbors(self):

        for i in range()

