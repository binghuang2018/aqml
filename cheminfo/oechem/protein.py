

from openeye import oechem

T, F = True, False


class protein(object):

    def __init__(self, ipf, keepAlts=F, verbose=T):
        self.ipf = ipf
        self.keepAlts = keepAlts
        self.verbose = verbose

        flavor = oechem.OEIFlavor_PDB_Default
        ims = oechem.oemolistream()
        ims.SetFlavor(oechem.OEFormat_PDB, flavor)

        if not ims.open(self.ipf):
            oechem.OEThrow.Fatal("Unable to open %s for reading." %self.ipf)

        if not oechem.OEIs3DFormat(ims.GetFormat()):
            oechem.OEThrow.Fatal("%s is not in a 3D format." %self.ipf)

        iftp = oechem.OEGetFileType(oechem.OEGetFileExtension(self.ipf))
        if (iftp == oechem.OEFormat_PDB) and not self.keepAlts:
            oechem.OEThrow.Verbose("Default processing of alt locations (keep just 'A' and ' ').")

        inmol = oechem.OEGraphMol()
        if not oechem.OEReadMolecule(ims, inmol):
            oechem.OEThrow.Fatal("Unable to read %s." % self.ipf)

        ims.close()

        if (inmol.NumAtoms() == 0):
            oechem.OEThrow.Fatal("Input molecule %s contains no atoms." % self.ipf)

        if inmol.GetTitle() == "":
            inmol.SetTitle(ipf[:-4])

        oechem.OEThrow.Verbose("Processing %s." % inmol.GetTitle())
        if not oechem.OEHasResidues(inmol):
            oechem.OEPerceiveResidues(inmol, oechem.OEPreserveResInfo_All)

        self.inmol = inmol
        self.mol = inmol.CreateCopy()


    def split(self):

        lig = oechem.OEGraphMol()
        prot = oechem.OEGraphMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        if oechem.OESplitMolComplex(lig, prot, wat, other, self.inmol):
            # work with the output molecules lig, prot, ...
            return [prot,lig]
        else:
            raise Exception('failure to detect prot & lig')


    def remove_water_and_ions(self):
        """
        This function remove waters and ions molecules
        from the input system
        Parameters:
        ----------
        in_system : oechem.OEMol
            The bio-molecular system to clean
        opt: python dictionary
            The system option
        Output:
        -------
        clean_system : oechem.OEMol
            The cleaned system
        """
        # Copy the input system
        system = self.inmol.CreateCopy()

        # Create a bit vector mask
        bv = oechem.OEBitVector(system.GetMaxAtomIdx())
        bv.NegateBits()

        # Create a Hierarchical View of the protein system
        hv = oechem.OEHierView(system, oechem.OEAssumption_BondedResidue +
                               oechem.OEAssumption_ResPerceived)

        # Looping over the system residues
        for chain in hv.GetChains():
            for frag in chain.GetFragments():
                for hres in frag.GetResidues():
                    res = hres.GetOEResidue()

                    # Check if a residue is a mono atomic ion
                    natoms = 0
                    for at in hres.GetAtoms():
                        natoms += 1

                    # Set the atom bit mask off
                    if oechem.OEGetResidueIndex(res) == oechem.OEResidueIndex_HOH or natoms == 1:
                        # Set Bit mask
                        atms = hres.GetAtoms()
                        for at in atms:
                            bv.SetBitOff(at.GetIdx())

        # Extract the system without waters or ions
        pred = oechem.OEAtomIdxSelected(bv)
        clean_system = oechem.OEMol()
        oechem.OESubsetMol(clean_system, system, pred)
        self.mol = clean_system


    def addh(self, altProcess='occupancy', processName='fullsearch', \
            ihopt=T, standardize=T, badclash=0.4, flipbias=1.0, maxStates=20):#7):

        imol = self.mol.CreateCopy()

        wp = oechem.OEPlaceHydrogensWaterProcessing_Ignore
        if processName == 'fullsearch':
            wp = oechem.OEPlaceHydrogensWaterProcessing_FullSearch
        elif processName == 'focused':
            wp = oechem.OEPlaceHydrogensWaterProcessing_Focused

        keepAlts = (altProcess != "a")
        highestOcc = (altProcess == "occupancy")
        compareAlts = (altProcess == "compare")
        print('#1')
        if highestOcc or compareAlts:
            alf = oechem.OEAltLocationFactory(imol)
            if alf.GetGroupCount() != 0:
                if highestOcc:
                    oechem.OEThrow.Verbose("Dropping alternate locations from input.")
                    alf.MakePrimaryAltMol(imol)
                elif compareAlts:
                    oechem.OEThrow.Verbose("Fixing alternate location issues.")
                    imol = alf.GetSourceMol()
        omol = imol
        print('#2')
        oechem.OEThrow.Verbose("Adding hydrogens to complex.")

        hopt = oechem.OEPlaceHydrogensOptions()
        if ihopt:
            hopt.SetAltsMustBeCompatible(compareAlts)
            hopt.SetStandardizeBondLen(standardize)
            hopt.SetWaterProcessing(wp)
            hopt.SetBadClashOverlapDistance(badclash)
            hopt.SetFlipBiasScale(flipbias)
            hopt.SetMaxSubstateCutoff(maxStates)

        if self.verbose:
            print('#3')
            details = oechem.OEPlaceHydrogensDetails()
            if not oechem.OEPlaceHydrogens(omol, details, hopt):
                oechem.OEThrow.Fatal("Unable to place hydrogens and get details on %s." % self.inmol.GetTitle())
            oechem.OEThrow.Verbose(details.Describe())
        else:
            if not oechem.OEPlaceHydrogens(omol, hopt):
                oechem.OEThrow.Fatal("Unable to place hydrogens on %s." % self.inmol.GetTitle())
        self.mol = omol


    def write(self, mol, opf):
        oms1 = oechem.oemolostream()
        if not oms1.open(opf):
            oechem.OEThrow.Fatal("Unable to open %s for writing." % opf)
        oechem.OEWriteMolecule(oms1, mol)


