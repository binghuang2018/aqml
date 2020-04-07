

from aqml.cheminfo.core import *

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from openeye import oechem
from aqml.cheminfo.oechem.oechem import *

from IPython.display import SVG, display, HTML
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import DrawingOptions

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import tempfile as tpf

import imolecule as imol

#DrawingOptions.atomLabelFontSize = 105
#DrawingOptions.dotsPerAngstrom = 2000
#DrawingOptions.bondLineWidth = 3.0

T,F = True,False

class draw_rdkit(object):

    def __init__(self): #, objs, istereo=False, woH=True, wo3d=False):
        #self.objs = objs
        #self.istereo = istereo
        #self.woH = woH
        #self.wo3d = wo3d
        #self.wo3d = wo3d
        return

    @staticmethod # @classmethod
    def str2mol(objs, istereo=False, woH=True, wo3d=False):
        mols = []
        labs = []
        if not isinstance(objs, (list,tuple)): objs=[objs]
        for obj in objs:
            if type(obj) is str:
                if os.path.exists(obj):
                    mol = Chem.MolFromMolFile(obj, removeHs=woH)
                    if wo3d:
                        string = Chem.MolToSmiles(mol, isomericSmiles=istereo)
                        mol = Chem.MolFromSmiles(string)
                else:
                    mol = Chem.MolFromSmiles(obj)
                labs.append( obj )
            elif type(obj) is Chem.rdchem.Mol:
                mol = obj
                labs.append( Chem.MolToSmiles(mol) )
            else:
                raise Exception('#ERROR: input not supported')
            mols.append(mol)
        return mols, labs

    @staticmethod # @classmethod
    def viewm0(strings, size=(200,200)):
        return Draw.MolsToImage(draw_rdkit.str2mol(strings), includeAtomNumbers=True, subImgSize=size)

    @staticmethod # @classmethod
    def viewm(string,molSize=(450,150),woH=True,wo3d=False,kekulize=True,atsh=[],bsh=[],colors=[]):
        """
        vars
        atsh : atoms to be highlighted (list)
        bsh: bonds to be highlighted (list)
        woH: without H atoms displayed
        wo3d: without 3d coordinates embeded
        """
        mol = draw_rdkit.str2mol(string,woH=woH,wo3d=wo3d)[0][0]
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])

        opts = drawer.drawOptions()
        for i in range(mol.GetNumAtoms()):
            opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)

        opts.legendFontSize=16

        _colors={ (1,0,0), (0,1,0), (0,0,1), (1,0,1)}

        if len(colors) > 0:
            drawer.DrawMolecule(mc,highlightAtoms=atsh,highlightBonds=bsh,highlightAtomColors=colors)
        else:
            drawer.DrawMolecule(mc,highlightAtoms=atsh,highlightBonds=bsh)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # It seems that the svg renderer used doesn't quite hit the spec.
        # Here are some fixes to make it work in the notebook, although I think
        # the underlying issue needs to be resolved at the generation step
        return SVG( svg.replace('svg:','') )

    @staticmethod # @classmethod
    def viewms(objs,wo3d=F,nmr=8,molSize=(100,80),wlgd=T,wlgd_text='smiles', labels=None, \
               wia=T,woH=T,kekulize=T,atsh=[],bsh=[],colors=[],filename=None):
        """
        vars
        atsh : atoms to be highlighted (list)
        """
        mols, strings = draw_rdkit.str2mol(objs,woH=woH)
        mcs = []
        _labels = []
        nat = 0
        nm = len(mols)
        smis = []
        for mol in mols:
            labels_i = [ chemical_symbols[ai.GetAtomicNum()]+'%d'%(ai.GetIdx()+1) for ai in mol.GetAtoms() ]
            nat += len(labels_i)
            _labels += labels_i
            mc = Chem.Mol(mol.ToBinary())
            if kekulize:
                try:
                    Chem.Kekulize(mc)
                except:
                    mc = Chem.Mol(mol.ToBinary())
            c2d = F
            if mc.GetNumConformers():
                if wo3d:
                    c2d = T
            else:
                c2d = T
            if c2d: rdDepictor.Compute2DCoords(mc)
            smis.append( Chem.MolToSmiles(mc) )
            mcs.append(mc)
        #print('labels=',labels)

        molsPerRow = nmr #5
        subImgSize= molSize

        nRows = len(mols) // molsPerRow
        if len(mols) % molsPerRow:
            nRows += 1
        fsize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])

        dw, dh = 0, 0
        drawer = rdMolDraw2D.MolDraw2DSVG(fsize[0]+dw,fsize[1]+dh, subImgSize[0], subImgSize[1])

        opts = drawer.drawOptions()
        ## [bad news] opts seems to be able to accept only the first `na1 labels (where na1=len(mols[0]))
        #for i in range(nat):
        #    opts.atomLabels[i] = labels[i]

        if labels:
            labs = labels
        else:
            if wlgd_text in ['id']:
                labs = [ str(i+1) for i in range(nm) ] #rows*cols) ]
            elif wlgd_text in ['smiles',]:
                labs = smis
            elif wlgd_text in ['file','filename']:
                assert len(fs) > 1, '#ERROR: input objs are not files!!'
                labs = [ fi.split('/')[-1] for fi in fs ]
            else:
                raise Exception('invalid `wlgd_text')

        drawer.DrawMolecules(mcs, legends=labs)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # It seems that the svg renderer used doesn't quite hit the spec.
        # Here are some fixes to make it work in the notebook, although I think
        # the underlying issue needs to be resolved at the generation step

        if filename is not None:
            open(filename+'.svg','w').write(svg)
            try:
                import cairosvg
                cairosvg.svg2pdf(url='%s.svg'%filename, write_to='%s.pdf'%filename)
            except:
                raise Exception('#ERROR: svg to pdf failed, as cairosvg seems to be not installed?')
        return SVG( svg.replace('svg:','') )


class draw_oechem(object):


    def __init__(self): #, objs, istereo=False, woH=True, wo3d=False):
        #self.objs = objs
        #self.istereo = istereo
        #self.woH = woH
        #self.wo3d = wo3d
        return

    @staticmethod # @classmethod
    def str2mol(objs, istereo=False, woH=True, wo3d=False):
        mols = []
        smis = []
        fs = []
        if type(objs) is str: objs=[objs]
        for obj in objs:
            if type(obj) is str:
                if os.path.exists(obj):
                    fs.append(fs)
                    mol = sdf2oem(obj) #Chem.MolFromMolFile(obj, removeHs=woH)
                    #print('++mol=',mol)
                    #assert iok, '#ERROR: sdf2oem() failed??'
                    smi = oem2can(mol) # Chem.MolToSmiles(mol, isomericSmiles=istereo)
                    if wo3d:
                        iok, mol = smi2oem(smi)
                        assert iok, '#ERROR: smi2oem() failed?'
                else:
                    smi = obj
                    iok, mol = smi2oem(obj)
                    assert iok, '#ERROR: smi2oem() failed?'

            elif type(obj) is oechem.OEGraphMol:
                mol = obj
                smi = oem2can(mol)
            else:
                raise Exception('#ERROR: input not supported')
            mols.append(mol)
            smis.append( obj )
        return mols, smis, fs


    @staticmethod # @classmethod
    def viewms(objs, nmr=8, molSize=[100,80], woH=T, wo3d=F, wlgd_text='smiles',\
               wlgd=T, wia=F, filename=None, tdir='/tmp/'):

        """
        vars
        ====================
        nmr: num of mol per row
        woH: without Hydrogens displayed, T/F
        wo3d: without 3d coordinates? T/F
        wlgd: with legend displayed? T/F
        wlgd_text: 'smiles' or 'idx'
        wia: with idx of atom shown? T/F
        """
        mols, smis, fs = draw_oechem.str2mol(objs,woH=woH,wo3d=wo3d)
        nm = len(mols)
        cols = nmr #4 if nm < nmr else nmr # num mol per row
        rows = int(nm/cols) if nm%cols==0 else int(nm/cols)+1

        image = oedepict.OEImage(cols*molSize[0], rows*molSize[1])

        grid = oedepict.OEImageGrid(image, rows, cols)

        opts = oedepict.OE2DMolDisplayOptions(grid.GetCellWidth(), \
                           grid.GetCellHeight(), oedepict.OEScale_AutoScale)

        if wlgd_text in ['id']:
            labs = [ str(i+1) for i in range(nm) ] #rows*cols) ]
        elif wlgd_text in ['smiles',]:
            labs = smis
        elif wlgd_text in ['file','filename']:
            assert len(fs) > 1, '#ERROR: input objs are not files!!'
            labs = [ fi.split('/')[-1] for fi in fs ]
        else:
            raise Exception('invalid `wlgd_text')

        for i in range(nm):
            mol = mols[i]
            #print('mol=',mol)
            oedepict.OEPrepareDepiction(mol)

            irow, icol = int(i/cols)+1, i%cols+1
            cell = grid.GetCell(irow,icol)

            if wlgd:
                mol.SetTitle(labs[i])

            opts.SetAtomVisibilityFunctor(oechem.OEIsTrueAtom())  # explicitly set the default

            font = oedepict.OEFont()
            font.SetSize(16)
            font.SetStyle(oedepict.OEFontStyle_Normal) #_Bold)
            font.SetColor(oechem.OEDarkGreen)
            opts.SetAtomPropLabelFont(font)

            if wia:
                opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomIdx())

            opts.SetAtomLabelFont(font)
            #opts.SetAtomLabelFontScale(1.5)

            opts.SetAtomPropLabelFont(font) # atom idx
            opts.SetAtomPropLabelFontScale(2) # atom idx

            opts.SetTitleLocation(oedepict.OETitleLocation_Bottom)
            opts.SetTitleHeight(10.0)

            disp = oedepict.OE2DMolDisplay(mol, opts)
            clearbackground = True
            oedepict.OERenderMolecule(cell, disp, not clearbackground)

        save = F # save as pdf
        if filename is not None:
            tf = filename+'.svg'
            save = T
        else:
            tf = tpf.NamedTemporaryFile(dir=tdir).name + '.svg'
        oedepict.OEWriteImage(tf, image)

        if save:
            try:
                import cairosvg
                cairosvg.svg2pdf(url='%s.svg'%filename, write_to='%s.pdf'%filename)
            except:
                raise Exception('#ERROR: svg to pdf failed, as cairosvg seems to be not installed?')

        return SVG(tf)


class draw_imol(object):
    """
    draw 3d molecule by imolecule
    """

    def __init__(self):
        return

    @staticmethod
    def html(ms,fmt,size=(240,240),nmr=6):
        renders = (imol.draw(m,fmt,size=size,display_html=False) for m in ms)
        columns = ('<div class="col-xs-8 col-sm-4">{}</div>'.format(r) for r in renders)
        return HTML('<div class="row">{}</div>'.format("".join(columns)))

    @staticmethod
    def viewms(objs,molSize=(160,160),nmr=6):

        if not isinstance(objs, (tuple,list)):
            objs = [objs]

        # convert to a file format recognizable by imolecule, i.e., sdf
        ms = []
        for obj in objs:
            typ = type(obj)
            if typ is Chem.rdchem.Mol:
                ms.append( Chem.MolToMolBlock(obj) )
            elif typ is str: #oechem.OEGraphMol:
                if os.path.exists(obj):
                    assert obj[-3:] in ['sdf','mol'], '#ERROR: file format not supported'
                    ms.append( ''.join(open(obj).readlines()) )
                else:
                    print(' ** assume smiles string ')
                    ms.append( Chem.MolToMolBlock( Chem.MolFromSmiles(obj) ) )
        if len(ms) == 1:
            return [ imol.draw(ms[0],'sdf',size=molSize) ]
        else:
            return draw_imol.html(ms,'sdf',size=molSize,nmr=nmr)
            #raise Exception('not know what to do next!')

import ase
import ase.io as aio
from ase.visualize import view
from tempfile import NamedTemporaryFile as ntf

class draw_ase(object):

    def __init__(self):
        return

    @staticmethod
    def viewms(obj):
        """ return the html representation the atoms object as string """
        renders = []
        ms = []
        if isinstance(obj, ase.Atoms):
            ms.append(obj)
        elif isinstance(obj, (tuple,list)):
            for oi in obj:
                ms.append( aio.read(f) )
        else:
            raise Exception('##')
        for atoms in ms:
            with ntf('r+', suffix='.html') as fo:
                atoms.write(fo.name, format='html')
                fo.seek(0)
                renders.append( fo.read() )
        columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
        return HTML('<div class="row">{}</div>'.format("".join(columns)))

