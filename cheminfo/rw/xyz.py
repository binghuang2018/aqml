#!/usr/bin/env python

import numpy as np
import ase.io.extxyz as rx
from aqml import cheminfo as co
from ase import Atoms
import aqml.io2 as io2
import os, re

def read_xyz_simple(f, opt='z', icol=None, property_names=None, idx=None):
    """
    read geometry & property from a xyz file

    icol: if not None, choose the `icol entry of line 2 of input xyz
          file as the default property of the molecule,
          and the default property to be assigned is "HF"
    """
    assert os.path.exists(f)
    cs = open(f,'r').readlines()
    assert len(cs) > 0
    #print('cs=',cs)
    na = int(cs[0])

    props = {}
    c2 = cs[1].strip() ## e.g., "E=-100.2045 ALPHA=-3.45" or pure property list "-100.2 -3.45"
    if len(c2) > 0 and property_names:
        _props = {}
        if '#' in c2:
            try:
                sk, sv = c2.split('#')
                _props = dict(zip(sk.strip().split(), [eval(svi) for svi in sv.split()]))
            except:
                print(' ** no property found from 2nd line of xyz file')
        elif '=' in c2:
            for c2i in c2.split():
                k,sv = c2i.split('=')
                _props[k] = eval(sv)
        else:
            print(' ** no property found from 2nd line') #raise Exception(' unknown property format in 2-nd line')

        if ('a' in property_names) or ('all' in property_names):
            property_names = list(_props.keys())
        #print('f=',f, 'pns=', property_names, 'props=', _props )
        for p in property_names:
            if p not in _props:
                raise Exception('No value for property_name %s is found!'%p)
            props[p] = _props[p]

    _ats = []; coords = []; nheav = 0
    chgs = []; nmr = []; grads = []; cls = []
    for i in range(2,na+2):
        #print cs[i]
        csi = cs[i].strip().split()
        _si, sx, sy, sz = csi[:4]
        csia = csi[4:]
        if len(csia)>0:
            if 'chgs' in props:
                #chgs.append( eval(csia[props['chgs']]) )
                syi = csia[props['chgs']]
                yi = np.nan if syi.lower() == 'nan' else eval(syi)
                chgs.append(yi)
            if 'nmr' in props:
                syi = csia[props['nmr']]
                yi = np.nan if syi.lower() == 'nan' else eval(syi)
                nmr.append(yi)
            if 'cls' in props:
                syi = csia[props['cls']]
                yi = np.nan if syi.lower() == 'nan' else eval(syi)
                cls.append(yi)
            if 'grads' in props:
                grads.append( [ eval(csia[props['grads']+j]) for j in range(3) ] )
        try:
            _zi = co.chemical_symbols_lowercase.index(_si.lower())
        except:
            _zi = int(_si)
        _si = co.chemical_symbols[_zi]
        if _si not in ['H']: nheav += 1
        si = _zi if opt=='z' else _si
        _ats.append(si)
        coords.append( [ eval(_s) for _s in [sx,sy,sz] ] )
    if len(chgs) > 0:
        props['chgs'] = np.array(chgs)
    if len(nmr) > 0:
        props['nmr'] = np.array(nmr)
    if len(grads) > 0:
        props['grads'] = np.array(grads)
    if len(cls) > 0:
        props['cls'] = np.array(cls)
    return [na], _ats, coords, [nheav], props


def read_xyz(fileobj, property_names=None, idx=None):
    props = {}; zs = []; coords = []; nas=[]; nsheav=[]
    nm = len(re.findall('^\s*\d\d*$', open(fileobj).read(), re.MULTILINE))
    index = slice(0,nm)
    _ms = []
    for i,mi in enumerate(rx.read_xyz(fileobj, index=index, properties_parser=rx.key_val_str_to_dict_regex)):
        _ms.append(mi)
    if idx is not None:
        ms = [ _ms[im] for im in idx ]
    else:
        ms = _ms
    for mi in ms:
        #print('mi=', mi.info)
        nas.append(len(mi))
        nsheav.append( (mi.numbers>1).sum() )
        zs += list(mi.numbers); coords += list(mi.positions)
        #if i%1000 == 0: print('i=',i)
        if property_names:
            if ('a' in property_names) or ('all' in property_names):
                property_names = list(mi.info.keys())
            #print('pns=', property_names)
            for key in property_names: #mi.info.keys():
                #if key not in mi.info.keys():
                #    print('#ERROR: key absent!')
                #    print('i,key=',i,key, mi.info.keys())
                #    raise
                #print('props=',props)
                if key in props.keys():
                    props[key] += [ mi.info[key] ]
                else:
                    props[key] = [ mi.info[key] ]
            #print('props=',props)
    return np.array(nas,int), np.array(zs,int), np.array(coords), np.array(nsheav,int), props


def write_xyz(fileobj, images, comments=''):
    fclose = False
    if isinstance(fileobj, str):
        fileobj = open(fileobj, 'w')
        fclose = True

    if isinstance(images, (Atoms,tuple)):
        images = [images]
    elif isinstance(images, list):
        pass
    else:
        raise('#ERROR: input should be a list')

    if not isinstance(comments, list):
        comments = [comments]

    for i, image in enumerate(images):
        if isinstance(image, Atoms):
            symbols, positions = image.get_chemical_symbols(), image.positions
        elif isinstance(image, (tuple,list)):
            symbols, positions = image #.get_chemical_symbols()
        else:
            raise Exception('#ERROR: unknown input')
        na = len(symbols)
        fileobj.write('%d\n%s\n' % (na, comments[i]))
        for s, (x, y, z) in zip(symbols, positions):
            fileobj.write('%-2s %15.8f %15.8f %15.8f\n' % (s, x, y, z))

    if fclose: fileobj.close()


def write_xyz_simple(fileobj, image, props={}):
    fclose = False
    if isinstance(fileobj, str):
        fileobj = open(fileobj, 'w')
        fclose = True

    if isinstance(image, Atoms):
        symbols, positions = image.get_chemical_symbols(), image.positions
    elif isinstance(image, (tuple,list)):
        symbols, positions = image #.get_chemical_symbols()
    else:
        raise Exception('#ERROR: unknown input')

    na = len(symbols)
    np = len(props)
    sl2 = ''
    if np != 0:
        keys = props.keys()
        for k in keys:
            sl2 += '%s=%s '%(k,props[k])
    fileobj.write('%d\n%s\n' % (na, sl2.strip()))
    for s, (x, y, z) in zip(symbols, positions):
        fileobj.write('%-2s %15.8f %15.8f %15.8f\n' % (s, x, y, z))
    if fclose: fileobj.close()


if __name__ == "__main__":
    import os, sys

    fs = sys.argv[1:]
    nf = len(fs)
    for i,f in enumerate(fs):
        if i%20 == 0: print('now %d/%d'%(i+1,nf))
        _, ats, coords, _, props = read_xyz_simple(f, opt='s')
        write_xyz_simple(f, (ats,coords), props)


