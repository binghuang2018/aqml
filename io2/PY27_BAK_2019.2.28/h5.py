
import numpy as np
import os, h5py

def load(f):
    assert os.path.exists(f)
    obj = h5py.File(f,'r')
    dic = {}
    for key in obj.keys():
        dic[key] = obj[key].value
    return dic

def save(f,dic):
    type_str = h5py.special_dtype(vlen=str)
    type_float64 = h5py.special_dtype(vlen=np.dtype('float64'))
    type_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
    obj = h5py.File(f, 'w')
    for key in dic.keys():
        d = np.array(dic[key])
        size = d.shape
        dt = { np.int64 : type_int64, float : type_float64, np.float64 : type_float64, str : type_str }[ type(d.item(0)) ]
        #obj.create_dataset(key, size, dtype=dt, data=d, chunks=True)
        obj.create_dataset(key, data=d, chunks=True)

