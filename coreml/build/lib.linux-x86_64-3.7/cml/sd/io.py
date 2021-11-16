





import numpy as np
import pickle as pkl
import os



class io(object):

    """

      A unified module for reading/writing structured
      arrays from/to one of the allowed formats, including
      ['npz', 'pkl', 'h5']

    """

    def __init__(self, d=None, f=None):

        self.d = d
        self.f = f

    @staticmethod
    def load(f):
        if f.endswith('.npz'):
            _ = np.load(f)
            d = { _[k] for k in _.files } ## ! 
        #elif f.endswith('.npy'):
        #    d = np.load(f)
        elif f.endswith('.pkl'):
            with open(f, 'rb') as fid:
                d = pkl.load(fid)
        elif f.endswith('.h5'):
            try:
                import cml.sd as dd
            except:
                raise Exception('package `deepdish` not found. Install it through `pip install deepdish`!')
            d = dd.io.load(f)
        else:
            raise Exception(' file format not recognized!')
        return d


    @staticmethod
    def save(f, d, ow=False):
        if os.path.exists(f) and (not ow):
            raise Exception('File already exists in cwd. To overwrite it, set `ow=True`')
        if f.endswith('.npz'):
            np.savez(f, **d)
        elif f.endswith('.pkl'):
            with open(f, 'wb') as fid:
                pkl.dump(d, fid)
        else:
            raise Exception('File format not supported')

