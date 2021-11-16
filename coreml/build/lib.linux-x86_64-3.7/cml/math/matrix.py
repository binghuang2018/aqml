
import numpy as np
from numpy import dot, zeros
from numpy.linalg import matrix_rank as mr


class Matrix(object):

    def __init__(self, M):
        self.M = M

    @property
    def rank(self):
        if not hasattr(self, '_rank'):
            self._rank = mr(self.M)
        return self._rank

    @property
    def idx(self):

        if not hasattr(self, '_idx'):

            nr, nc = self.M.shape
            r = self.rank
            idxc = [0]
            for i in range(1,nc):
                if mr(self.M[:,idxc+[i]]) > mr(self.M[:,idxc]):
                    idxc.append(i)

            assert len(idxc) == self.rank, '#ERROR: rank != len(idxc)??'
            self._idx = idxc

        return self._idx

