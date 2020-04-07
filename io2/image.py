#!/usr/bin/env python

import PIL as pil
import sys
from IPython.display import Image, display_png


class ImageObj(object):

    def __init__(self, fin):
        self.f = fin
        self.image = pil.Image.open(fin)

    def save(self, fmt='png'):
        self.im.save(self.f[:-4]+fmt)

    def show(self):
        display_png(Image(self.f))








if __name__ == "__main__":

    import os, sys

    fs = sys.argv[1:]
    for f in fs:
        obj = ImageObj(f)
        obj.save(fmt='png')


