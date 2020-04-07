import os, sys
from numpy.distutils.core import Extension, setup

__author__ = "Bing Huang"
__copyright__ = "Copyright 2020"
__credits__ = ["Bing Huang, et al. (2020) https://github.com/binghuang2018/aqml"]
__license__ = "MIT"
__version__ = "0.1.0.0"
__maintainer__ = "Bing Huang"
__email__ = "hbdft2008@gmail.com"
__status__ = "Beta"
__description__ = "Amons-based Quantum Machine Learning"
__url__ = "https://github.com/binghuang2018/aqml"


FORTRAN = "f90"

# GNU (default)
COMPILER_FLAGS = ["-O3", "-fopenmp", "-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
LINKER_FLAGS = ["-lgomp"]
MATH_LINKER_FLAGS = ["-lblas", "-llapack"]

# UNCOMMENT TO FORCE LINKING TO MKL with GNU compilers:
#if mkl_exists(verbose=True):
#    LINKER_FLAGS = ["-lgomp", " -lpthread", "-lm", "-ldl"]
#    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

# For clang without OpenMP: (i.e. most Apple/mac system)
if sys.platform == "darwin" and all(["gnu" not in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-O3", "-m64", "-march=native", "-fPIC"]
    LINKER_FLAGS = []
    MATH_LINKER_FLAGS = ["-lblas", "-llapack"]


# Intel
if any(["intelem" in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-xHost", "-O3", "-axAVX", "-qopenmp"]
    LINKER_FLAGS = ["-liomp5", " -lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

ext_modules = []
fs = [ ['cml/atomdb.f90', 'cml/representation/utils.f90', 'cml/representation/fslatm.f90'], ]
fs += [ ['cml/representation/fcore.f90'] ]
fs1 = ['cml/fmorse.f90', 'cml/fdistance.f90', 'cml/fkernels.f90'] # 'cml/fvcp.f90'],
fs1 += [ 'cml/amon/famon.f90', 'cml/amon/famoneib.f90']
fs += [ [f1] for f1 in fs1 ]
for fsi in fs:
  fi = fsi[-1]
  name = '.'.join( fi[:-4].split('/')[1:] )
  #name = re.sub(name, '\/', '.')
  inc = []
  #if 'fslatm.f90' in fi:
  #  inc = ['atomdb', 'representation.utils']
  ext_mod = Extension(name = name,
                          sources = fsi,
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          include_dirs=inc,
                          language = FORTRAN,
                          f2py_options=['--quiet'])
  ext_modules.append( ext_mod )


py_modules = []
for root, dirs, files in os.walk("cml"):
    for f in files:
        #if f[-3:] == '.py':
        f1 = root+'/'+f
        if ('__init__.py' not in f1) and f1.endswith(".py"):
            f2 = '.'.join( f1[:-3].split('/') ) #[1:] )
            py_modules.append(f2) # print(os.path.join(root, f))
#print('py_modules=',py_modules)
#sys.exit(2)

def setup_pepytools():

    setup(

        name="cml",
        packages=['cml'],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Linux/Mac',
        description = __description__,
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        url = __url__,

        # set up package contents

        ext_package = 'cml',
        py_modules = py_modules,
        ext_modules = ext_modules
)

if __name__ == '__main__':

    setup_pepytools()

