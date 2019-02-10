import sys
from numpy.distutils.core import Extension, setup

from mkldiscover import mkl_exists

__author__ = "Anders S. Christensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Anders S. Christensen et al. (2016) https://github.com/qmlcode/qml"]
__license__ = "MIT"
__version__ = "0.4.0.12"
__maintainer__ = "Anders S. Christensen"
__email__ = "andersbiceps@gmail.com"
__status__ = "Beta"
__description__ = "Quantum Machine Learning"
__url__ = "https://github.com/qmlcode/qml"


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
fs = [ [ 'cml/atomdb.f90', 'cml/utils.f90', 'cml/fslatm.f90' ] ]
fs1 = ['cml/fmorse.f90', 'cml/frepresentations.f90', 'cml/fdistance.f90', 'cml/fkernels.f90'] # 'cml/fvcp.f90'],
for f1 in fs1:
    fs += [ [f1] ]
fs += [ ['cml/amon/famon.f90'], ['cml/amon/famoneib.f90'] ]
for fsi in fs:
  name = fsi[-1][:-4].split('/')[-1]
  ext_mod = Extension(name = name,
                          sources = fsi,
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])
  ext_modules.append( ext_mod )


def setup_pepytools():

    setup(

        name="cml",
        packages=['cml'],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        url = __url__,

        # set up package contents

        ext_package = 'cml',
        ext_modules = ext_modules
)

if __name__ == '__main__':

    setup_pepytools()
