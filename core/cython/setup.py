from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy 
import scipy

ext_module = Extension("kfpack",
     ["kfpack.pyx","Optimize.cpp","Moment.cpp","Likelihood.cpp"],
     libraries=['m','mkl_intel_lp64','mkl_intel_thread','mkl_core','iomp5','pthread','lbfgs'],
     language="c++",
     extra_compile_args=["-std=c++11","-xHost","-ipo","-O3","-qopenmp"],
     extra_link_args=["-std=c++11","-xHost","-ipo","-O3","-qopenmp"],
     #library_dirs=[cython_gsl.get_library_dir()],
     include_dirs=[numpy.get_include(), scipy.get_include()])

setup(
	name="kfpack",
    #include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module]
    )