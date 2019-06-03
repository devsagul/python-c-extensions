import numpy as np
from distutils.core import setup, Extension

setup(name = 'helloModule', version = '1.0',  \
   include_dirs = [np.get_include()], \
   ext_modules = [Extension('helloModule', ['hello.c'],  extra_compile_args = ["-O3"])], \
  )
