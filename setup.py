from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

import os
import torch

setup(name='fastconv_nu',
      version='1.0.0',
      ext_modules=[
                   CUDAExtension(name='custom_conv',
                                 sources=[
                                    'src/nu_conv_cuda.cpp',
                                    'src/nu_conv_cuda_kernel.cu'],
                                 extra_compile_args={'cxx': ['-std=c++14'], 'nvcc': []}
                                ),
                  ],
     cmdclass={'build_ext': BuildExtension})


if __name__ == "__main__":
    print()
    print("#########################")
    print("Test if compilation is ok")
    print("#########################")
    print()
    import custom_conv