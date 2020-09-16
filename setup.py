from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

import os
import torch

setup(name='fastconv_nu',
      version='0.0.0',
      ext_modules=[
                   CUDAExtension(name='nu_conv',
                                 sources=[
                                    'src/nu_conv_cuda.cpp',
                                    'src/nu_conv_cuda_kernel.cu'],
#                                  extra_compile_args={'cxx': ['-g', '-O3'],
#                                                      # 'nvcc': ['-arch=sm_70', '-O3']}
#                                                      # 'nvcc': ['-arch=sm_52', '-O3']}
#                                                      'nvcc': ['-arch=sm_61', '-O3']}
#                                                      # 'nvcc': ['-arch=sm_52', '-gencode=arch=compute_52,code=sm_52', '-O3']}
                                ),
                  ],
     cmdclass={'build_ext': BuildExtension})


if __name__ == "__main__":
    print()
    print("#########################")
    print("Test if compilation is ok")
    print("#########################")
    print()
    import nu_conv