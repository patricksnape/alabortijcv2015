from setuptools import setup, find_packages
import numpy as np

include_dirs = [np.get_include()]


setup(name='alabortijcv2015',
      version='0.0.1',
      description='Repository containing the code of the paper: '
                  'Advances on Gradient Descent Algorithms for AAMs Fitting',
      author='Joan Alabort-i-Medina',
      author_email='joan.alabort@gmail.com',
      include_dirs=include_dirs,
      packages=find_packages())
