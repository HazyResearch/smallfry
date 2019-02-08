from setuptools import find_packages, setup
from os.path import basename, splitext
from glob import glob

setup(name='smallfry',
      version='0.1',
      description='Code for smallfry.',
      packages=find_packages("src"),
      package_dir={"": "src"},
      py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
      url='https://github.com/HazyResearch/smallfry.git',
      author='Avner May / Jian Zhang',
      author_email='zjian@stanford.edu',
      license='Apache Version 2',
      install_requires = ['numpy',
                          'torch']
                          
      )