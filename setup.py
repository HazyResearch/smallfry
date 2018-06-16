import setuptools

setuptools.setup(name='smallfry',
      version='0.1',
      description='Insanely compressed word embeddings',
      url='https://github.com/HazyResearch/small-fry',
      author='tginart et al.',
      author_email='tginart@stanford.edu',
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      packages=setuptools.find_packages())
