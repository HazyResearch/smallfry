from setuptools import setup

setup(name='smallfry',
      version='0.1',
      description='Insanely compressed word embeddings',
      url='https://github.com/HazyResearch/small-fry',
      author='tginart et al.',
      author_email='tginart@stanford.edu',
      license='MIT',
      packages=['smallfry'],
      scripts=['smallfry/smallfry.py'],
      install_requires=['scipy','marisa_trie', 'sklearn', 'argh','numpy'],
      zip_safe=False)
