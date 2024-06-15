from setuptools import setup, find_packages

setup(
  name = 'lion-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.1',
  license='MIT',
  description = 'Lion Optimizer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/lion-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
