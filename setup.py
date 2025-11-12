from setuptools import setup

setup(
   name='opt_frac',
   version='0.1',
   description='Optimal Adaptive Fractionation Schedules',
   author='Anqi Fu',
   author_email='fua@mskcc.org',
   packages=['opt_frac'],
   install_requires=['numpy', 'matplotlib', 'cvxpy'],
)