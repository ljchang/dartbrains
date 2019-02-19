# !/usr/bin/env python

from setuptools import setup, find_packages

version = {}
with open("dartbrains/version.py") as f:
    exec(f.read(), version)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='dartbrains',
    install_requires = requirements,
    packages = find_packages(exclude=['dartbrains/tests']),
    version = version['__version__'],
    description='Toolbox for Dartmouth Neuroimaging Analysis Course',
    author='Luke Chang',
    license = 'LICENSE.txt',
    author_email='luke.j.chang@dartmouth.edu',
    url='https://github.com/ljchang/dartbrains',
    keywords=['neuroimaging', 'analysis', 'package', ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development',
    ],
    **extra_setuptools_args
)
