# setup.py

from setuptools import setup, find_packages

setup(
    name='WarpNeRF',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tyro',
    ],
    entry_points='''
        [console_scripts]
        wn=warpnerf.cli:main
    ''',
)
