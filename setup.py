# setup.py

from setuptools import setup, find_packages

setup(
    name='WarpNeRF',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'tyro',
        'viser',
        'warp-lang',
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu124',
        'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch',
    ],
    entry_points='''
        [console_scripts]
        wn=warpnerf.cli:main
    ''',
)
