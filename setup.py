# setup.py

from setuptools import setup, find_packages

setup(
    name='WarpNeRF',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'lightglue @ git+https://github.com/cvg/LightGlue.git#egg=lightglue',
        'msgpack',
        'pycolmap',
        'tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch',
        'torch',
        'torchvision',
        'torchaudio',
        'tyro',
        'viser',
        'warp-lang==1.5.0',
        'websockets',
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu124',
    ],
    entry_points='''
        [console_scripts]
        wn=warpnerf.cli:main
    ''',
)
