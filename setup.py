from setuptools import setup, find_packages
from sks_version import __version__
    
setup(
    name="skspeech",
<<<<<<< Updated upstream
    version="0.0.2",
    url="https://github.com/compi1234/scikit-speech",
=======
    version=__version__,
    url="",
>>>>>>> Stashed changes

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = ['skspeech'],
    # add skspeech_version to the required install modules
    py_modules = ['skspeech','sks_version'],
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=['numpy','pandas'],

    classifiers=['Development Status: Pre-Alpha, Unstable',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6'],
                 
    include_package_data=True

)