from setuptools import setup, find_packages

setup(
    name="skspeech",
    version="0.0.1",
    url="https://github.com/compi1234/scikit-speech",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = ['skspeech'],
    py_modules = [],
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=['numpy','pandas'],

    classifiers=['Development Status: Pre-Alpha, Unstable',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6'],
                 
    include_package_data=True

)