# our setup file so we can start formalizing this into a formal package
from setuptools import setup, find_packages

setup(
    name='TBD_v_v',                        # Name of the package
    version='0.1.0',                         # Version number
    description="The Weir Labs H-bond Systems Analyses modules!",
    author='Luis Perez',                     
    author_email='lperez@wesleyan.edu',   
    packages=find_packages(),                # Automatically finds all packages with __init__.py
    include_package_data=True,               # If you have data files inside packages
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'mdtraj',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',                 # Or whatever version you’re targeting
)