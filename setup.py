# our setup file so we can start formalizing this into a formal package
from setuptools import setup, find_packages

setup(
    name='TBD_v_v',                    
    version='0.1.0',                         
    description="The Weir Labs H-bond Systems Analyses modules!",
    author='Luis Perez',                     
    author_email='lperez@wesleyan.edu',   
    packages=find_packages(),                
    include_package_data=True,               
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'mdtraj==1.10.3',
        'umap-learn ',
        'python-circos'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',                
)