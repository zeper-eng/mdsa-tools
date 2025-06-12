# A brief introduction to the goal of this project
This was my BA/MA project which leveraged network theoretic concepts and machine learning techniques to perform a systems biology analysis of molecular dynamics simulations of a ribsomal subsystem we deemed the N2 nighborhood.

The labs current focus is on computational genetics expirements.
We modify various adjacent codon identites at the A-site and +1 (poised to enter the A site 5'-3').

# A note on setting up

First it we must download and install the github repository, this can be done easily by cloning the github repository as follows

```bash
git clone https://github.com/pardoluis123/workspace.git
cd workspace
```

There various necessary packages that we need in order to run all of the features as a part of this analysisThe workaround that makes the most sense for projects such as this one is to have a virtual environment that we can activate any time that we want to run  

I have included an environment.yml file which should make it easy to create and then activate the enviornment file any time you need to get going 

In order to create the virtual environment in your python virtual environment you should do as follows:

```bash
conda env create -f environment.yml #create enviornment (only needs to be done once)
conda activate

```

in order to install on another machine you should run the following line below. In all honesty for now marc I am just gonna have you go ahead and install my whole miniconda but, in theory it should work fine like this for you to start messing around. 

```bash
conda env create -f environment.yml
```

Alternatively, we could also set it up by pip installing the requirements file ourselves

# Specifically for Marc

Marc in our case I think the easiest way to go about this is to just source my miniconda install in your personal bashrc 


