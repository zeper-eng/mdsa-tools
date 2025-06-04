# A brief introduction to the goal of this project
This was my BA/MA project which leveraged network theoretic concepts and machine learning techniques to perform a systems biology analysis of molecular dynamics simulations of a ribsomal subsystem we deemed the N2 nighborhood.

The labs current focus is on computational genetics expirements.
We modify various adjacent codon identites at the A-site and +1 (poised to enter the A site 5'-3').

# A note on setting up
In order to properly get work underway we leverage various python packages we need to load in our environment. In order to do this the most easily I have set up a bootsrrap file that can be easily run inside
of any bash environment! Below is how you would run the file

in order to install on another machine you should run the following line below. In all honesty for now marc I am just gonna have you go ahead and install my whole miniconda but, in theory it should work fine like this for you to start messing around. 

```bash
conda env create -f environment.yml
```

Alternatively, we could also set it up by pip installing the requirements file ourselves

# Specifically for Marc

Marc in our case I think the easiest way to go about this is to just source my miniconda install in your personal bashrc 


