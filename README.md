# A brief introduction to the goal of this project
This was my BA/MA project which leveraged network theoretic concepts and machine learning techniques to perform a systems biology analysis of molecular dynamics simulations of a ribsomal subsystem we deemed the N2 nighborhood.

The labs current focus is on computational genetics expirements.
We modify various adjacent codon identites at the A-site and +1 (poised to enter the A site 5'-3')
in order to model the varying behaviors of the CAR interaction surface and how they correllate to varying
translation rate changes.

# Installing miniconda for the intelligent naive user

If you are unfamiliar with installing a version of miniconda yourself the following is mean to be a quick 
guide on how to go ahead and do it yourslef.

```bash
# Download Miniconda installer (Mac/Linux example)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Run the installer
bash Miniconda3-latest-MacOSX-x86_64.sh

# Follow prompts, then restart your terminal or run:
source ~/.bashrc   # or ~/.zshrc depending on shell

```
As with most python packages the expectation is that you have some level of familiarity in a computing environment but somequick notes on installation are

- The password prompts do not show your password on screen so make sure to type them in carefully 
- Restarting your terminal is equivalent to running the miniconda installer

When running the installer you will also see a prompt that looks like this at one point 

```bash
Do you wish the installer to initialize Miniconda3
by running conda init? [yes]
```

It is highly recommended that you just say yes so your miniconda gets added to your shell config (your startup env file)
and it should end up looking something like this inside of your actual setup file

```bash
# >>> conda initialize >>>
. /Users/username/miniconda3/etc/profile.d/conda.sh
conda activate base
# <<< conda initialize <<<
```

If for whatever reason you choose not to activate conda you will need to do this every single time

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate
```

Before being able to do any of the other steps.

# Forking the repository and making it your own
First it we must download and install the github repository, this can be done easily by cloning the github repository as follows

```bash
git clone https://github.com/pardoluis123/workspace.git
cd workspace
```

We still need to go ahead and create an environment that would include all of the various python packages we need in order to run our analysis.
 
The workaround that makes the most sense for projects such as this one is to have a virtual environment that we can activate any time that we want to run. 

I have included an environment.yml file which should make it easy to create and then activate the enviornment file any time you need to get going 

In order to create the virtual environment in your python virtual environment you should do as follows:

```bash
conda env create -f environment.yml #create enviornment (only needs to be done once)
conda activate SBTA_ENV #This is what you would need to run the environment
```

Alternatively, you could set it up as a part of your bashrc file such that it always activates on setup you would just include this line after opening your bashrc
```bash
conda activate SBTA_ENV
```

# Specifically for Marc

Marc in our case I think the easiest way to go about this is to just source my miniconda install in your personal bashrc 


