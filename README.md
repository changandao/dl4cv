# This repositary contains the exersice from the course Introduction to Deep Learning in WS 2018/2019

## Exercise 1
1. Linear classifiers  
2. Two-Layer NN

## Exercise 2
Build my own DL library:  
1. conected layer  
2. Bacth normalization  layer
3. Drop out layer

## Exercise 3
1. Classification on CIFAR-10 Dataset  
2. Semantic segmentation based on FCN using transfering learning

## Exercise 4
Some intersting application exersices:  
1. Fatial keypints detection  
2. RNN

%% Acknnowledgements are on TUM 
# Technical University Munich - WS 2018/19

1. Python Setup
2. PyTorch Installation
3. Exercise Download

## 1. Python Setup

Prerequisites:
- Unix system (Linux or MacOS)
- Python version 3
- Terminal (e.g. iTerm2 for MacOS)
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text)

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.

If you are using Windows, the procedure might slightly vary and you will have to Google for the details. A fellow student of yours compiled this (https://gitlab.lrz.de/yuesong.shen/DL4CV-win) very detailed Windows tutorial for a previous course. Please keep in mind, that we will not offer any kind of support for its content.

To avoid issues with different versions of Python and Python packages we recommend to always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*.

In this README we provide you with a short tutorial on how to use and setup a *virtuelenv* environment. To this end, install or upgrade *virtualenv*. There are several ways depending on your OS. At the end of the day, we want

`which virtualenv`

to point to the installed location.

On Ubuntu, you can use:

`apt-get install python-virtualenv`

Also, installing with pip should work (the *virtualenv* executable should be added to your search path automatically):

`pip3 install virtualenv`

Once *virtualenv* is successfully installed, go to the root directory of the i2dl repository (where this README.md is located) and execute:

`virtualenv -p python3 --no-site-packages .venv`

Basically, this installs a sandboxed Python in the directory `.venv`. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this *virtualenv* in a shell you have to first
activate it by calling:

`source .venv/bin/activate`

To test whether your *virtualenv* activation has worked, call:

`which python`

This should now point to `.venv/bin/python`.

From now on we assume that that you have activated your virtual environment.

Installing required packages:
We have made it easy for you to get started, just call from the i2dl root directory:

`pip3 install -r requirements.txt`

The exercises are guided via Jupyter Notebooks (files ending with `*.ipynb`). In order to open a notebook dedicate a separate shell to run a Jupyter Notebook server in the i2dl root directory by executing:

`jupyter notebook`

A browser window which depicts the file structure of directory should open (we tested this with Chrome). From here you can select an exercise directory and one of its exercise notebooks!


## 2. PyTorch installation

In exercise 3 we will introduce the *PyTorch* deep learning framework which provides a research oriented interface with a dynamic computation graph and many predefined, learning-specific helper functions.

Unfortunately, the installation depends on the individual system configuration (OS, Python version and CUDA version) and therefore is not possible with the usual `requirements.txt` file.

Follow the *Get Started* section on the official PyTorch [website](http://pytorch.org/) to choose and install your version.


## 3. Dataset Download

To download the datasets required for an exercise, execute the respective download script located in the exercise directory:

`./get_datasets.sh`

You will need ~400MB of disk space.

