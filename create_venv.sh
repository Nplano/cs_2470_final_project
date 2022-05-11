#!/bin/bash

# install virtualenv module
python3 -m pip install virtualenv

# create virtual environment named "env"
virtualenv -p python3 env

# activate new virtual environment
source env/bin/activate

# update pip
pip install -U pip 

# install required python packages to the virtual environment
pip install -r requirements.txt

echo DL project environment created!
