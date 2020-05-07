#!/usr/bin/env bash

# This file consists of the build steps needed to populate an existing
# environment with what is needed by DDPD Flownet. The existing environment
# could e.g. be a standard virtual environment or a conda environment.
#
# This build script is suitable for being run e.g. in CI pipelines or a VM.
#
# Note that there are also some apt-get installs (i.e. there are
# side effects also outside of your Python environment)
#
# The script takes one argument: The path to a Python environment (either a standard
# python virtual environment, or a conda environment). If the path exists, the environment
# is assumed to be active. If it doesn't exist, a virtual envirnoment is created at
# the given path.

LIBRES_VERSION="6d7ac59"
ERT_VERSION="c74e1e6"

set -e

if [ -z "$1" ]; then
    echo "ERROR: Environment folder argument not given"
    exit 1
fi

INSTALL_ENV=`readlink -f $1`
ONLY_APT_INSTALL=$2

####################
# UNPACK TEST DATA #
####################

gunzip tests/data/norne.tar.gz -c | tar -xvzf - -C tests/data/

############################
# INSTALL apt-get PACKAGES #
############################

sudo apt-get install zlib1g-dev  # libecl
sudo apt-get install libblas-dev liblapack-dev  # libres
sudo apt-get install libnss3-tools # webviz
sudo apt-get install libboost-all-dev liblapack-dev # opm-common

# Flow:
sudo apt-get install software-properties-common -y
sudo apt-get install libdune-common-dev libdune-geometry-dev -y
sudo apt-add-repository ppa:opm/ppa -y
sudo apt-get update
sudo apt-get install mpi-default-bin libopm-simulators-bin -y

if [ "$ONLY_APT_INSTALL" = "true" ]; then
  exit 0
fi

if [[ ! -d "$INSTALL_ENV" ]]; then
    echo "Folder $INSTALL_ENV does not exist."
    echo "Building a virtual environment at $INSTALL_ENV"

    python3 -m venv $INSTALL_ENV

    source $INSTALL_ENV/bin/activate
    pip install --upgrade pip

fi

##################
# INSTALL LIBECL #
##################

# Download and set LIBECL_VERSION specified in libres:
$(wget https://raw.githubusercontent.com/equinor/libres/$LIBRES_VERSION/.libecl_version -q -O -)

git clone https://github.com/equinor/libecl
pushd libecl
git checkout $LIBECL_VERSION
pip install -r requirements.txt
mkdir build
pushd build
cmake .. -DENABLE_PYTHON=ON \
         -DBUILD_TESTS=OFF \
         -DCMAKE_INSTALL_PREFIX=$INSTALL_ENV
make
make install

popd # build
popd # libecl

rm -rf libecl
python -c "import ecl"  # Check able to import

echo "Finished installing libecl"

##################
# INSTALL LIBRES #
##################

git clone https://github.com/equinor/libres
pushd libres
git checkout $LIBRES_VERSION
pip install -r requirements.txt
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_ENV
make
make install
popd # build
popd # libres

rm -rf libres
python -c "import res"  # Check able to import

echo "Finished installing libres"

###############
# INSTALL ERT #
###############

git clone https://github.com/equinor/ert

pushd ert
git checkout $ERT_VERSION

pip install -r requirements.txt
pip install .

popd

rm -rf ./ert

######################
# INSTALL OPM-COMMON #
######################

git clone --recursive https://github.com/OPM/opm-common.git
pushd opm-common
git checkout tags/testing/2020.3/rc1
mkdir build
pushd build
cmake .. -DCMAKE_PREFIX_PATH=$INSTALL_ENV \
         -DCMAKE_INSTALL_PREFIX=$INSTALL_ENV \
         -DBUILD_TESTING=OFF \
         -DBUILD_SHARED_LIBS=ON \
         -DOPM_ENABLE_PYTHON=ON \
         -DOPM_INSTALL_PYTHON=ON
make -j4 install
popd # build
popd # opm-common

rm -rf ./opm-common
python -c "import opm"  # Check able to import

######################
# CREATE FLOW CONFIG #
######################

PACKAGE_FOLDER="dist-packages"
FLOW_PATH="/usr/bin/flow"

cat >`ls -d $INSTALL_ENV/lib/python3*/$PACKAGE_FOLDER/res/fm/ecl`/flow_config.yml <<EOL
default_version: default
versions:
  'default':
    scalar:
      executable: $FLOW_PATH
EOL

########################
# INSTALL FMU-ENSEMBLE #
########################

git clone https://github.com/equinor/fmu-ensemble

pushd fmu-ensemble

git checkout 'v1.1.0'

pip install -r requirements.txt
pip install .

popd

rm -rf ./fmu-ensemble
