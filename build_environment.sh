#!/usr/bin/env bash

# This file consists of the build steps needed to populate an existing
# environment with what is needed by DDPD Flownet. The existing environment
# could e.g. be a standard virtual environment or a conda environment.
#
# This build script is suitable for being run e.g. in CI pipelines or a VM.
#
# The script takes two arguments: The path to a Python environment (either a standard
# python virtual environment, or a conda environment). If the path exists, the environment
# is assumed to be active. If it doesn't exist, a virtual envirnoment is created at
# the given path. The second argument is the path to your installed flow binary, typically
# something like /usr/bin/flow

LIBRES_VERSION="6d7ac59"
ERT_VERSION="c74e1e6"

set -e

if [ -z "$1" ]; then
    echo "ERROR: Environment folder argument not given"
    exit 1
fi

INSTALL_ENV=`readlink -f $1`
FLOW_PATH=`readlink -f $2`

####################
# UNPACK TEST DATA #
####################

tar -zxvf tests/data/norne.tar.gz -C tests/data/

############################
# INSTALL apt-get PACKAGES #
############################

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

pip install git+https://github.com/equinor/ert@$ERT_VERSION

######################
# INSTALL OPM-COMMON #
######################

pip install -i https://test.pypi.org/simple/ CeeSolOpm

######################
# CREATE FLOW CONFIG #
######################

cat >`ls -d $INSTALL_ENV/lib/python3*/*/res/fm/ecl`/flow_config.yml <<EOL
default_version: default
versions:
  'default':
    scalar:
      executable: $FLOW_PATH
EOL
