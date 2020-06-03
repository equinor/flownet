apt-get install zlib1g-dev  # libecl
apt-get install libblas-dev liblapack-dev  # libres
apt-get install libnss3-tools # webviz
apt-get install libboost-all-dev liblapack-dev # opm-common

# Flow:
apt-get install software-properties-common -y
apt-get install libdune-common-dev libdune-geometry-dev -y
apt-add-repository ppa:opm/ppa -y
apt-get update
apt-get install mpi-default-bin libopm-simulators-bin -y
