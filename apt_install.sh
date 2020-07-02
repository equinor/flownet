apt-get install zlib1g-dev  # libecl
apt-get install libblas-dev liblapack-dev  # libres
apt-get install libnss3-tools # webviz

# Flow:
apt-get update
apt-get install software-properties-common -y
apt-add-repository ppa:opm/ppa -y
apt-get update
apt-get install mpi-default-bin libopm-simulators-bin -y
