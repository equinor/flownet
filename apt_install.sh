set -e
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y software-properties-common zlib1g-dev libblas-dev liblapack-dev libnss3-tools
add-apt-repository -y ppa:opm/ppa
apt-get update
apt-get install -y mpi-default-bin libopm-simulators-bin
