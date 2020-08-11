apt-get install -y software-properties-common
apt-get install build-essential gfortran pkg-config cmake -y
apt-get install libblas-dev liblapack-dev -y
apt-get install libboost-all-dev -y
apt-get install libsuitesparse-dev -y
apt-get install libsuitesparse-doc -y
apt-get install libdune-common-dev libdune-geometry-dev libdune-grid-dev libdune-istl-dev -y
apt-get install libtrilinos-zoltan-dev -y

for repo in opm-common opm-material opm-grid opm-models opm-simulators opm-upscaling
do
    echo "=== Cloning and building module: $repo"
    git clone https://github.com/OPM/$repo.git
    mkdir $repo/build
    cd $repo/build
    cmake ..
    make
    cd ../..
done
