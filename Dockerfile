FROM ubuntu:focal

ENV DEBIAN_FRONTEND noninteractive

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID flownet
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID flownet

RUN apt update && \
    apt install -y --no-install-recommends software-properties-common && \
    apt-add-repository ppa:opm/ppa -y && \
    apt-get update && \
    apt install -y --no-install-recommends \
    mpi-default-bin \
    libopm-simulators-bin \
    zlib1g-dev \
    libblas-dev liblapack-dev \
    libgl1 \
    git \
    python3-pip \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    make \
    python3-ipython \
    python3-ipdb \
    black \
    pylint \
    libxrender1 \
    python3-pytest \
    libnss3-tools && \
    apt-get -y install -o Dpkg::Options::="--force-confold" sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN adduser flownet sudo && \
    echo "flownet:docker" | chpasswd && \
    echo "flownet ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER flownet

RUN echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
