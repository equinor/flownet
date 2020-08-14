FROM mcr.microsoft.com/azure-functions/python:3.0-python3.7

# LABEL about the custom image
LABEL maintainer="wbr@equinor.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
running FlowNet forward jobs on the Azure cloud using an Azure Function."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update package list
RUN apt-get update -y

# Install Git
RUN apt-get install git -y

# Install OPM Flow
COPY build_opm_flow.sh build_opm_flow.sh
RUN bash build_opm_flow.sh

# Install additional requirements
RUN apt install libgl1-mesa-glx -y

# Install python requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Add FlowNet Python Azure Function
COPY . /home/site/wwwroot
