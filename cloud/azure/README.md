# FlowNet Azure Function

This document describes how to enable running FlowNet forward models in the cloud use Azure Functions.

## Installation

It is assumed that you have [Docker](https://docs.docker.com/get-docker/) installed.

## Build & Test the Docker Container

1. Build the docker container and upload it to some location accessible by Azure Functions
   ```
   docker build ./ -t flownet_azure_function
   ```  
1. Run the docker container
   ```
   docker run -p 8080:80 flownet_azure_function
   ```
1. Launch a browser and goto 
   ```
   http://localhost:8080/api/HttpTrigger/?name=Test
   ```

## Publish the Docker Container

### Publish to Docker Hub

Not described yet. This solution is preferred to publish an official FlowNet Azure Function container.

### Publish to Azure Container Registry

1. Login use the Azure CLI
   ```
   az login
   ```
1. Login to the Azure Container Registry you want to use:
   ```
   az acr login --name [acr_name]
   ```
1. Retag your Docker image
   ```
   docker tag flownet_azure_function [acr_name].azurecr.io/flownet_azure_function
   ```
1. Publish your Docker image
   ```
   docker push [acr_name].azurecr.io/flownet_azure_function
   ```

## Create the Azure Function App

1. Create a new Azure Function App, and give it a name (eg, "my_function")
1. Choose "Docker Container" in the wizzard
1. Finalize the wizzard
1. Open your newly created Azure Function App and goto container settings.
1. Locate the published container and save.
1. Set "FUNCTIONS_WORKER_PROCESS_COUNT" to 10.

Check that your new Azure Function App is up and running by going to: 
```
https://[my_function].azurewebsites.net/api/HttpTrigger/?name=test
```
