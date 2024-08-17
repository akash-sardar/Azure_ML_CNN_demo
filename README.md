# Azure_ML_CNN_demo
CNN on MNIST to be deployed on Azure
## Connect to Azure Workspace
1. Connect to Azure Machine Learning Workspace. The Azure Machine Learning Workspace is the top-level resource for the service. It provides a centralized place to work (a curated environment) with all the artifacts
2. Get a Handle to the workspace by providing subscription Id, Resource Group name and workspace name.
## Create a compute Resource
1. Azure Machine Learning needs a compute resource to run a job. This resource can be simgle or multi-node machines with Linux or Windows OS
2. For this example - we will use Standard_NC4as_T4_v3 with 4 cores, 28 GB RAM, 176 GB Storage
3. More reosurces are listed in - https://azure.microsoft.com/en-us/pricing/details/machine-learning/#pricing
## Create a job environment
1. An Azure Machine learning eenvironment encapsulates the dependencies needed to run machine elarning training script on the created compute resource. This environment is similar to python venv on local machine.
2. The Azure machine learning environemtn allows us to either use a curated environment - useful for common training and inference scenarios - or create a custom environment using a docker image or Conda configuration
3. In this scenario we will use curated Azure Machine Learning environment - "AzureML-tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu"
## Configure and submit the training job
1. This section will deal entirely with training a model on standard MNIST dataset and then executing the training script on Azure Machine Learning

