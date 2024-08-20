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
This section will deal entirely with training a model on standard MNIST dataset and then executing the training script on Azure Machine Learning
### Prepare the training script
In the training script, app.py, we create a simple CNN that trains on the MNIST Dataset 
1. Configuration: Sets up hyperparameters - batch size, epochs and learning rate
2. Get Current Run: Retrieves the curent Azure ML run context to log metrics and artifacts
3. Load and Preprocess Data: Loads the MNIST dataset
4. Create CNN Model
5. Train Model
6. SaveModel - to outputs directory
7. Log Metrics - each epoch to azure ml
8. Complete Run - end of azure ml run
## Configure the command (CLI)
We use the general purpose command to run the training script and perform our desired tasks.
For the parameter values:
1. provide the **compute cluster** gpu_compute_target = "gpu-cluster"
2. provide the **curated envirnment name** curated_env_name = "AzureML-tensorflow-2.16-cuda11@latest"
3. configure the command line action - in this case - python app.py
3. configure the metadata such as **display name** and **experiment** name - an experiment is a container for  all the iterations on a certain project. All the jobs submited under the same experiment would be listed next to each other im Azure ML studio.
## Submit the job
create_or_update on ml_client.jobs is used to submit the job
Once completed, the job will register a model in ML workspace (as a result of training) and output a link for viewing the job in Azure Machine Learning Studio
## Job execution
1. Preparing: A docker image is created according to the environment defined. This image is uploaded to the workspace's container registry and cached for later runs.
2. Logs are streamed to the job history and can be viewed to monitor progress. If a curated environment is specified, the cached image backing that curated environemnt will be used.
3. Scaling: The cluster attempts to scale up if it requires more nodes to execute the run than are currently available
4. Running: All script in the script folder src are uploaded to the compute target, data stores are mounted or copied, and the script is executed. Outputs from STDOUT and the ./logs folder are streamed to the job history and can be used to monitor the job.
## Register the Model
Register the model thru UI itself instead of code
1. Navigate tot he training job executed in Jobs under Assets in the left panel
2. Select the jobs outputs corresponding to the executed training job.
3. Enter the name and Description of the Model as per your desired choice and set it's version to 1. Next >>
4. Register the Model
5. You will be able to see the registered model in the **Models** Section under **Assets**
## Store the model artifacts in outputs folder
Once the model is registered, the artifacts of the models can be stored from mnist_outputs to **outputs** folder
1. Navigate to the model registered in **Models**, download all into local machine
## Create Model Object for Deployment Purpose
## Deploy the model as an online endpoint
After registration, the model can be deployed as an online endpoint - that is an webservice in Azure Cloud
To deploy a machine learning service, we typically need:
1. The model assets - such as model file and metadata. Create a unqiue name using a universally unique identifier (UUID)
2. Some code to run as a service. The code executes the model on a given input request ( an entry script). This entry script receives data submitted to a deployed web service and passes it to the model. After the moodel processes the data, the script returns the model's response to the client. When we use an MLFlow Model. Azure ML automatically creates this script for us.
## Create a Scoring Script
## Deploy the model to the endpoint
After the endpoint is created, the model can be deployed with an entry script. An endpoint can have multiple deployments. Using rules, the endpoint can direct traffic to these deployments.
In this scenario, we create a single deployment that handles 100% of the incoming traffic. 
1. Deploy the resitered model
2. Score the model using score.py (evaluate)
3. Use the same curated environment to perform inferencing
## Test the deployment with a sample query
After the model is deployed, predict the output of the deployed modek using the invoke method on the endpoint. To run the inference, we use the sample files
## Cleanup resource