# Moon_Data
Training a custom,PyTorch neural network to create a binary classifier for data that is separated into two classes; the data looks like two moon shapes when it is displayed.



The notebook will be broken down into a few steps:

Generating the moon data
Loading it into an S3 bucket
Defining a PyTorch binary classifier
Completing a training script
Training and deploying the custom model
Evaluating its performance
Being able to train and deploy custom models is a really useful skill to have. Especially in applications that may not be easily solved by traditional algorithms like a LinearLearner.

Load in required libraries, below.

In [1]:
# data 
import pandas as pd 
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

%matplotlib inline
Generating Moon Data
Below, I have written code to generate some moon data, using sklearn's make_moons and train_test_split.

I'm specifying the number of data points and a noise parameter to use for generation. Then, displaying the resulting data.

In [3]:
# plot
# points are colored by class, Y_train
# 0 labels = purple, 1 = yellow
plt.figure(figsize=(8,5))
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
plt.title('Moon Data')
plt.show()

SageMaker Resources

# sagemaker
import boto3
import sagemaker
from sagemaker import get_execution_role
In [5]:
# SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# default S3 bucket
bucket = sagemaker_session.default_bucket()
EXERCISE: Create csv files
Define a function that takes in x (features) and y (labels) and saves them to one .csv file at the path data_dir/filename. SageMaker expects .csv files to be in a certain format, according to the documentation:

Amazon SageMaker requires that a CSV file doesn't have a header record and that the target variable is in the first column.

It may be useful to use pandas to merge your features and labels into one DataFrame and then convert that into a .csv file. When you create a .csv file, make sure to set header=False, and index=False so you don't include anything extraneous, like column names, in the .csv file.

In [6]:
import os

def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # first column is the labels and rest is features 
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1)\
             .to_csv(os.path.join(data_dir, filename), header=False, index=False)
    
    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))
The next cell runs the above function to create a train.csv file in a specified directory.

In [7]:
data_dir = 'data_moon' # the folder we will use for storing data
name = 'train.csv'

# create 'train.csv'
make_csv(X_train, Y_train, name, data_dir)
Path created: data_moon/train.csv
Upload Data to S3
Upload locally-stored train.csv file to S3 by using sagemaker_session.upload_data. This function needs to know: where the data is saved locally, and where to upload in S3 (a bucket and prefix).

In [8]:
# specify where to upload in S3
prefix = 'moon-data'

# upload to S3
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
print(input_data)
s3://sagemaker-us-west-1-467380521728/moon-data
Check that you've uploaded the data, by printing the contents of the default bucket.

In [9]:
# iterate through S3 objects and print contents
for obj in boto3.resource('s3').Bucket(bucket).objects.all():
     print(obj.key)
moon-data/train.csv
sagemaker-pytorch-2019-03-12-03-19-53-656/sourcedir.tar.gz
Modeling
Now that you've uploaded your training data, it's time to define and train a model!

In this notebook, you'll define and train a custom PyTorch model. This will be a neural network that performs binary classification.

EXERCISE: Define a model in model.py
To implement a custom classifier, the first thing you'll do is define a neural network. You've been give some starting code in the directory source, where you can find the file, model.py. You'll need to complete the class SimpleNet; specifying the layers of the neural network and its feedforward behavior. It may be helpful to review the code for a 3-layer MLP.

This model should be designed to:

Accept a number of input_features (the number of anonymized transaction features).
Create some Linear, hidden layers of a desired size
Return a single output value that indicates the class score
The returned output value should be a sigmoid-activated class score; a value between 0-1 that can be rounded to get a predicted, class label.

Below, you can use !pygmentize to display the code in the model.py file. Read through the code; all of your tasks are marked with TODO comments. You should navigate to the file, and complete the tasks to define a SimpleNet.

In [10]:
!pygmentize source_solution/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
                
        # defining 2 linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.3)
        # sigmoid layer
        self.sig = nn.Sigmoid()
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        out = F.relu(self.fc1(x)) # activation on hidden layer
        out = self.drop(out)
        out = self.fc2(out)
        return self.sig(out) # returning class score
Training Script
To implement a custom classifier, you'll also need to complete a train.py script. You can find this in the source directory.

A typical training script:

Loads training data from a specified directory
Parses any training & model hyperparameters (ex. nodes in a neural network, training epochs, etc.)
Instantiates a model of your design, with any specified hyperparams
Trains that model
Finally, saves the model so that it can be hosted/deployed, later
EXERCISE: Complete the train.py script
Much of the training script code is provided for you. Almost all of your work will be done in the if name == 'main': section. To complete the train.py file, you will:

Define any additional model training hyperparameters using parser.add_argument
Define a model in the if name == 'main': section
Train the model in that same section
Below, you can use !pygmentize to display an existing train.py file. Read through the code; all of your tasks are marked with TODO comments.

In [11]:
!pygmentize source_solution/train.py
from __future__ import print_function # future proof
import argparse
import sys
import os
import json

import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# import model
from model import SimpleNet


def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], 
                      model_info['hidden_dim'], 
                      model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)


# Load the training data from a csv file
def _get_train_loader(batch_size, data_dir):
    print("Get data loader.")

    # read in csv file
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None)

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # features are the rest
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save trained model, after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)


## TODO: Complete the main code
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    ## TODO: Add args for the three model parameters: input_dim, hidden_dim, output_dim
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=2, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H',
                        help='hidden dim of model (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')

    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # get train loader
    train_loader = _get_train_loader(args.batch_size, args.data_dir) # data_dir from above..
    
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = SimpleNet(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    
    # Given: save the parameters used to construct the model
    save_model_params(model, args.model_dir)

    ## TODO: Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, args.epochs, optimizer, criterion, device)
    
    
EXERCISE: Create a PyTorch Estimator
You've had some practice instantiating built-in models in SageMaker. All estimators require some constructor arguments to be passed in. When a custom model is constructed in SageMaker, an entry point must be specified. The entry_point is the training script that will be executed when the model is trained; the train.py function you specified above!

See if you can complete this task, instantiating a PyTorch estimator, using only the PyTorch estimator documentation as a resource. It is suggested that you use the latest version of PyTorch as the optional framework_version parameter.

Instance Types
It is suggested that you use instances that are available in the free tier of usage: 'ml.c4.xlarge' for training and 'ml.t2.medium' for deployment.

In [12]:
# import a PyTorch wrapper
from sagemaker.pytorch import PyTorch

# specify an output path
# prefix is specified above
output_path = 's3://{}/{}'.format(bucket, prefix)

# instantiate a pytorch estimator
estimator = PyTorch(entry_point='train.py',
                    source_dir='source_solution', # this should be just "source" for your code
                    role=role,
                    framework_version='1.0',
                    train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    output_path=output_path,
                    sagemaker_session=sagemaker_session,
                    hyperparameters={
                        'input_dim': 2,  # num of features
                        'hidden_dim': 20,
                        'output_dim': 1,
                        'epochs': 80 # could change to higher
                    })
Train the Estimator
After instantiating your estimator, train it with a call to .fit(). The train.py file explicitly loads in .csv data, so you do not need to convert the input data to any other format.

In [13]:
%%time 
# train the estimator on S3 training data
estimator.fit({'train': input_data})
INFO:sagemaker:Creating training-job with name: sagemaker-pytorch-2019-03-12-03-20-56-668
2019-03-12 03:20:56 Starting - Starting the training job...
2019-03-12 03:21:00 Starting - Launching requested ML instances......
2019-03-12 03:22:00 Starting - Preparing the instances for training......
2019-03-12 03:23:24 Downloading - Downloading input data
2019-03-12 03:23:24 Training - Training image download completed. Training in progress.
2019-03-12 03:23:24 Uploading - Uploading generated training model
bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
2019-03-12 03:23:14,349 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training
2019-03-12 03:23:14,352 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
2019-03-12 03:23:14,368 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.
2019-03-12 03:23:15,772 sagemaker_pytorch_container.training INFO     Invoking user training script.
2019-03-12 03:23:16,040 sagemaker-containers INFO     Module train does not provide a setup.py. 
Generating setup.py
2019-03-12 03:23:16,040 sagemaker-containers INFO     Generating setup.cfg
2019-03-12 03:23:16,040 sagemaker-containers INFO     Generating MANIFEST.in
2019-03-12 03:23:16,040 sagemaker-containers INFO     Installing module with the following command:
/usr/bin/python -m pip install -U . 
Processing /opt/ml/code
Building wheels for collected packages: train
  Running setup.py bdist_wheel for train: started
  Running setup.py bdist_wheel for train: finished with status 'done'
  Stored in directory: /tmp/pip-ephem-wheel-cache-wii_guhj/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3
Successfully built train
Installing collected packages: train
Successfully installed train-1.0.0
You are using pip version 18.1, however version 19.0.3 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
2019-03-12 03:23:17,599 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
2019-03-12 03:23:17,612 sagemaker-containers INFO     Invoking user script

Training Env:

{
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "train": "/opt/ml/input/data/train"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_pytorch_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "input_dim": 2,
        "hidden_dim": 20,
        "epochs": 80,
        "output_dim": 1
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "train": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        }
    },
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "sagemaker-pytorch-2019-03-12-03-20-56-668",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-us-west-1-467380521728/sagemaker-pytorch-2019-03-12-03-20-56-668/source/sourcedir.tar.gz",
    "module_name": "train",
    "network_interface_name": "ethwe",
    "num_cpus": 4,
    "num_gpus": 0,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "hosts": [
            "algo-1"
        ],
        "network_interface_name": "ethwe"
    },
    "user_entry_point": "train.py"
}

Environment variables:

SM_HOSTS=["algo-1"]
SM_NETWORK_INTERFACE_NAME=ethwe
SM_HPS={"epochs":80,"hidden_dim":20,"input_dim":2,"output_dim":1}
SM_USER_ENTRY_POINT=train.py
SM_FRAMEWORK_PARAMS={}
SM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"ethwe"}
SM_INPUT_DATA_CONFIG={"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}
SM_OUTPUT_DATA_DIR=/opt/ml/output/data
SM_CHANNELS=["train"]
SM_CURRENT_HOST=algo-1
SM_MODULE_NAME=train
SM_LOG_LEVEL=20
SM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main
SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_NUM_CPUS=4
SM_NUM_GPUS=0
SM_MODEL_DIR=/opt/ml/model
SM_MODULE_DIR=s3://sagemaker-us-west-1-467380521728/sagemaker-pytorch-2019-03-12-03-20-56-668/source/sourcedir.tar.gz
SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"train":"/opt/ml/input/data/train"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"epochs":80,"hidden_dim":20,"input_dim":2,"output_dim":1},"input_config_dir":"/opt/ml/input/config","input_data_config":{"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-pytorch-2019-03-12-03-20-56-668","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-west-1-467380521728/sagemaker-pytorch-2019-03-12-03-20-56-668/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"ethwe","num_cpus":4,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"ethwe"},"user_entry_point":"train.py"}
SM_USER_ARGS=["--epochs","80","--hidden_dim","20","--input_dim","2","--output_dim","1"]
SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
SM_CHANNEL_TRAIN=/opt/ml/input/data/train
SM_HP_INPUT_DIM=2
SM_HP_HIDDEN_DIM=20
SM_HP_EPOCHS=80
SM_HP_OUTPUT_DIM=1
PYTHONPATH=/usr/local/bin:/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages

Invoking script with the following command:

/usr/bin/python -m train --epochs 80 --hidden_dim 20 --input_dim 2 --output_dim 1


Get data loader.
Epoch: 1, Loss: 0.8094824403524399
Epoch: 2, Loss: 0.8087087497115135
Epoch: 3, Loss: 0.81452776491642
Epoch: 4, Loss: 0.7784294486045837
Epoch: 5, Loss: 0.7738661542534828
Epoch: 6, Loss: 0.7469970062375069
Epoch: 7, Loss: 0.7502057999372482
Epoch: 8, Loss: 0.7338991761207581
Epoch: 9, Loss: 0.7060637846589088
Epoch: 10, Loss: 0.70991051197052
Epoch: 11, Loss: 0.6829910576343536
Epoch: 12, Loss: 0.6872642189264297
Epoch: 13, Loss: 0.6796583235263824
Epoch: 14, Loss: 0.6750186383724213
Epoch: 15, Loss: 0.6474509462714195
Epoch: 16, Loss: 0.6489695161581039
Epoch: 17, Loss: 0.6278565004467964
Epoch: 18, Loss: 0.6373456940054893
Epoch: 19, Loss: 0.6274193376302719
Epoch: 20, Loss: 0.6120161935687065
Epoch: 21, Loss: 0.6031808331608772
Epoch: 22, Loss: 0.5973624065518379
Epoch: 23, Loss: 0.5929201915860176
Epoch: 24, Loss: 0.5777265653014183
Epoch: 25, Loss: 0.58209378272295
Epoch: 26, Loss: 0.5992862954735756
Epoch: 27, Loss: 0.5764492303133011
Epoch: 28, Loss: 0.5612449124455452
Epoch: 29, Loss: 0.5468951836228371
Epoch: 30, Loss: 0.5476799719035625
Epoch: 31, Loss: 0.5541884526610374
Epoch: 32, Loss: 0.5619134604930878
Epoch: 33, Loss: 0.5442568808794022
Epoch: 34, Loss: 0.5280377380549908
Epoch: 35, Loss: 0.5323713757097721
Epoch: 36, Loss: 0.5295680649578571
Epoch: 37, Loss: 0.5200584568083286
Epoch: 38, Loss: 0.5405271053314209
Epoch: 39, Loss: 0.526910662651062
Epoch: 40, Loss: 0.517667829990387
Epoch: 41, Loss: 0.5065649971365929
Epoch: 42, Loss: 0.5186041817069054
Epoch: 43, Loss: 0.48348696157336235
Epoch: 44, Loss: 0.4920388348400593
Epoch: 45, Loss: 0.48314499482512474
Epoch: 46, Loss: 0.4986172467470169
Epoch: 47, Loss: 0.4838500805199146
Epoch: 48, Loss: 0.5062692202627659
Epoch: 49, Loss: 0.4790864698588848
Epoch: 50, Loss: 0.4843643642961979
Epoch: 51, Loss: 0.48715561255812645
Epoch: 52, Loss: 0.4674951285123825
Epoch: 53, Loss: 0.4855138994753361
Epoch: 54, Loss: 0.4692947752773762
Epoch: 55, Loss: 0.47163090109825134
Epoch: 56, Loss: 0.46736880764365196
Epoch: 57, Loss: 0.4675491377711296
Epoch: 58, Loss: 0.4782943092286587
Epoch: 59, Loss: 0.45643892511725426
Epoch: 60, Loss: 0.46863533183932304
Epoch: 61, Loss: 0.46333420276641846
Epoch: 62, Loss: 0.48059647530317307
Epoch: 63, Loss: 0.45397504046559334
Epoch: 64, Loss: 0.45192109420895576
Epoch: 65, Loss: 0.46614759787917137
Epoch: 66, Loss: 0.4529973901808262
Epoch: 67, Loss: 0.4322005622088909
Epoch: 68, Loss: 0.44345563650131226
Epoch: 69, Loss: 0.44864876195788383
Epoch: 70, Loss: 0.4391746446490288
Epoch: 71, Loss: 0.4342072270810604
Epoch: 72, Loss: 0.4262286312878132
Epoch: 73, Loss: 0.4268296808004379
Epoch: 74, Loss: 0.4437255598604679
Epoch: 75, Loss: 0.43882467597723007
Epoch: 76, Loss: 0.44645144790410995
Epoch: 77, Loss: 0.4345819540321827
Epoch: 78, Loss: 0.4224093407392502
Epoch: 79, Loss: 0.4226452559232712
Epoch: 80, Loss: 0.42246998474001884
Saving the model.
2019-03-12 03:23:19,489 sagemaker-containers INFO     Reporting training SUCCESS

2019-03-12 03:23:29 Completed - Training job completed
Billable seconds: 30
CPU times: user 431 ms, sys: 20.8 ms, total: 451 ms
Wall time: 3min 11s
Create a Trained Model
PyTorch models do not automatically come with .predict() functions attached (as many Scikit-learn models do, for example) and you may have noticed that you've been give a predict.py file. This file is responsible for loading a trained model and applying it to passed in, numpy data. When you created a PyTorch estimator, you specified where the training script, train.py was located.

How can we tell a PyTorch model where the predict.py file is?

Before you can deploy this custom PyTorch model, you have to take one more step: creating a PyTorchModel. In earlier exercises you could see that a call to .deploy() created both a model and an endpoint, but for PyTorch models, these steps have to be separate.

EXERCISE: Instantiate a PyTorchModel
You can create a PyTorchModel (different that a PyTorch estimator) from your trained, estimator attributes. This model is responsible for knowing how to execute a specific predict.py script. And this model is what you'll deploy to create an endpoint.

Model Parameters
To instantiate a PyTorchModel, (documentation, here) you pass in the same arguments as your PyTorch estimator, with a few additions/modifications:

model_data: The trained model.tar.gz file created by your estimator, which can be accessed as estimator.model_data.
entry_point: This time, this is the path to the Python script SageMaker runs for prediction rather than training, predict.py.
In [14]:
# importing PyTorchModel
from sagemaker.pytorch import PyTorchModel

# Create a model from the trained estimator data
# And point to the prediction script
model = PyTorchModel(model_data=estimator.model_data,
                     role = role,
                     framework_version='1.0',
                     entry_point='predict.py',
                     source_dir='source_solution')
EXERCISE: Deploy the trained model
Deploy your model to create a predictor. We'll use this to make predictions on our test data and evaluate the model.

In [15]:
%%time
# deploy and create a predictor
predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
INFO:sagemaker:Creating model with name: sagemaker-pytorch-2019-03-12-03-24-09-384
INFO:sagemaker:Creating endpoint with name sagemaker-pytorch-2019-03-12-03-24-09-384
---------------------------------------------------------------------------------------!CPU times: user 591 ms, sys: 62.9 ms, total: 654 ms
Wall time: 7min 21s
Evaluating Your Model
Once your model is deployed, you can see how it performs when applied to the test data.

The provided function below, takes in a deployed predictor, some test features and labels, and returns a dictionary of metrics; calculating false negatives and positives as well as recall, precision, and accuracy.

In [16]:
# code to evaluate the endpoint on test data
# returns a variety of model metrics
def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """
    
    # rounding and squeezing array
    test_preds = np.squeeze(np.round(predictor.predict(test_features)))
    
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()
    
    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # print metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actuals'], colnames=['predictions']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()
        
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
Test Results
The cell below runs the evaluate function.

The code assumes that you have a defined predictor and X_test and Y_test from previously-run cells.

In [17]:
# get metrics for custom predictor
metrics = evaluate(predictor, X_test, Y_test, True)
predictions  0.0  1.0
actuals              
0             53   18
1             11   68

Recall:     0.861
Precision:  0.791
Accuracy:   0.807

Delete the Endpoint
Finally, I've add a convenience function to delete prediction endpoints after we're done with them. And if you're done evaluating the model, you should delete your model endpoint!

In [18]:
# Accepts a predictor endpoint as input
# And deletes the endpoint by name
def delete_endpoint(predictor):
        try:
            boto3.client('sagemaker').delete_endpoint(EndpointName=predictor.endpoint)
            print('Deleted {}'.format(predictor.endpoint))
        except:
            print('Already deleted: {}'.format(predictor.endpoint))
In [19]:
# delete the predictor endpoint 
delete_endpoint(predictor)
Deleted sagemaker-pytorch-2019-03-12-03-24-09-384
Final Cleanup!
Double check that you have deleted all your endpoints.
I'd also suggest manually deleting your S3 bucket, models, and endpoint configurations directly from your AWS console.
You can find thorough cleanup instructions, in the documentation.

Conclusion
In this notebook, you saw how to train and deploy a custom, PyTorch model in SageMaker. SageMaker has many built-in models that are useful for common clustering and classification tasks, but it is useful to know how to create custom, deep learning models that are flexible enough to learn from a variety of data.
