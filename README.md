# Data Engineering II Lab 3

## Desription
- In this assignment I will explore some of the challenges and possible solutions arising when
scaling up machine learning tasks and pipelines to a distributed setting.
### Part 1: Distributed machine learning
1.1) Hyper Parameter Tuning: Implement a random forest on the forest cover dataset to identify forest over type.
- The data: https://www.kaggle.com/c/forest-cover-type-prediction
- Fit default parameters to the dataset using sklearn.ensemble.RandomForestClassifier
- Then use Ray Tune to do hyper parameter tuning.
To run the default tuning, cd into /tuning and run `docker compose up` or alternatively python main.py with the proper dependencies (listed in requirements.txt). 
Questions to answer:

Q1) The hyperparameters found and the associated cross-validation score. How does it compare to the score for the default parameters?

Q1) The time taken to complete the tuning, when using 1, 2 and 3 VMs of “small” flavor.

1.2) Distributed training of deep neural networks (on CPU)
- Use Ray Train to distribute learning across multiple machines

Q1) Equipped with this background knowledge, explain the main difference between data parallel training strategies and model parallel training strategies. In which situations are the respective strategies most appropriate to use? Assuming you also need to do hyperparameter tuning, in which situations would you consider distributing the training of individual neural networks rather than distributing test cases for the tuning pipeline?

### Part 2: Federated Learning
- Use FEDn to set up and run fully distributed federated training for either the Keras or the PyTorch MNIST example (bundled in the repository, examples folder) on three virtual machines. Use one for the base services (MongoDB and Minio)+reducer, one for one combiner and one to run 2 clients. Include a snapshot of the training result in the Dashboard. 

Q1) Write a short (max 0.5 page Arial 11pt) reflection on how federated learning differs from distributed machine learning. Focus on statistical and system heterogeneity.

Q2) Write a short (max 0.5 page Arial 11pt) discussion on how federated learning relates to the Parameter Server strategy for distributed machine learning.


## Running the project

Directory structure:
each directory from root represents one part of the above description. Everything is made to be run inside of Docker. More specifically each part of the assignment is made to be runnable in a distributed environment using Docker Swarm mode.

Get started with your cluster by spinning up a node. cloud-cfg.txt can be used to contextualize a VM with Docker so you can easily add nodes to the cluster. Once you have a node/instance up, you can run any of the element of the assignment by using the docker commands outlined below. To add nodes to the cluster, configure a VM as the swarm manager using:

```bash
docker swarm init --advertise-addr <YOUR MANGER'S LOCAL IP>
```

Add new nodes to the cluster by running the following on the other VMs:
```bash
docker swarm join --token <SWARM TOKEN> <YOUR MANGER'S LOCAL IP>:2377
```

Get your managers token again by running (it will print the command you need to add a worker node the cluster):
```bash
docker swarm join-token worker
```

cd into data_eng_lab3 to run the different application related to the assignment.
ex. after starting the services)
```bash
docker exec -it <RAY HEAD CONTAINER ID> /bin/bash
ray job submit -- python tune.py
```

### /tuning 
The "tuning" directory contains everyting needed to run part 1.1 ie running the cross validation on tree classification of forestry type with the default parameters. The script has been dockerized and running `docker compose up --build` will build the python image and run the main.py script and print the results. The script can also be run with a python virutal environment (venv) using `pip install -r requirements.txt` and then `python main.py`.


### /ray_tuning
The "ray_tuning" directory contains everything needed to run the distributed tuning of the model using Ray Tune. The app has also been dockerized and the ray instances can be controlled with docker compose. The default compose spawns one Ray head node (in a container) and one worker node. Testing the tuning times can be controlled by modifiying the deploy resources and/or the worker count. You can also modify the script to tell Ray to limit resources but I wanted to guarantee resource limits on the services and more easily add nodes to the cluster (via Docker Swarm).

#### To run on a single node:


#### To run on across the swarm cluster:











