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
Q1) The hyperparameters found and the associated cross-validation score. How does it
compare to the score for the default parameters?
Q1) The time taken to complete the tuning, when using 1, 2 and 3 VMs of “small” flavor.

1.2) Distributed training of deep neural networks (on CPU)
- Use Ray Train to distribute learning across multiple machines
Q1) Equipped with this background knowledge, explain the main difference between data parallel training strategies and model parallel training strategies. In which situations are the respective strategies most appropriate to use? Assuming you also need to do hyperparameter tuning, in which situations would you consider distributing the training of individual neural networks rather than distributing test cases for the tuning pipeline?

### Part 2: Federated Learning
- Use FEDn to set up and run fully distributed federated training for either the Keras or the PyTorch MNIST example (bundled in the repository, examples folder) on three virtual machines. Use one for the base services (MongoDB and Minio)+reducer, one for one combiner and one to run 2 clients. Include a snapshot of the training result in the Dashboard. 
Q1) Write a short (max 0.5 page Arial 11pt) reflection on how federated learning differs from distributed machine learning. Focus on statistical and system heterogeneity.
Q2) Write a short (max 0.5 page Arial 11pt) discussion on how federated learning relates to the Parameter Server strategy for distributed machine learning.


