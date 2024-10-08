For training and evaluating the model for each experiments,everything is done by CLI. So all the arguments are parsed. Therefore for each components the arguments are parsed.
And a configuration file is setup for the values for each components in the embedding, encoding, decoding, output, training etc..
So mainly the configuration files are setup for 5 main components of the pipeline
1. Datasets
2. Loader
3. model - embedding, encoder, decoder, mlp(can be used for all the FFN), output, SSM(Closed loop, Openloop) etc..
4. optimizer
5. scheduler

So all these are created as yaml files.This are processed by Omegaconf library. Sometimes when experimenting we need to change the meta data inside this config.
 So  functions are required to read and update the values. So the required functions are defined in this directory (setup).
 #todo Check all the python scripts below utils to understand how the whole pipeline is scripted
 
 #todo(22/07/24) 
 go through this directory thoroughly to understand how the arguments are parsed. Read the paper in parallel to understand the functions defined in the (model) directory
 Objective:_ Our main Objective is to understand the code components thoroughly so that we can write new classes and functionalities to create our new model and train it 
 with our own Datasets and do experiments with different hyperparameters to improve the efficiency
 #todo- check the code part where the authors defined a function to evaluate computation performance
