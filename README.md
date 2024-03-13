# Neural Network from Scratch for Digit Recognition

## Introduction to the Project

This repository contains the implementation of a simple neural network from scratch using NumPy for digit recognition. The neural network architecture consists of one hidden layer with ReLU activation and an output layer with softmax activation for multi-class classification.

In the `ann_no_packages.ipynb` notebook, include the code for building a neural network with 784 input layer neurons, 1 hidden layer, and 10 neuron output layer for 10 classes. In the `my_note.md` and `my_note.pdf`, include all the notes that were built while learning the mathematics behind neural networks. The purpose of this project is to unravel the mathematics behind deep learning. To set up the project on your local environment, please refer to the `setup.md` file. This file contains instructions on how to configure the necessary dependencies and environment settings to run the project locally.

## What is Artificial Neural Networks

Artificial Neural Networks (ANNs) are computational models inspired by the structure and functioning of biological neural networks, such as the human brain. ANNs consist of interconnected nodes, or neurons, organized into layers. These networks can learn to perform tasks by adjusting the connections between neurons based on input data.

## Nodes or Neurons

The basic building blocks of ANNs are neurons, which receive input signals, perform computations, and produce output signals. Each neuron applies an activation function to the weighted sum of its inputs, including a bias term. The bias term allows neurons to capture additional information and enables the network to learn complex relationships in the data.

<img src="https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/288ec499-6a80-4911-a504-cd421b5c86ec" alt="image" width="400">

### Activation Function

In the above figure, you can see **Activation function**. the The purpose of the activation function in a neural network is to introduce non-linearity into the model. Without an activation function, the output of each neuron would be a linear combination of its inputs, regardless of the number of layers in the network. This would limit the network's ability to learn complex patterns and relationships in the data.

### Weights and bias

Weights and biases are crucial parameters in artificial neural networks. Weights determine the strength of connections between neurons, while biases allow neurons to capture non-linear relationships in data. During training, both weights and biases are adjusted to minimize the difference between the network's output and the desired output. Together, they enable neural networks to learn from data and make accurate predictions by adapting their behavior to model complex relationships.

## Layers

In neural networks, layers refer to the different levels of computation within the network. Each layer consists of a collection of neurons (also called nodes or units) that perform specific tasks. There are typically three types of layers in a neural network:

![asf](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/92cdd0e6-4131-4caf-9bc0-b22965e4e79e)

### Input Layer
The input layer serves as the initial entry point for the raw input data into the neural network architecture. It consists of neurons, each representing a specific feature or attribute of the input data. For instance, in an image classification task, each neuron in the input layer may correspond to a pixel intensity value. The number of neurons in the input layer is determined by the dimensionality of the input data. If the input data has 10 features, the input layer will contain 10 neurons. 

In the digit dataset, each image consists of 28 x 28 pixels, resulting in a total of 784 features for one data sample. Therefore, there should be 784 nodes in the input layer.


### Hidden (or Dense) Layers
Hidden layers are intermediary layers between the input and output layers. They perform computations on the input data through weighted connections and apply activation functions to produce output. These layers enable the network to learn complex representations and patterns in the data. In a fully connected neural network, also known as a dense neural network, each neuron in a hidden layer is connected to every neuron in the previous layer. The term "dense" is often used interchangeably with "hidden" to describe layers where all neurons are connected to every neuron in the previous layer.

### Output Layer
The output layer is the final layer of the neural network architecture. It produces the network's output predictions based on the learned representations from the hidden layers. Each neuron in the output layer typically represents a class label in classification tasks or a continuous value in regression tasks. For example, in a binary classification task, the output layer may consist of two neurons representing the probability of each class. The number of neurons in the output layer depends on the specific task at hand, with each neuron contributing to the final prediction or output of the network.


