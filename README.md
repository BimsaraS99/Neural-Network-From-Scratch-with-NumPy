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
![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/0b2a15b3-d19f-46d6-bfac-d15b716bafe6)


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

## Data and Dataset

In the project focused on digit classification, the dataset typically comprises handwritten digits, each represented as a 28x28 pixel image. These images are grayscale, meaning each pixel's intensity ranges from 0 (black) to 255 (white).

The dataset usually contains a large number of such images, with each image labeled with the corresponding digit it represents (0 to 9). This type of dataset is commonly used for multiclass classification tasks, where the goal is to train a model to correctly classify each image into one of the ten possible digit classes.

Each image in the dataset serves as a data point, with its pixel values serving as features. In total, there are 784 features (28x28) representing each image. These features capture various aspects of the handwritten digits, such as the curvature of strokes, the thickness of lines, and the overall shape of the digit.

The dataset is typically divided into training and testing sets, where the training set is used to train the model, and the testing set is used to evaluate its performance on unseen data. Additionally, preprocessing techniques such as normalization or standardization may be applied to the pixel values to enhance model training and generalization.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/0fe58ac3-4324-4618-9d36-dc1d03f96ee0)


## Training Loop

The training loop is a fundamental component of training a machine learning model, including neural networks. It's essentially a process where the model iteratively learns from the training data to improve its performance over time. The training loop typically consists of the following steps:

1. **Initialization:** 

2. **Forward Pass:** 

3. **Loss Computation:** 

4. **Backpropagation:**

5. **Parameter Update:**

6. **Repeat:**

In summary:

- **Learning rate** is a hyperparameter controlling the step size of parameter updates during training. It affects the convergence speed and stability of the optimization process.
- **Epochs** represent the number of times the entire training dataset is passed forward and backward through the neural network during training. They help the model learn from the data and refine its parameters

Learning rate and epochs are essential hyperparameters that need to be carefully chosen to ensure effective training and optimal performance of the machine learning or deep learning model. Adjusting these hyperparameters can significantly impact the training process and the model's ability to generalize to unseen data.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/3d1e1517-e489-489f-a535-0a8e6b0a50ca)



### 1. Initialization

Initialize the model parameters, such as weights and biases, usually randomly or with predefined values. Random initialization from a uniform or normal distribution with small values (e.g., between -0.1 and 0.1 or -0.01 and 0.01).

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/b6a94443-9b64-40f3-8905-afcd36dfefdd)

### 2. Forward Pass

Feed the input data through the model to generate predictions. This step involves applying the model's forward computation, which consists of matrix multiplications and activation functions applied to the input data.

- Input data moves through the network layer by layer, undergoing transformations via weighted connections and activation functions.
- At each layer, a weighted sum of inputs is computed, followed by the application of an activation function to introduce non-linearity.
- The output of each layer becomes the input to the next layer, culminating in the final predictions or outputs of the network.
- The forward pass is a critical step in the network's operation, converting input data into meaningful predictions or outputs.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/b9d6f521-28aa-4b73-a93d-3f2a913d80ef)

### 3. Loss Calcultaions

A loss function is applied to quantify the discrepancy between the predictions and the actual targets. Common loss functions include mean squared error (MSE) for regression tasks and categorical cross-entropy for classification tasks.
The loss function calculates a single scalar value representing the overall error or discrepancy between the predictions and the actual targets.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/bf6a16d6-b820-4332-acbf-fd3777ad225f)

### 4. Backpropagation

In the backward pass, the gradient of the loss function with respect to each parameter (weight and bias) in the network is calculated using the chain rule of calculus. The gradients are computed layer by layer, starting from the output layer and moving backward towards the input layer.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/dd722d66-74cf-43c9-bc8d-aaf5aab23bdf)

### 5. Parameter Update

Once the gradients are computed, the parameters of the network (weights and biases) are updated using an optimization algorithm such as gradient descent. The updates are made in the opposite direction of the gradient to minimize the loss function.

![image](https://github.com/BimsaraS99/Neural-Network-From-Scratch-with-NumPy/assets/107334404/6401b076-3cd2-4e39-831b-a9d9762eb7e0)

### 6. Repeat

Repeat steps 2-5 for a fixed number of iterations (epochs) or until a convergence criterion is met. This process allows the model to iteratively learn from the training data, gradually improving its performance.


