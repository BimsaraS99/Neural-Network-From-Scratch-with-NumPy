# Neural Network from Scratch for Digit Recognition

This repository contains the implementation of a simple neural network from scratch using NumPy for digit recognition. The neural network architecture consists of one hidden layer with ReLU activation and an output layer with softmax activation for multi-class classification.

## Dataset

The code is designed to work with the MNIST dataset, a collection of grayscale images of handwritten digits (0 to 9).

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the Code:**
   - Execute the Jupyter Notebook or Python script containing the neural network implementation.
   ```bash
   jupyter notebook ann_no_packages.ipynb
   ```
   or
   ```bash
   python neural_network_script.py
   ```

## Implementation Details

- **Data Preprocessing:**
  - Features are normalized using min-max scaling.

- **Initialization:**
  - Weights are initialized using a normal distribution.

- **Activation Functions:**
  - ReLU activation is used for the hidden layer.
  - Softmax activation is used for the output layer.

- **Loss Function:**
  - MSE loss is employed, suitable for multi-class classification.

- **Training:**
  - Gradient descent is applied to update parameters.
  - Accuracy is monitored throughout training.

## Suggestions for Improvement

- Experiment with different neural network architectures.
- Adjust hyperparameters, including learning rate, iterations, and batch size.
- Consider exploring alternative optimization algorithms.

## Visualization

The repository includes a script for plotting accuracy over iterations, providing a visual representation of the training progress.

