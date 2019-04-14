# Backpropagation NN
Using numpy, we can train a basic neural network of an arbitrary size to correctly classify a test dataset according to either 2, 3 or 4 classes (depending which input dataset is selected, default is 4 classes). Please see files for comments.

Entry point: [main.py](main.py)

# Instructions
Select an input dataset of X number of columns which will scale the neural net input and hidden layers to match. Choose a learning rate and a number of epochs for which the network will train. Inference is then performed on a testset of same size and the inferred classes can be returned using NN.infer().