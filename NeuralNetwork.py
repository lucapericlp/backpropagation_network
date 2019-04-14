import numpy as np
from sigmoids import sigmoid, sigmoidPrime

class NeuralNetwork(object):
    def __init__(self,inputSize,outputSize,hiddenSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.firstWeightsMatrix = np.random.randn(self.inputSize, self.hiddenSize) #randomly instantiate a 3x3 matrix
        self.secondWeightsMatrix = np.random.randn(self.hiddenSize, self.outputSize) #randomly instantiate a 3x1 matrix

    def forward(self, X):
        '''
        Performs forward propagation by calculating dot product of inputs * firstWeightsMatrix and then the sigmoid activation,
        then the matrix output of the sigmoid is dot product'd with secondWeightsMatrix which is then also 
        passed through the sigmoid activation. This is returned so that we can calculate the output error and perform
        backpropagation to adjust and improve the weights for our desired outcomes.
        '''
        self.net_h = np.dot(X, self.firstWeightsMatrix)  
        self.a_h = sigmoid(self.net_h)
        self.output_net = np.dot(self.a_h, self.secondWeightsMatrix)  
        output_a = sigmoid(self.output_net)
        return output_a

    def backward(self, X, t, a, lr):
        # taking common notation within cs academia.
        # the errors on output
        self.o_error = t - a  # (t-a)
        self.epsilon = self.o_error * sigmoidPrime(a) # (t-a) * a * (1-a)

        # the error of hidden units
        self.a_h_error = self.epsilon.dot(self.secondWeightsMatrix.T) # epsilon * w_h
        self.a_h_delta = self.a_h_error * sigmoidPrime(self.a_h)  # epsilon * a_h (1 - a_h) * w_h - the error for a hidden unit h

        # the weight changes
        self.firstWeightsMatrix += lr * (X.T.dot(self.a_h_delta)) #the change of weight from input i to hidden h
        self.secondWeightsMatrix += lr * (self.a_h.T.dot(self.epsilon)) #the change of weight from hidden unit h
        # return epsilon to show output loss
        return self.epsilon

    def executeTrain(self, X, t, lr):
        a = self.forward(X)
        epsilon = self.backward(X, t, a, lr)
        return a,epsilon

    def infer(self,X,unique_classes):
        activations = self.forward(X)
        activations_classes = []
        for index,activation in enumerate(activations):
            smallestDistance = 1;
            chosenClass = None
            for a_class in unique_classes:
                distance = abs(activation - a_class)
                if distance < smallestDistance:
                    smallestDistance = distance
                    chosenClass = a_class
            activations_classes.append([index,chosenClass])
        return activations_classes

    def train(self,epochs,X,t,lr,verbose):
        for i in range(epochs+1):
            a,epsilon = self.executeTrain(X, t, lr)
            if i%100 == 0 and verbose:
                print("Progress: {}/{}".format(i,epochs))
                print("Input: \n" + str(X))
                print("Actual Output: \n" + str(t))
                print("Predicted Output: \n" + str(a))
                print("Loss: \n{}".format(str(epsilon)))
                print("\n")





