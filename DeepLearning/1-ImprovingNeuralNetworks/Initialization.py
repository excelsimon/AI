#coding:utf8
import numpy as np
"""
- Understand that different regularization methods that could help your model.

- Implement dropout and see it work on data.

- Recognize that a model without regularization gives you a better accuracy on the training set but nor necessarily on the test set.

- Understand that you could use both dropout and regularization on your model.
"""

"""
A well chosen initialization can:
Speed up the convergence of gradient descent
Increase the odds of gradient descent converging to a lower training (and generalization) error
"""

"""
一、Zero initialization
n general, initializing all the weights to zero results in the network failing to break symmetry. 
This means that every neuron in each layer will learn the same thing, and you might as well be training 
a neural network with  n[l]=1n[l]=1  for every layer, and the network is no more powerful than a linear 
classifier such as logistic regression.
What you should remember:
   The weights W[l]W[l] should be initialized randomly to break symmetry.
   It is however okay to initialize the biases b[l]b[l] to zeros. Symmetry is still broken so long as 
   W[l]W[l] is initialized randomly.

"""


# GRADED FUNCTION: initialize_parameters_zeros

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
    return parameters

"""
二、Random initialization
To break symmetry, lets intialize the weights randomly. Following random initialization, each neuron can 
then proceed to learn a different function of its inputs. 
In this exercise, you will see what happens if the weights are intialized randomly, but to very large values.

If you see "inf" as the cost after the iteration 0, this is because of numerical roundoff; 
a more numerically sophisticated implementation would fix this. But this isn't worth worrying about 
for our purposes.
Anyway, it looks like you have broken symmetry, and this gives better results. than before. 
The model is no longer outputting all 0s.


"""


def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)  # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters


"""
三、He initialization
Finally, try "He Initialization"; this is named for the first author of He et al., 2015. 
(If you have heard of "Xavier initialization", this is similar except Xavier initialization uses 
a scaling factor for the weights  W[l]W[l]  of sqrt(1./layers_dims[l-1]) where He initialization would use 
sqrt(2./layers_dims[l-1]).)
Exercise: Implement the following function to initialize your parameters with He initialization.
Hint: This function is similar to the previous initialize_parameters_random(...). The only difference is that 
instead of multiplying np.random.randn(..,..) by 10, you will multiply it by  2dimension of the previous
layer⎯⎯⎯⎯⎯√2dimension of the previous layer , which is what He initialization
recommends for layers with a ReLU activation.
"""


# GRADED FUNCTION: initialize_parameters_he

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.multiply(np.random.randn(layers_dims[l], layers_dims[l - 1]),
                                               np.sqrt(2 / layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters



"""
You have seen three different types of initializations. For the same number of iterations and same 
hyper parameters the comparison is:
Model	Train accuracy	Problem/Comment
3-layer NN with zeros initialization	50%	fails to break symmetry
3-layer NN with large random initialization	83%	too large weights
3-layer NN with He initialization	99%	recommended method
What you should remember from this notebook:
Different initializations lead to different results
Random initialization is used to break symmetry and make sure different hidden units can learn different things
Don't intialize to values that are too large
He initialization works well for networks with ReLU activations.
"""























