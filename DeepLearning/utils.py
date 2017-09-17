#coding:utf8
import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)

    return v


# GRADED FUNCTION: normalizeRows

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    # keepdims is keep the np.dims, in this case, [2, 3] will become [2, 1], but not [2, ]
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    # x_norm = np.linalg.norm(x, ord = 2, axis = 1)
    # print(x_norm.shape)
    # Divide x by its norm.
    x = x / x_norm
    return x


# GRADED FUNCTION: softmax

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    # print("x_exp: {}, x_sum: {}".format(x_exp.shape, x_sum.shape))


    return s

# The loss is used to evaluate the performance of your model. The bigger your loss is,
# the more different your predictions ($ \hat{y} $) are from the true values ($y$). In deep learning,
# you use optimization algorithms like Gradient Descent to train your model and to minimize the cost.
# GRADED FUNCTION: L1
# L1正则
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(yhat - y), axis=0)
    ### END CODE HERE ###

    return loss


# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.dot(abs(y - yhat), abs(y - yhat).T)
    # loss = np.square(np.linalg.norm(abs(y - yhat), ord = 2, axis = 0))
    return loss