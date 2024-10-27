
import numpy as np


def mse(y,tx,w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N=len(y)
    e = y - tx@w
    L = e@e.T/(2*N)
    return L

def mae(y,tx,w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N=len(y)
    L = np.sum(np.abs(y - tx@w))/N
    return L

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N=len(y)
    e = y - tx@w
    grad = -tx.T@e/N
    return grad

def compute_stoch_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx@w
    grad = -tx.T@e
    return grad

def mean_squared_error_gd (y, tx, initial_w,max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient 
        grad = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma*grad

    loss = mse(y, tx, w)
    return w,loss

def mean_squared_error_sgd (y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
    """

    w = initial_w
    N = len(y)

    for n_iter in range(max_iters):
        n = np.random.randint(N)
        grad = compute_stoch_gradient(y[n], tx[n], w)
        w = w-gamma*grad
    loss = mse(y, tx, w)
    return w,loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    w = np.linalg.solve(tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    loss = mse(y,tx,w)
    return w, loss

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    sig = sigmoid(tx @ w)
    eps = 1e-8
    # Add eps to ensure that we dont have log(0)
    return float(np.sum(y * np.log(sig + eps) + (1 - y)*np.log(1 - sig + eps)) / -y.shape[0])

def calculate_gradient(y, tx, w, lambda_):
    sig = sigmoid(tx @ w)
    return (tx.T @ (sig - y) / y.shape[0]) + 2*lambda_*w

def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    return calculate_loss(y, tx, w), w - gamma * calculate_gradient(y, tx, w, lambda_)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using gradient descent or SGD.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), initial guess for the model parameters.
        max_iters: scalar, number of iterations.
        gamma: scalar, stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w
    N = y.shape[0]
    for n_iter in range(max_iters):
        grad = tx.T@(sigmoid(tx@w)-y)/N
        w = w - gamma*grad
    loss = np.mean([-y[i]*tx[i].T@w+ np.log(1+np.exp(tx[i].T@w)) for i in range(N)])
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement ridge logistic regression using gradient descent or SGD.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization parameter.
        initial_w: numpy array of shape (D,), initial guess for the model parameters.
        max_iters: scalar, number of iterations.
        gamma: scalar, stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w
    N = y.shape[0]
    for n_iter in range(max_iters):
        grad = tx.T@(sigmoid(tx@w)-y)/N + 2*lambda_*w
        w = w - gamma*grad
    loss = np.mean([-y[i]*tx[i].T@w+ np.log(1+np.exp(tx[i].T@w)) for i in range(N)])
    return w, loss

# def learning_by_gradient_descent(y, tx, w, gamma):
#     return calculate_loss(y, tx, w), w - gamma * calculate_gradient(y, tx, w)

# def calculate_weighted_loss(y, tx, w, sample_weights):
#     sig = sigmoid(tx @ w)
#     return (np.sum(sample_weights * (y * np.log(sig + 0.00001) + (1 - y)*np.log(1 - sig + 0.00001))) / -np.sum(sample_weights))

# def calculate_weighted_gradient(y, tx, w, sample_weights):
#     sig = sigmoid(tx @ w)
#     return tx.T @ (sample_weights * (sig - y)) / np.sum(sample_weights)

# def learning_by_weighted_gradient_descent(y, tx, w, gamma, sample_weights):
#     loss = calculate_weighted_loss(y, tx, w, sample_weights)
#     return loss, w - gamma * calculate_weighted_gradient(y, tx, w, sample_weights)