import numpy as np
from implementations import *
import eval

###### TRAINING THE MODEL #########


def split_data(x, y, val_size, seed=1):
    """
    Split the dataset into training and validation sets

    Args:
        x: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        val_size: scalar, the proportion of the training set.
        seed: scalar, the random seed

    Returns:
        x_train: numpy array of shape (N_train,D), the training set
        x_validate: numpy array of shape (N_test,D), the validation set
        y_train: numpy array of shape (N_train,), the labels of the training set
        y_validate: numpy array of shape (N_test,), the labels of the validation set
    """
    ids = np.arange(x.shape[0])
    np.random.shuffle(ids)
    split_index = int(val_size * len(ids))
    train_ids = ids[split_index:]
    validate_ids = ids[:split_index]
    x_train = x[train_ids, :]
    y_train = y[train_ids]
    x_validate = x[validate_ids, :]
    y_validate = y[validate_ids]
    return x_train, x_validate, y_train, y_validate


def oversampling(x, y, proportion):
    """
    Oversample the minority class

    Args :
        x: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        proportion: scalar, the proportion of the minority class in the new dataset

    Returns :
        X_oversample: numpy array of shape (N,D), the oversampled dataset
        y_oversample: numpy array of shape (N,), the oversampled labels
    """

    class_0_id = np.where(y == 0)[0]
    class_1_id = np.where(y == 1)[0]
    resample_id = np.random.choice(
        class_1_id,
        size=int(len(class_0_id) * proportion / (1 - proportion)),
        replace=True,
    )
    X_oversample = np.concatenate((x[class_0_id], x[resample_id]), axis=0)
    y_oversample = np.concatenate((y[class_0_id], y[resample_id]), axis=0)
    return X_oversample, y_oversample


def undersampling(x, y, proportion):
    """
    Undersample the majority class

    Args :
        x: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        proportion: scalar, the proportion of the minority class in the new dataset

    Returns :

        X_oversample: numpy array of shape (N,D), the undersampled dataset
        y_oversample: numpy array of shape (N,), the undersampled labels
    """

    class_0_id = np.where(y == 0)[0]
    class_1_id = np.where(y == 1)[0]
    resample_id = np.random.choice(
        class_0_id,
        size=int(len(class_1_id) * (1 - proportion) / proportion),
        replace=False,
    )
    X_undersample = np.concatenate((x[class_0_id], x[resample_id]), axis=0)
    y_undersample = np.concatenate((y[class_0_id], y[resample_id]), axis=0)
    return X_undersample, y_undersample


def undersample(x, y, undersampling_ratio):
    healthy_samples_mask = np.where(y == 0)[0]
    unhealthy_samples_mask = np.where(y == 1)[0]

    num_observations = len(healthy_samples_mask) // undersampling_ratio

    random_healthy = np.random.choice(
        healthy_samples_mask, num_observations, replace=False
    )
    indexes_kept = np.concat((random_healthy, unhealthy_samples_mask))

    return x[indexes_kept], y[indexes_kept]


# def train(x, y, lr, max_iters, lambda_):
#     """
#     Train the model

#     Args  :
#         x: numpy array of shape (N,D), D is the number of features.
#         y: numpy array of shape (N,), N is the number of samples.
#         lr: scalar, learning rate.
#         max_iters: scalar, number of iterations.
#         lambda_: scalar, regularization parameter.

#     Returns :
#         w: optimal weights, numpy array of shape(D,), D is the number of features.
#         loss: scalar
#     """
#     w_0 = np.zeros(x.shape[1])
#     w,loss = reg_logistic_regression(y, x, lr, w_0, max_iters, lambda_)
#     return w,loss


def train(x_tr, y_tr, x_val, y_val, gamma, max_iters, lambda_, threshold):
    w = np.zeros(x_tr.shape[1])
    previous_loss = 100000
    for i in range(max_iters):
        loss, w = learning_by_gradient_descent(
            y_tr, x_tr, w, gamma=gamma, lambda_=lambda_
        )
        val_loss = calculate_loss(y_val, x_val, w)

        if i % 20 == 0:
            print(f"Iteration {i} - Train Loss: {loss} - Valid Loss: {val_loss}")

        if i >= 100 and previous_loss - val_loss < threshold:
            print(f"Converged in {i} iterations")
            break

        previous_loss = val_loss
    return w, loss


def predict_labels(w, x):
    """
    Predict the labels

    Args :
        w : numpy array of shape (D,), D is the number of features, weights of the model
        x: numpy array of shape (N,D), N is the number of samples.

    Returns :
        y_pred : numpy array of shape (N,D), the predicted labels

    """
    p = [sigmoid(x[i].T @ w) for i in range(x.shape[0])]
    y_pred = np.array([1 if p[i] > 0.5 else 0 for i in range(len(p))])
    return y_pred


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    Args:
        y : numpy array of shape (N,), N is the number of samples.
        k_fold: int, the fold number
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_k(
    y,
    x,
    k_indices,
    k,
    lambda_=0.001,
    lr=0.1,
    n_epochs=1000,
    sampling_method="oversampling",
    prop=0.5,
):
    """
    Return the f1 score for a fold corresponding to k_indices

    Args:
        y : numpy array of shape (N,), N is the number of samples.
        x: numpy array of shape (N,D), D is the number of features.
        k_indices:  2D array returned by build_k_indices()
        k: scalar, the k-th fold
        lambda_: scalar, regularization parameter.
        lr: scalar, learning rate.
        n_epochs: scalar, number of iterations.
        sampling_method: string, the sampling method to use (oversampling or undersampling)
        prop: scalar, the proportion of the minority class in the new dataset

    Returns:
        F-1 score over the test set for the k-fold

    """
    # get k'th subgroup in test, others in train
    id_train = np.delete(np.arange(x.shape[0]), k_indices[k])
    x_train = x[id_train]
    y_train = y[id_train]
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]

    w, loss = train(x_train, y_train, lr, n_epochs, lambda_)
    y_pred = predict_labels(w, x_test)
    f1 = eval.f1_score(y_pred, y_test)
    return f1


def cross_validation(
    y,
    x,
    k_fold,
    seed=1,
    lambda_=0.001,
    lr=0.1,
    n_epochs=1000,
    sampling_method="oversampling",
    prop=0.5,
):
    """
    return mean F-1 score over the k folds

    Args:

        y : numpy array of shape (N,), N is the number of samples.
        x : numpy array of shape (N,D), D is the number of features.
        k_fold : scalar, the number of folds
        seed : scalar, the random seed
        lambda_: scalar, regularization parameter.
        lr: scalar, learning rate.
        n_epochs: scalar, number of iterations.
        sampling_method: string, the sampling method to use (oversampling or undersampling)
        prop: scalar, the proportion of the minority class in the new dataset

    Returns:
        Mean F-1 score ove the k folds
    """
    k_indices = build_k_indices(y, k_fold, seed)
    f1 = 0
    for k in range(k_fold):
        f1 += cross_validation_k(
            y, x, k_indices, k, lambda_, lr, n_epochs, sampling_method, prop
        )
    return float(f1 / k_fold)
