import numpy as np
from implementations import *


###### FEATURES SELECTION ########


def list_features(x_train, y_train):
    """returns the list of features selected

    Args :
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of samples.

    Returns:

    list_features: index of selected features
    """

    index_features = np.arange(x_train.shape[1])
    nb_features = len(x_train.shape[1])

    # We remove the features that are redundant
    correlation_matrix = np.corrcoef(x_train, rowvar=False)
    list_suppr = []
    for i in range(nb_features):
        for j in range(i + 1, nb_features):
            if np.abs(correlation_matrix[i, j]) > 0.9:
                corr_i = np.corrcoef(x_train[:, i], y_train)
                corr_j = np.corrcoef(x_train[:, j], y_train)
                if np.abs(corr_i[0, 1]) < np.abs(corr_j[0, 1]):
                    list_suppr.append(i)
                else:
                    list_suppr.append(j)
    x = x_train.copy()
    x = np.delete(x, list_suppr, axis=1)

    index_features = np.delete(index_features, list_suppr)
    # We keep only the features that have a correlation > 0.1 with the target
    list_features = []
    for i in range(x.shape[1]):
        corr = np.corrcoef(x[:, i], y_train)
        if np.abs(corr[0, 1]) > 0.1:
            list_features.append(i)
    list_features = index_features[list_features]
    return list_features


def select_features(x_train, y_train, x_test):
    """returns the selected features for the train and test set

    Args :
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of training samples.
    x_test: numpy array of shape (n,D), n is the number of test samples.

    Returns:

    x_train: numpy array of shape (N,d), d is the number of selected features.
    x_test: numpy array of shape (N,d), d is the number of selected features.
    """
    list_features = list_features(x_train, y_train)
    x_train = x_train[:, list_features]
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    x_test = x_test[:, list_features]
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    return x_train, x_test


###### TRAINING THE MODEL #########


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
        class_1_id, size=int(len(class_0_id) * proportion), replace=True
    )
    X_oversample = np.concatenate((x[class_0_id], x[resample_id]), axis=0)
    y_oversample = np.concatenate((y[class_0_id], y[resample_id]), axis=0)
    return X_oversample, y_oversample


def undersampling(x, y, prop):
    """
    Undersample the majority class

    Args :
        x: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        prop: scalar, the proportion of the majority class in the new dataset

    Returns :

        X_oversample: numpy array of shape (N,D), the undersampled dataset
        y_oversample: numpy array of shape (N,), the undersampled labels
    """

    class_0_id = np.where(y == 0)[0]
    class_1_id = np.where(y == 1)[0]
    resample_id = np.random.choice(
        class_0_id, size=int((1 + prop) * len(class_1_id)), replace=False
    )
    X_undersample = np.concatenate((x[class_0_id], x[resample_id]), axis=0)
    y_undersample = np.concatenate((y[class_0_id], y[resample_id]), axis=0)
    return X_undersample, y_undersample


def train(x, y, gamma, max_iters, lambda_):
    """
    Train the model

    Args  :
        x: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        gamma: scalar, stepsize
        max_iters: scalar, number of iterations.
        lambda_: scalar, regularization parameter.

    Returns :
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar
    """
    w_0 = np.zeros(x.shape[1])
    w, loss = reg_logistic_regression(y, x, gamma, w_0, max_iters, lambda_)
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


###### TESTING THE MODEL #########
def f1_score(y_true, y_pred):
    """
    Compute the F-1 score

    Args :
        y_true: numpy array of shape (N,), N is the number of samples.
        y_pred: numpy array of shape (N,), predicted labels.

    Returns :
        f1: float, F1-score.
    """
    # Compute True Positives (TP), False Positives (FP), et False Negatives (FN)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Calculer la précision et le rappel
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculer le F1-score
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


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


def cross_validation_k(y, x, k_indices, k, lambda_=0.001, lr=0.1, n_epochs=1000):
    """
    Return the f1 score for a fold corresponding to k_indices

    Args:
        y : numpy array of shape (N,), N is the number of samples.
        x: numpy array of shape (N,D), D is the number of features.
        k_indices:  2D array returned by build_k_indices()
        k: scalar, the k-th fold
        lambda_: scalar, regularization parameter.

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
    f1 = f1_score(y_pred, y_test)
    return f1


def cross_validation(y, x, k_fold, seed=1, lambda_=0.001, lr=0.1, n_epochs=1000):
    """
    return mean F-1 score over the k folds

    Args:

        y : numpy array of shape (N,), N is the number of samples.
        x : numpy array of shape (N,D), D is the number of features.
        k_fold : scalar, the number of folds
        seed : scalar, the random seed
        lambda_: scalar, regularization parameter.

    Returns:
        Mean F-1 score ove the k folds
    """
    k_indices = build_k_indices(y, k_fold, seed)
    f1 = 0
    for k in range(k_fold):
        f1 += cross_validation_k(y, x, k_indices, k, lambda_, lr, n_epochs)
    return float(f1 / k_fold)
