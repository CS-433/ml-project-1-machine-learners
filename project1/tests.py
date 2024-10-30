from implementations import *
from model import *
from helpers import *
from data_preprocessing import *
import numpy as np
import argparse


def test_lr(
    learning_rates,
    x_train,
    y_train,
    lambda_=0.01,
    sampling_method="oversampling",
    prop=0.5,
):
    """
    Test the learning rates in the list learning_rates
    """
    f_1 = []
    for lr in learning_rates:
        f_1.append(
            cross_validation(
                y_train,
                x_train,
                5,
                seed=1,
                lr=lr,
                lambda_=lambda_,
                sampling_method=sampling_method,
                prop=prop,
            )
        )

    best = np.argmax(f_1)
    print(
        "The learning rate leading to the best f1-score is ",
        learning_rates[best],
        "with a f1-score of ",
        f_1[best],
    )
    return learning_rates[best]


def test_lambdas(
    lambdas, x_train, y_train, lr=0.01, sampling_method="oversampling", prop=0.5
):
    """
    Test the lambdas in the list lambdas
    """
    f_1 = []
    for lambda_ in lambdas:
        f_1.append(
            cross_validation(
                y_train,
                x_train,
                5,
                seed=1,
                lambda_=lambda_,
                lr=lr,
                sampling_method=sampling_method,
                prop=prop,
            )
        )
    best = np.argmax(f_1)

    print(
        "The lambda leading to the best f1-score is ",
        lambdas[best],
        "with a f1-score of ",
        f_1[best],
    )
    return lambdas[best]


def test_over_undersample(x_train, y_train, prop, lr=0.1, lambda_=0.01):
    """
    Test the oversampling and undersampling methods
    """

    # Test  for oversampling
    f1_oversampling = []
    for proportion in prop:
        f1_oversampling.append(
            cross_validation(
                y_train,
                x_train,
                5,
                seed=1,
                lr=lr,
                lambda_=lambda_,
                sampling_method="oversampling",
                prop=proportion,
            )
        )
    best_oversampling = np.argmax(f1_oversampling)
    print(
        "With oversampling the best proportion is ",
        prop[best_oversampling],
        "leading to a f1-score of ",
        f1_oversampling[best_oversampling],
    )

    # Test for undersampling
    f1_undersampling = []
    for proportion in prop:
        f1_undersampling.append(
            cross_validation(
                y_train,
                x_train,
                5,
                seed=1,
                lr=lr,
                lambda_=lambda_,
                sampling_method="undersampling",
                prop=proportion,
            )
        )
    best_undersampling = np.argmax(f1_undersampling)
    print(
        "With undersampling the best proportion is ",
        prop[best_undersampling],
        "leading to a f1-score of ",
        f1_undersampling[best_undersampling],
    )

    best = np.argmax(
        [f1_oversampling[best_oversampling], f1_undersampling[best_undersampling]]
    )
    if best == 0:
        print(
            "The best method is oversampling with a proportion of ",
            prop[best_oversampling],
        )
        return "oversampling", prop[best_oversampling]
    else:
        print(
            "The best method is undersampling with a proportion of ",
            prop[best_undersampling],
        )
        return "undersampling", prop[best_undersampling]


if __name__ == "__main__":

    # Load and preprocess the data
    x_train, x_test, y_train, train_ids, test_ids = preprocess_data()

    # Select the features
    x_train, x_test = select_features(x_train, y_train, x_test)

    # Test the sampling method :

    prop = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sampling_method, proportion = test_over_undersample(x_train, y_train, prop)

    # Test the learning rate

    learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr = test_lr(
        learning_rates,
        x_train,
        y_train,
        lambda_=0.01,
        sampling_method=sampling_method,
        prop=proportion,
    )

    # Test the lambda

    lambdas = [0.1, 0.05, 0.01, 0.005, 0.001]
    lambda_ = test_lambdas(
        lambdas,
        x_train,
        y_train,
        lr=lr,
        sampling_method=sampling_method,
        prop=proportion,
    )
