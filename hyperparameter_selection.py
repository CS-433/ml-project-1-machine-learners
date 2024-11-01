import itertools

import numpy as np

from src import model


def compute_best_hyperparams(x_train: np.ndarray, y_train: np.ndarray, hyperparams_grid: list) -> None:
    """
    Test the oversampling and undersampling methods
    """
    # Test for undersampling
    f1_undersampling = []
    for proportion, lr, lambda_ in hyperparams_grid:
        print(
            f"Hyperparams: Undersampling proportion: {proportion}, lr: {lr}, lambda: {lambda_}"
        )
        mean_f1_score = model.cross_validation(
            y=y_train,
            x=x_train,
            k_fold=5,
            lr=lr,
            lambda_=lambda_,
            prop=proportion,
        )
        print(f"mean f1 score: {mean_f1_score}")
        f1_undersampling.append(mean_f1_score)
    best_undersampling = np.argmax(f1_undersampling)
    print(
        "The optimal set of hyperparams is:",
        hyperparams_grid[best_undersampling],
        "leading to a f1-score of ",
        f1_undersampling[best_undersampling],
    )


if __name__ == "__main__":
    # Set random seed to ensure that results are deterministic
    np.random.seed(42)

    # Load already processed datasets
    x_train = np.load("data/processed_x_train.npy")
    x_test = np.load("data/processed_x_test.npy")
    y_train = np.load("data/processed_y_train.npy")

    # Test the sampling method :
    prop = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    learning_rates = [0.1, 0.15, 0.2]
    lambdas = [0.1, 0.05, 0.01, 0.005, 0]
    hyperparams_grid = list(itertools.product(prop, learning_rates, lambdas))
    compute_best_hyperparams(x_train, y_train, hyperparams_grid)
