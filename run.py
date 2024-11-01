import argparse

import numpy as np

from src import config, evaluation, helpers, data_preprocessing, model


def set_execution_arguments() -> argparse.Namespace:
    """
    Sets all the possible execution arguments and retrieves their values

    This method allows the user to customize the training parameters as
    well as the seed of execution to ensure that results are deterministic.

    Returns:
        args: A namespace containing every execution argument and their
        values

    """
    parser = argparse.ArgumentParser(
        description="Training model with customizable parameters."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to ensure that results are deterministic.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.2, help="Learning rate for model training."
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Maximum number of iterations for training.",
    )
    parser.add_argument(
        "--lambda_", type=float, default=0.01, help="Regularization parameter."
    )
    parser.add_argument("--undersampling_ratio", type=float, default=0.25, help=".")
    return parser.parse_args()


if __name__ == "__main__":

    args: argparse.Namespace = set_execution_arguments()

    # Set random seed to ensure that results are deterministic
    np.random.seed(args.seed)

    try:
        # Try to load already processed datasets
        print("Attempting to read already processed datasets...")
        x_train = np.load(f"{config.DATA_FOLDER}/processed_x_train.npy")
        x_test = np.load(f"{config.DATA_FOLDER}/processed_x_test.npy")
        y_train = np.load(f"{config.DATA_FOLDER}/processed_y_train.npy")
        train_ids = np.arange(x_train.shape[0])
        test_ids = np.arange(x_test.shape[0]) + x_train.shape[0]
    except FileNotFoundError:
        print("Failed - Reading original datasets...")
        # If they do not exist, load the original datasets
        # Load and preprocess the data, ready for modelling
        x_train, x_test, y_train, train_ids, test_ids = data_preprocessing.preprocess_data(
            data_dir=config.DATA_FOLDER
        )

    # Since the healthy class is overrepresented, undersampling is a good idea
    # to improve predictions over the unhealthy class
    x_train_undersampled, y_train_undersampled = model.undersample(
        x_train, y_train, undersampling_ratio=args.undersampling_ratio
    )

    # Train our model with a validation set to prevent overfitting
    x_tr, x_val, y_tr, y_val = model.split_data(
        x_train_undersampled, y_train_undersampled, val_size=0.2
    )
    w, loss = model.train(
        x_tr,
        y_tr,
        x_val=x_val,
        y_val=y_val,
        gamma=args.gamma,
        max_iters=args.max_iters,
        lambda_=args.lambda_,
        threshold=1e-6,
    )

    # Perform predictions over the train and test datasets
    y_pred_train = model.predict_labels(w, x_train)
    y_pred = model.predict_labels(w, x_test)

    # Obtain classification metrics in order to assess the quality of our model
    evaluation.evaluate_predictions(y_train, y_pred_train)

    # Convert the labels back to their original value range (-1, 1)
    y_pred = y_pred * 2 - 1

    # Save the predictions
    helpers.create_csv_submission(test_ids, y_pred, "submission.csv")
