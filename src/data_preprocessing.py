import json

import numpy as np

from src import (
    config, 
    feature_type_detection, 
    data_cleaning, 
    feature_engineering
)

def load_original_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the original training and testing datasets from specified directory.

    Parameters:
    data_dir: Path to the directory containing dataset files. The function
              expects the directory to contain:
                - 'train_dataset.npz' for training data features.
                - 'test_dataset.npz' for testing data features.
                - 'train_targets.npz' for training data labels.

    Returns:
        x_train: Array of training data features.
        x_test: Array of testing data features.
        y_train: Array of training data labels.
    """
    print("Pipeline Stage 1 - Loading Datasets...")
    x_train = np.load(f"{data_dir}/train_dataset.npz")["arr_0"]
    x_test = np.load(f"{data_dir}/test_dataset.npz")["arr_0"]
    y_train = np.load(f"{data_dir}/train_targets.npz")["arr_0"]
    return x_train, x_test, y_train


def convert_array_to_dict(
    x: np.ndarray, feature_names: list[str]
) -> dict[str, np.ndarray]:
    """Converts a 2D numpy array into a dict of 1D numpy arrays.

    Args:
        x: numpy array of shape (N, D) containing the dataset
        feature_names: list that contains the name of every feature
        in the dataset, the index of each feature in the list
        corresponds to the same index in x.

    Returns
        x_dataset: Returns a dictionary containing feature names as
        keys and their respective array of values as values

        The dict is in the following form:
        {
            "feature_names[0]: x[:, 0],
            ...
        }
    """
    return {feature: x[:, ind] for ind, feature in enumerate(feature_names)}


def convert_dict_to_array(dict_dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Converts a dict of 1D numpy arrays into a 2D array

    This method reverses the conversion from 'convert_array_to_dict'
    after having done the necessary transformations

    Args
        x_dataset: A dictionary containing feature names as
        keys and their respective array of values as values

        The dict is in the following form:
        {
            "feature_names[0]: x[:, 0],
            ...
        }

    Returns:
        x: numpy array of shape (N, D) containing the concatenation
        of all the arrays inside the dictionary.
    """
    return np.stack(list(dict_dataset.values()), axis=1)


def preprocess_data(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function acts as a pipeline that performs data cleaning,
    feature selection and standardization, among other transformations.

    The resulting dataset is ready for modelling.
    Returns:
    x_train: numpy array of shape (N,d), d is the number of selected features.
    x_test: numpy array of shape (N,d), d is the number of selected features.
    y_train: numpy array of shape (N, 1), containing the labels in the [0, 1] range.
    """
    # Load the original datasets in .npy format (much faster)
    x_train, x_test, y_train = load_original_dataset(data_dir)

    # Convert each dataset to a dict of arrays so we can manipulate features by name
    # This is similar to a dataframe in Pandas, but less efficient
    train_dataset = convert_array_to_dict(x_train, config.FEATURE_NAMES)
    test_dataset = convert_array_to_dict(x_test, config.FEATURE_NAMES)

    train_dataset, test_dataset = data_cleaning.merge_landline_cellphone_features(
        train_dataset, test_dataset
    )
    train_dataset, test_dataset = data_cleaning.drop_useless_features(
        train_dataset, test_dataset, features_to_drop=config.FEATURES_TO_DROP
    )
    train_dataset, test_dataset = data_cleaning.replace_weird_values(
        train_dataset, test_dataset, config.ABNORMAL_FEATURE_VALUES
    )
    feat_types = feature_type_detection.detect_feature_types(train_dataset)
    train_dataset, test_dataset = data_cleaning.fill_nans(
        train_dataset, test_dataset, feat_types
    )

    categorical_features = [
        feature
        for feature, type in feature_type_detection.FEATURE_TYPES.items()
        if type == feature_type_detection.FeatureType.CATEGORICAL
    ]
    train_dataset, test_dataset, x_train_cat, x_test_cat = (
        feature_engineering.separate_categorical_features(
            train_dataset, test_dataset, categorical_features
        )
    )

    # Convert dataset dict back to numpy ndarray
    x_train = convert_dict_to_array(train_dataset)
    x_test = convert_dict_to_array(test_dataset)
    del train_dataset, test_dataset

    # Analyze correlations between non-cat features in order to remove
    # redundant data and help with explainability
    x_train, x_test = feature_engineering.select_features(x_train, y_train, x_test)

    # Binary encode categorical features
    x_train_cat, x_test_cat = feature_engineering.binary_encode_multiple_features(
        x_train_cat, x_test_cat
    )

    # Standardize non categorical features
    x_train, x_test = feature_engineering.standardize(x_train, x_test)

    # Add bias feature to the dataset so our log regression model is complete
    x_train, x_test = feature_engineering.add_bias_feature(x_train, x_test)

    # Add categorical and non-cat features back
    x_train = np.hstack((x_train, x_train_cat))
    x_test = np.hstack((x_test, x_test_cat))

    # Convert the targets into 1 or 0 so we can apply the lab loss function
    y_train = (y_train + 1) / 2

    train_ids = np.arange(x_train.shape[0])
    test_ids = np.arange(x_test.shape[0]) + x_train.shape[0]
    return x_train, x_test, y_train, train_ids, test_ids
