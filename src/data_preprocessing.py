import json

import numpy as np

from src import config, feature_type_detection, data_cleaning, feature_engineering


def load_original_dataset(data_dir):
    """
    Loads the original training and testing datasets from specified directory.

    Parameters:
    data_dir (str): Path to the directory containing dataset files. The function
                    expects the directory to contain:
                    - 'train_dataset.npy' for training data features.
                    - 'test_dataset.npy' for testing data features.
                    - 'train_targets.npy' for training data labels.

    Returns:
    tuple: A tuple containing:
           - x_train (np.ndarray): Array of training data features.
           - x_test (np.ndarray): Array of testing data features.
           - y_train (np.ndarray): Array of training data labels.
    """
    print("Pipeline Stage 1 - Loading Datasets...")
    x_train = np.load(f"{data_dir}/train_dataset.npy")
    x_test = np.load(f"{data_dir}/test_dataset.npy")
    y_train = np.load(f"{data_dir}/train_targets.npy")
    return x_train, x_test, y_train


def convert_array_to_dict(x, feature_names):
    """Converts a 2D numpy array into a dict of 1D numpy arrays.

    Args:

    Returns

    """
    return {feature: x[:, ind] for ind, feature in enumerate(feature_names)}


def convert_dict_to_array(dict_dataset):
    """Converts a dict of 1D numpy arrays into a 2D array
    Args:

    Returns:
    """
    return np.stack(list(dict_dataset.values()), axis=1)


def preprocess_data(data_dir: str):
    """This function acts as a pipeline that performs data cleaning, feature selection and standardization,
    among other transformations.

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

    # x_train, x_test = merge_landline_cellphone_features(x_train, x_test, feat_indexes) # TODO
    train_dataset, test_dataset = data_cleaning.drop_useless_features(
        train_dataset, test_dataset
    )
    train_dataset, test_dataset = data_cleaning.replace_weird_values(
        train_dataset, test_dataset, config.ABNORMAL_FEATURE_VALUES
    )
    feat_types = feature_type_detection.detect_feature_types(train_dataset)
    train_dataset, test_dataset = data_cleaning.fill_nans(
        train_dataset, test_dataset, feat_types
    )
    # train_dataset, test_dataset, feature_names, feat_indexes = one_hot_categoricals(train_dataset, test_dataset, feature_names, feat_indexes, feat_types) # why commented

    # Convert the datasets back to a numpy array so we can apply operations over all columns in parallel
    x_train = convert_dict_to_array(train_dataset)
    x_test = convert_dict_to_array(test_dataset)

    # Delete the auxiliary structures so we free memory
    del train_dataset, test_dataset

    # Standardize features so we are dealing with variables in the same scale
    x_train, x_test = feature_engineering.standardize(x_train, x_test)

    # Analyze correlations between features in order to remove redundant data and help with explainability
    x_train, x_test = feature_engineering.select_features(x_train, y_train, x_test)

    # Add bias feature to the dataset so our log regression model is complete
    x_train, x_test = feature_engineering.add_bias_feature(x_train, x_test)

    # Convert the targets into 1 or 0 so we can apply the lab loss function
    y_train = (y_train + 1) / 2

    train_ids = np.arange(x_train.shape[0])
    test_ids = np.arange(x_test.shape[0]) + x_train.shape[0]
    return x_train, x_test, y_train, train_ids, test_ids
