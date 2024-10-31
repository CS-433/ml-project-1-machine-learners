import numpy as np

from src import feature_type_detection


def separate_categorical_features(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    categorical_features: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Separate dataset between categorical features and the rest.

    This is helpful to speedup categorical encoding and numerical standardization.

    Args:
        train_dataset: Dictionary of training data features, where each key is a feature name and the value is an array of feature values.
        test_dataset: Dictionary of test data features, where each key is a feature name and the value is an array of feature values.
        categorical_features: List of the categorical features in the dataset

    Returns:
        train_dataset: Dictionary of training data features, where each key is a feature name and the value is an array of feature values.
        test_dataset: Dictionary of test data features, where each key is a feature name and the value is an array of feature values.
    """
    x_train_cat = []
    x_test_cat = []
    for feature in categorical_features:
        array = train_dataset.pop(feature, None)
        x_train_cat.append(array)
        array_te = test_dataset.pop(feature, None)
        x_test_cat.append(array_te)

    x_train_cat = np.stack(x_train_cat, axis=1)
    x_test_cat = np.stack(x_test_cat, axis=1)

    return train_dataset, test_dataset, x_train_cat, x_test_cat


def binary_encode_column(
    column: np.ndarray, unique_train_values: list[float]
) -> np.ndarray:
    """
    Encode a column with binary encoding using the unique values from the training set.

    Parameters:
        column: The column to be encoded.
        unique_train_values: The unique values from the training dataset.

    Returns:
        binary_encoded: Binary encoded matrix for the column.
    """
    # Create a mapping from unique training values to integers
    value_to_int = {value: idx for idx, value in enumerate(unique_train_values)}

    # Encode the column using the mapping, assigning unseen values to 0
    integer_encoded = np.array([value_to_int.get(value, -1) for value in column])

    # Determine the number of bits needed for binary encoding
    max_bits = len(bin(len(unique_train_values) - 1)) - 2

    # Create a binary encoded matrix
    binary_encoded = np.zeros((len(integer_encoded), max_bits), dtype=np.uint8)

    # Fill in the binary representation for known values
    for i, value in enumerate(integer_encoded):
        if value != -1:  # Only process known values
            binary_representation = np.binary_repr(value, width=max_bits)
            binary_encoded[i] = np.array(list(binary_representation), dtype=np.uint8)

    return binary_encoded


def binary_encode_multiple_features(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform binary encoding over multiple categorical columns, using training data for reference.

    Parameters:
        x_train: 2D array for training data where each column is a categorical feature.
        x_test: 2D array for test data where each column is a categorical feature.

    Returns:
        binary_encoded_train: The binary encoded version of the training input.
        binary_encoded_test: The binary encoded version of the test input.
    """
    print("Pipeline Stage 7 - Binary Encoding Categorical Features...")
    binary_encoded_train = []
    binary_encoded_test = []

    # Loop through each feature in training data and perform binary encoding
    for col in range(x_train.shape[1]):
        train_arr = x_train[:, col]
        test_arr = x_test[:, col]
        unique_values = np.unique(train_arr)

        # Encode training data
        binary_encoded_col_train = binary_encode_column(train_arr, unique_values)
        binary_encoded_train.append(binary_encoded_col_train)

        # Encode test data using the same unique values from the training set
        binary_encoded_col_test = binary_encode_column(test_arr, unique_values)
        binary_encoded_test.append(binary_encoded_col_test)

    # Concatenate all binary encoded columns horizontally for train and test
    binary_encoded_train = np.hstack(binary_encoded_train)
    binary_encoded_test = np.hstack(binary_encoded_test)

    return binary_encoded_train, binary_encoded_test


def list_features(x_train: np.ndarray, y_train: np.ndarray) -> list[int]:
    """
    Returns the list of features selected

    Args :
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of samples.

    Returns:

    list_features: list, index of selected features
    """

    index_features = np.arange(x_train.shape[1])
    nb_features = x_train.shape[1]

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
        if np.abs(corr[0, 1]) > 0:
            list_features.append(i)
    list_features = index_features[list_features]
    return list_features


def select_features(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the selected features for the train and test set

    Args :
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of training samples.
    x_test: numpy array of shape (n,D), n is the number of test samples.

    Returns:

    x_train: numpy array of shape (N,d)
    x_test: numpy array of shape (N,d), d is the number of selected features.
    """
    features = list_features(x_train, y_train)
    x_train = x_train[:, features]
    x_test = x_test[:, features]
    return x_train, x_test


def add_bias_feature(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds a bias feature column to the train and test datasets

    Args:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (n,D), n is the number of test samples.

    Returns:
        x_train: numpy array of shape (N,D+1)
        x_test: numpy array of shape (N,D+1)
    """
    print("Pipeline Stage 9 - Adding Bias Feature...")
    bias = np.ones((x_train.shape[0], 1))
    bias_test = np.ones((x_test.shape[0], 1))

    x_train = np.hstack((x_train, bias))
    x_test = np.hstack((x_test, bias_test))

    return x_train, x_test


def standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Performs standardization over a dataset to ensure
    that all features are in the same scale

    Args:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (n,D), n is the number of test samples.

    Returns:
        x_train: standardized numpy array of shape (N,D)
        x_test: standardized numpy array of shape (N,D)
    """
    print("Pipeline Stage 8 - Standardizing Numerical Features...")
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test
