import numpy as np

from src import feature_type_detection


# TODO
def one_hot_categoricals(x_train, x_test, feature_names, feat_indexes, feat_types):
    """Computes the one-hot categorical encoding of categorical features in the dataset"""
    print("Pipeline Stage 7 - One Hot Encoding Categoricals...")
    categorical_features = [
        feature
        for feature in feature_names
        if feat_types[feature] == feature_type_detection.FeatureType.CATEGORICAL
    ]

    def one_hot_encode(x_train, x_test, features):
        one_hots_tr = []
        one_hots_test = []
        for feature in features:
            index = feat_indexes[feature]
            feature_col = x_train[:, index]
            unique_values = np.unique(feature_col)
            one_hot = np.zeros((x_train.shape[0], len(unique_values)))

            feature_col_test = x_test[:, index]
            one_hot_test = np.zeros((x_test.shape[0], len(unique_values)))

            for i, category in enumerate(feature_col):
                index = np.where(unique_values == category)
                one_hot[i, index] = 1  # Set the corresponding index to 1

            for i, category in enumerate(feature_col_test):
                index = np.where(unique_values == category)
                one_hot_test[i, index] = 1  # Set the corresponding index to 1

            one_hots_tr.append(one_hot)
            one_hots_test.append(one_hot_test)
        return np.concatenate(one_hots_tr, axis=1), np.concatenate(
            one_hots_test, axis=1
        )

    one_hot_tr, one_hot_te = one_hot_encode(
        x_train, x_test, features=categorical_features
    )
    # remove one-hotted variables from datasets
    x_train, x_test, feature_names, feat_indexes = drop_features(
        x_train,
        x_test,
        feature_names,
        feat_indexes,
        features_to_drop=categorical_features,
    )

    # Remove the corresponding features from our datasets
    for feature in categorical_features:
        train_dataset.pop(feature, None)
        test_dataset.pop(feature, None)

    return train_dataset, test_dataset

    x_train = np.hstack((x_train, one_hot_tr))
    x_test = np.hstack((x_test, one_hot_te))
    return x_train, x_test, feature_names, feat_indexes


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
    print("Pipeline Stage 8 - Standardizing Data...")
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test
