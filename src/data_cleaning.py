import numpy as np

from src import feature_type_detection


# TODO
def merge_landline_cellphone_features(x_train, x_test, feat_indexes):
    print("Pipeline Stage 2 - Merging Landline and Cellphone Features...")
    cstate_idx = feat_indexes["CSTATE"]

    x_train[:, cstate_idx] = np.where(
        np.isnan(x_train[:, cstate_idx]), 1, x_train[:, cstate_idx]
    )
    x_test[:, cstate_idx] = np.where(
        np.isnan(x_test[:, cstate_idx]), 1, x_test[:, cstate_idx]
    )

    pvtresd1_idx = feat_indexes["PVTRESD1"]
    pvtresd2_idx = feat_indexes["PVTRESD2"]

    x_train[:, pvtresd1_idx] = np.where(
        np.isnan(x_train[:, pvtresd1_idx]),
        x_train[:, pvtresd2_idx],
        x_train[:, pvtresd1_idx],
    )
    x_test[:, pvtresd1_idx] = np.where(
        np.isnan(x_test[:, pvtresd1_idx]),
        x_test[:, pvtresd2_idx],
        x_test[:, pvtresd1_idx],
    )

    numadult_idx = feat_indexes["NUMADULT"]
    hhadult_idx = feat_indexes["HHADULT"]

    x_train[:, hhadult_idx] = np.where(
        (x_train[:, hhadult_idx] >= 6) & (x_train[:, hhadult_idx] <= 76),
        6,
        x_train[:, hhadult_idx],
    )
    x_test[:, hhadult_idx] = np.where(
        (x_test[:, hhadult_idx] >= 6) & (x_test[:, hhadult_idx] <= 76),
        6,
        x_test[:, hhadult_idx],
    )

    x_train[:, hhadult_idx] = np.where(
        np.isnan(x_train[:, hhadult_idx]),
        x_train[:, numadult_idx],
        x_train[:, hhadult_idx],
    )
    x_test[:, hhadult_idx] = np.where(
        np.isnan(x_test[:, hhadult_idx]),
        x_test[:, numadult_idx],
        x_test[:, hhadult_idx],
    )

    return x_train, x_test


def drop_useless_features(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    features_to_drop: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Drops unwanted or redundant features.

    Parameters:
    train_dataset: Dictionary containing training data features.
    test_dataset: Dictionary containing testing data features.
    features_to_drop: List of features to drop from the dataset

    Returns:
    - train_dataset: Training dataset with specified features removed.
    - test_dataset: Testing dataset with specified features removed.
    """
    print("Pipeline Stage 3 - Dropping Unwanted Features...")

    # Remove the corresponding features from our datasets
    for feature in features_to_drop:
        train_dataset.pop(feature, None)
        test_dataset.pop(feature, None)

    return train_dataset, test_dataset


def replace_values_with_dict(x: np.ndarray, replace_dict: dict[float, float]) -> np.ndarray:
    """Replaces specified values in an array based on a dictionary mapping.

    Iterates through a dictionary of values and their replacements, and replaces each
    occurrence of a specified value in the array with its corresponding replacement.
    
    Args:
        x: Input array in which values will be replaced.
        replace_dict: Dictionary where each key is a value to be replaced 
        and each value is the replacement.
                                           
    Returns:
        np.ndarray: Array with values replaced according to the dictionary.
    """
    for value, replacement in replace_dict.items():
        x = np.where(x == value, replacement, x)
    return x

def replace_weird_values(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    abnormal_feature_values: dict[str, dict[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Replaces abnormal values in the training and testing datasets with their specified replacements.

    Parameters:
        train_dataset: Dictionary of training data features, where each key is a feature name and the value is an array of feature values.
        test_dataset: Dictionary of testing data features, where each key is a feature name and the value is an array of feature values.
        abnormal_feature_values: A dictionary specifying abnormal values for each feature, structured as:
        {feature_name: {abnormal_value: replacement_value}}
        where each `abnormal_value` in a feature column is replaced by `replacement_value`.
        This dict is defined in config.ABNORMAL_FEATURE_VALUES
        
    Returns:
        train_dataset: Training dataset with abnormal values replaced.
        test_dataset: Testing dataset with abnormal values replaced.
    """
    print("Pipeline Stage 4 - Replacing Abnormal Dataset Values...")

    abnormal_feature_values = {
        feature: {float(key): value for key, value in dic.items()}
        for feature, dic in abnormal_feature_values.items()
    }

    for feature, feature_col in train_dataset.items():
        feature_col_test = test_dataset[feature]

        replacement_dict = abnormal_feature_values.get(feature, None)
        if replacement_dict is None:
            # If the feature has no replacements, just skip to the next value
            continue

        # Replace every abnormal value with Nans or 0s to ensure correct treatment
        # in the following steps
        train_dataset[feature] = replace_values_with_dict(feature_col, replacement_dict)
        test_dataset[feature] = replace_values_with_dict(
            feature_col_test, replacement_dict
        )

    return train_dataset, test_dataset

def mode(feature_col: np.ndarray) -> float:
    """Calculates the mode of a given feature column.

    Identifies the most frequently occurring value in a column of features.
    Useful for filling missing values in categorical or binary features.
    
    Args:
        feature_col (array-like): Array of feature values, can contain NaN values.

    Returns:
        Most frequently occurring value (mode) in the input array.
    """
    unique_values, counts = np.unique(
        feature_col, return_counts=True, equal_nan=False
    )
    return unique_values[np.argmax(counts)]

def fill_nans_mode(feature_col: np.ndarray, feature_col_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fills NaN values in the training and testing arrays using the mode of the training data.

    Useful for categorical and binary features where missing values are replaced
    with the most frequently occurring value in the training data.

    Args:
        feature_col: Array of feature values in the training set, can contain NaN values.
        feature_col_test: Array of feature values in the test set, can contain NaN values.

    Returns:
        filled_x_tr: Training array with NaNs filled by mode.
        filled_x_te: Testing array with NaNs filled by mode of training data.
    """
    feature_mode = mode(feature_col)
    filled_x_tr = np.where(np.isnan(feature_col), feature_mode, feature_col)
    filled_x_te = np.where(
        np.isnan(feature_col_test), feature_mode, feature_col_test
    )
    return filled_x_tr, filled_x_te

def fill_nans_mean(feature_col: np.ndarray, feature_col_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fills NaN values in the training and testing arrays using the mean of the training data.

    Useful for continuous numerical features, where missing values are replaced by
    the mean of the training data.

    Args:
        feature_col: Array of feature values in the training set, can contain NaN values.
        feature_col_test: Array of feature values in the test set, can contain NaN values.

    Returns:
        filled_x_tr: Training array with NaNs filled by mean.
        filled_x_te: Testing array with NaNs filled by mean of training data.
    """
    feature_mean = np.nanmean(feature_col)
    filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
    filled_x_te = np.where(
        np.isnan(feature_col_test), feature_mean, feature_col_test
    )
    return filled_x_tr, filled_x_te

def fill_nans_discrete_mean(feature_col: np.ndarray, feature_col_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fills NaN values in the training and testing arrays using the rounded mean of the training data.

    Useful for ordinal or discrete numerical features, where missing values are replaced
    by the rounded mean of the training data.

    Args:
        feature_col: Array of feature values in the training set, can contain NaN values.
        feature_col_test: Array of feature values in the test set, can contain NaN values.

    Returns:
        filled_x_tr: Training array with NaNs filled by rounded mean.
        filled_x_te: Testing array with NaNs filled by rounded mean of training data.
    """
    feature_mean = np.round(np.nanmean(feature_col))
    filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
    filled_x_te = np.where(
        np.isnan(feature_col_test), feature_mean, feature_col_test
    )
    return filled_x_tr, filled_x_te

def fill_nans(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    feat_types: dict[str, feature_type_detection.FeatureType],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fills the NAN values in the dataset according to their feature type.

    In the case of a binary or categorical, the values are filled with the mode
    of the train array.
    If the feature is continuous we use the mean, and if it is ordinal or numerical
    discrete, the discrete mean is used.

    Args:
        train_dataset: Dictionary of training data features, where each key is a feature name and the value is an array of feature values.
        test_dataset: Dictionary of testing data features, where each key is a feature name and the value is an array of feature values.
        feat_types: Dictionary containing every feature as key and their respective types as value

    Returns:
        train_dataset: train dataset with NANs filled
        test_dataset: test dataset with NANs filled
    """
    print("Pipeline Stage 6 - Filling Nan Values...")

    for feature, feature_col in train_dataset.items():
        feature_col_test = test_dataset[feature]
        feature_type = feat_types[feature]

        fill_func_factory = {
            feature_type_detection.FeatureType.BINARY: fill_nans_mode,  # binary
            feature_type_detection.FeatureType.CATEGORICAL: fill_nans_mode,  # categorical
            feature_type_detection.FeatureType.CONTINUOUS: fill_nans_mean,  # numerical continuous
            feature_type_detection.FeatureType.NUMERICAL_DISCRETE: fill_nans_discrete_mean,  # categorical ordinal or numerical discrete
            feature_type_detection.FeatureType.CATEGORICAL_ORDINAL: fill_nans_discrete_mean,
        }
        fill_func = fill_func_factory[feature_type]
        train_dataset[feature], test_dataset[feature] = fill_func(
            feature_col, feature_col_test
        )

    return train_dataset, test_dataset
