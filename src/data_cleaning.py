import numpy as np

from src import feature_type_detection


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


def drop_useless_features(train_dataset, test_dataset):
    """
    Drops unwanted or redundant features.

    Parameters:
    train_dataset (dict): Dictionary containing training data features.
    test_dataset (dict): Dictionary containing testing data features.

    Returns:
    - train_dataset (dict): Training dataset with specified features removed.
    - test_dataset (dict): Testing dataset with specified features removed.
    """
    print("Pipeline Stage 3 - Dropping Unwanted Features...")
    variables_to_drop = [
        "CTELENUM",
        "CTELNUM1",  # -> they are always yes
        "LADULT",
        "CADULT",  # they are always yes
        "COLGHOUS",
        "CCLGHOUS",  # always yes when pvtresidence is no
        "LANDLINE",
        "IDATE",
        "FMONTH",
        "SEQNO",
        "_PSU",
        "STATERES",
        "NUMADULT",
        "PVTRESD2",
        "CELLFON3",
        "CELLFON2",
        "_STATE",
    ]

    # Remove the corresponding features from our datasets
    for feature in variables_to_drop:
        train_dataset.pop(feature, None)
        test_dataset.pop(feature, None)

    return train_dataset, test_dataset


def replace_weird_values(train_dataset, test_dataset, abnormal_feature_values):
    """
    Replaces abnormal values in the training and testing datasets with their specified replacements.

    Parameters:
    train_dataset (dict): Dictionary of training data features, where each key is a feature name and the value is an array of feature values.
    test_dataset (dict): Dictionary of testing data features, where each key is a feature name and the value is an array of feature values.
    abnormal_feature_values (dict): A dictionary specifying abnormal values for each feature, structured as:
                                    {feature_name: {abnormal_value: replacement_value, ...}, ...}
                                    where each `abnormal_value` in a feature column is replaced by `replacement_value`.

    Returns:
    - train_dataset (dict): Training dataset with abnormal values replaced.
    - test_dataset (dict): Testing dataset with abnormal values replaced.
    """
    print("Pipeline Stage 4 - Replacing Abnormal Dataset Values...")

    def replace_values_with_dict(x, replace_dict):
        for value, replacement in replace_dict.items():
            x = np.where(x == value, replacement, x)
        return x

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


def fill_nans(train_dataset, test_dataset, feat_types):
    """Fills the NAN values in the dataset according to their feature type.

    In the case of a binary or categorical, the values are filled with the mode
    of the train array.
    If the feature is continuous we use the mean, and if it is ordinal or numerical
    discrete, the discrete mean is used.
    """
    print("Pipeline Stage 6 - Filling Nan Values...")

    def mode(feature_col):
        unique_values, counts = np.unique(
            feature_col, return_counts=True, equal_nan=False
        )
        return unique_values[np.argmax(counts)]

    def fill_nans_mode(feature_col, feature_col_test):
        feature_mode = mode(feature_col)
        filled_x_tr = np.where(np.isnan(feature_col), feature_mode, feature_col)
        filled_x_te = np.where(
            np.isnan(feature_col_test), feature_mode, feature_col_test
        )
        return filled_x_tr, filled_x_te

    def fill_nans_mean(feature_col, feature_col_test):
        feature_mean = np.nanmean(feature_col)
        filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
        filled_x_te = np.where(
            np.isnan(feature_col_test), feature_mean, feature_col_test
        )
        return filled_x_tr, filled_x_te

    def fill_nans_discrete_mean(feature_col, feature_col_test):
        feature_mean = np.round(np.nanmean(feature_col))
        filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
        filled_x_te = np.where(
            np.isnan(feature_col_test), feature_mean, feature_col_test
        )
        return filled_x_tr, filled_x_te

    for feature, feature_col in train_dataset.items():
        feature_col_test = test_dataset[feature]
        type = feat_types[feature]

        fill_func_factory = {
            feature_type_detection.FeatureType.BINARY: fill_nans_mode,  # binary
            feature_type_detection.FeatureType.CATEGORICAL: fill_nans_mode,  # categorical
            feature_type_detection.FeatureType.CONTINUOUS: fill_nans_mean,  # numerical continuous
            feature_type_detection.FeatureType.NUMERICAL_DISCRETE: fill_nans_discrete_mean,  # categorical ordinal or numerical discrete
            feature_type_detection.FeatureType.CATEGORICAL_ORDINAL: fill_nans_discrete_mean,
        }
        fill_func = fill_func_factory[type]
        train_dataset[feature], test_dataset[feature] = fill_func(
            feature_col, feature_col_test
        )

    return train_dataset, test_dataset
