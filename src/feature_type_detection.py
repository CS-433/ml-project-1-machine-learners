from enum import Enum

import numpy as np


class FeatureType(Enum):
    """
    Enum that represents the possible types of a feature.
    The type of a feature will determine how this is treated in the next steps of the pipeline.
    """
    BINARY = 1
    CATEGORICAL = 2
    CONTINUOUS = 3
    NUMERICAL_DISCRETE = 4
    CATEGORICAL_ORDINAL = 5

"""
Auxiliary data structure computed manually to distinguish between categorical,
ordinal and numerically discrete features.

The feature detection process follows this process:
    1. Detect binary and continuous features (Easy process)
    2. Retrieve the rest of features
    3. Assess manually with the dataset descriptions whether
       these features are categorical, ordinal or numerically
       discrete.
    4. Store them in the FEATURE_TYPES dictionary for later use
"""
FEATURE_TYPES = {
    "IMONTH": FeatureType.NUMERICAL_DISCRETE,
    "IDAY": FeatureType.NUMERICAL_DISCRETE,
    "NUMMEN": FeatureType.NUMERICAL_DISCRETE,
    "NUMWOMEN": FeatureType.NUMERICAL_DISCRETE,
    "HHADULT": FeatureType.CATEGORICAL,
    "GENHLTH": FeatureType.CATEGORICAL_ORDINAL,
    "PHYSHLTH": FeatureType.NUMERICAL_DISCRETE,
    "MENTHLTH": FeatureType.NUMERICAL_DISCRETE,
    "POORHLTH": FeatureType.NUMERICAL_DISCRETE,
    "PERSDOC2": FeatureType.CATEGORICAL,
    "CHECKUP1": FeatureType.CATEGORICAL_ORDINAL,
    "BPHIGH4": FeatureType.CATEGORICAL,
    "CHOLCHK": FeatureType.CATEGORICAL_ORDINAL,
    "DIABETE3": FeatureType.CATEGORICAL,
    "DIABAGE2": FeatureType.CATEGORICAL_ORDINAL,
    "MARITAL": FeatureType.CATEGORICAL,
    "EDUCA": FeatureType.CATEGORICAL_ORDINAL,
    "RENTHOM1": FeatureType.CATEGORICAL,
    "NUMPHON2": FeatureType.NUMERICAL_DISCRETE,
    "EMPLOY1": FeatureType.CATEGORICAL,
    "CHILDREN": FeatureType.NUMERICAL_DISCRETE,
    "INCOME2": FeatureType.CATEGORICAL_ORDINAL,
    "WEIGHT2": FeatureType.NUMERICAL_DISCRETE,
    "HEIGHT3": FeatureType.NUMERICAL_DISCRETE,
    "SMOKDAY2": FeatureType.CATEGORICAL,
    "LASTSMK2": FeatureType.CATEGORICAL_ORDINAL,
    "USENOW3": FeatureType.CATEGORICAL,
    "ALCDAY5": FeatureType.CATEGORICAL,
    "AVEDRNK2": FeatureType.NUMERICAL_DISCRETE,
    "DRNK3GE5": FeatureType.NUMERICAL_DISCRETE,
    "MAXDRNKS": FeatureType.NUMERICAL_DISCRETE,
    "FRUITJU1": FeatureType.CATEGORICAL_ORDINAL,
    "FRUIT1": FeatureType.CATEGORICAL_ORDINAL,
    "FVBEANS": FeatureType.CATEGORICAL_ORDINAL,
    "FVGREEN": FeatureType.CATEGORICAL_ORDINAL,
    "FVORANG": FeatureType.CATEGORICAL_ORDINAL,
    "VEGETAB1": FeatureType.CATEGORICAL_ORDINAL,
    "EXRACT11": FeatureType.CATEGORICAL,
    "EXEROFT1": FeatureType.CATEGORICAL_ORDINAL,
    "EXERHMM1": FeatureType.NUMERICAL_DISCRETE,
    "EXRACT21": FeatureType.CATEGORICAL,
    "EXEROFT2": FeatureType.CATEGORICAL_ORDINAL,
    "EXERHMM2": FeatureType.NUMERICAL_DISCRETE,
    "STRENGTH": FeatureType.CATEGORICAL_ORDINAL,
    "ARTHSOCL": FeatureType.CATEGORICAL_ORDINAL,
    "JOINPAIN": FeatureType.NUMERICAL_DISCRETE,
    "SEATBELT": FeatureType.CATEGORICAL_ORDINAL,
    "FLSHTMY2": FeatureType.CATEGORICAL,
    "IMFVPLAC": FeatureType.CATEGORICAL,
    "HIVTSTD3": FeatureType.CATEGORICAL,
    "WHRTST10": FeatureType.CATEGORICAL,
    "PREDIAB1": FeatureType.CATEGORICAL,
    "BLDSUGAR": FeatureType.CATEGORICAL,
    "FEETCHK2": FeatureType.CATEGORICAL,
    "DOCTDIAB": FeatureType.CATEGORICAL,
    "CHKHEMO3": FeatureType.CATEGORICAL,
    "FEETCHK": FeatureType.CATEGORICAL,
    "EYEEXAM": FeatureType.CATEGORICAL_ORDINAL,
    "CAREGIV1": FeatureType.CATEGORICAL,
    "CRGVREL1": FeatureType.CATEGORICAL,
    "CRGVLNG1": FeatureType.CATEGORICAL_ORDINAL,
    "CRGVHRS1": FeatureType.CATEGORICAL_ORDINAL,
    "CRGVPRB1": FeatureType.CATEGORICAL,
    "CRGVMST2": FeatureType.CATEGORICAL,
    "VIDFCLT2": FeatureType.CATEGORICAL_ORDINAL,
    "VIREDIF3": FeatureType.CATEGORICAL_ORDINAL,
    "VIPRFVS2": FeatureType.CATEGORICAL_ORDINAL,
    "VINOCRE2": FeatureType.CATEGORICAL,
    "VIEYEXM2": FeatureType.CATEGORICAL_ORDINAL,
    "VIINSUR2": FeatureType.CATEGORICAL,
    "VICTRCT4": FeatureType.CATEGORICAL,
    "CDHOUSE": FeatureType.CATEGORICAL_ORDINAL,
    "CDASSIST": FeatureType.CATEGORICAL_ORDINAL,
    "CDHELP": FeatureType.CATEGORICAL_ORDINAL,
    "CDSOCIAL": FeatureType.CATEGORICAL_ORDINAL,
    "LONGWTCH": FeatureType.CATEGORICAL_ORDINAL,
    "ASTHMAGE": FeatureType.CATEGORICAL_ORDINAL,
    "ASERVIST": FeatureType.NUMERICAL_DISCRETE,
    "ASDRVIST": FeatureType.NUMERICAL_DISCRETE,
    "ASRCHKUP": FeatureType.NUMERICAL_DISCRETE,
    "ASACTLIM": FeatureType.NUMERICAL_DISCRETE,
    "ASYMPTOM": FeatureType.CATEGORICAL_ORDINAL,
    "ASNOSLEP": FeatureType.CATEGORICAL_ORDINAL,
    "ASTHMED3": FeatureType.CATEGORICAL_ORDINAL,
    "ASINHALR": FeatureType.CATEGORICAL_ORDINAL,
    "ASPUNSAF": FeatureType.CATEGORICAL,
    "ARTTODAY": FeatureType.CATEGORICAL_ORDINAL,
    "TETANUS": FeatureType.CATEGORICAL_ORDINAL,
    "HPVADSHT": FeatureType.NUMERICAL_DISCRETE,
    "HOWLONG": FeatureType.CATEGORICAL_ORDINAL,
    "LASTPAP2": FeatureType.CATEGORICAL_ORDINAL,
    "HPLSTTST": FeatureType.CATEGORICAL_ORDINAL,
    "LENGEXAM": FeatureType.CATEGORICAL_ORDINAL,
    "LSTBLDS3": FeatureType.CATEGORICAL_ORDINAL,
    "LASTSIG3": FeatureType.CATEGORICAL_ORDINAL,
    "PSATIME": FeatureType.CATEGORICAL_ORDINAL,
    "PCPSARS1": FeatureType.CATEGORICAL,
    "PCPSADE1": FeatureType.CATEGORICAL,
    "PCDMDECN": FeatureType.CATEGORICAL,
    "SCNTMNY1": FeatureType.CATEGORICAL_ORDINAL,
    "SCNTMEL1": FeatureType.CATEGORICAL_ORDINAL,
    "SCNTPAID": FeatureType.CATEGORICAL,
    "SCNTWRK1": FeatureType.NUMERICAL_DISCRETE,
    "SCNTLPAD": FeatureType.CATEGORICAL,
    "SCNTLWK1": FeatureType.NUMERICAL_DISCRETE,
    "SXORIENT": FeatureType.CATEGORICAL,
    "TRNSGNDR": FeatureType.CATEGORICAL,
    "RCSRLTN2": FeatureType.CATEGORICAL,
    "EMTSUPRT": FeatureType.CATEGORICAL_ORDINAL,
    "LSATISFY": FeatureType.CATEGORICAL_ORDINAL,
    "ADPLEASR": FeatureType.CATEGORICAL_ORDINAL,
    "ADDOWN": FeatureType.CATEGORICAL_ORDINAL,
    "ADSLEEP": FeatureType.CATEGORICAL_ORDINAL,
    "ADENERGY": FeatureType.CATEGORICAL_ORDINAL,
    "ADEAT1": FeatureType.CATEGORICAL_ORDINAL,
    "ADFAIL": FeatureType.CATEGORICAL_ORDINAL,
    "ADTHINK": FeatureType.CATEGORICAL_ORDINAL,
    "ADMOVE": FeatureType.CATEGORICAL_ORDINAL,
    "QSTVER": FeatureType.CATEGORICAL,
    "QSTLANG": FeatureType.CATEGORICAL,
    "MSCODE": FeatureType.CATEGORICAL,
    "_STSTR": FeatureType.CATEGORICAL,
    "_CRACE1": FeatureType.CATEGORICAL,
    "_CPRACE": FeatureType.CATEGORICAL,
    "_DUALUSE": FeatureType.CATEGORICAL,
    "_CHOLCHK": FeatureType.CATEGORICAL,
    "_ASTHMS1": FeatureType.CATEGORICAL,
    "_PRACE1": FeatureType.CATEGORICAL,
    "_MRACE1": FeatureType.CATEGORICAL,
    "_RACE": FeatureType.CATEGORICAL,
    "_RACEGR3": FeatureType.CATEGORICAL,
    "_RACE_G1": FeatureType.CATEGORICAL,
    "_AGEG5YR": FeatureType.CATEGORICAL_ORDINAL,
    "_AGE80": FeatureType.CATEGORICAL_ORDINAL,
    "_AGE_G": FeatureType.CATEGORICAL_ORDINAL,
    "HTIN4": FeatureType.NUMERICAL_DISCRETE,
    "_BMI5CAT": FeatureType.CATEGORICAL_ORDINAL,
    "_CHLDCNT": FeatureType.CATEGORICAL_ORDINAL,
    "_EDUCAG": FeatureType.CATEGORICAL_ORDINAL,
    "_INCOMG": FeatureType.CATEGORICAL_ORDINAL,
    "_SMOKER3": FeatureType.CATEGORICAL,
    "DROCDY3_": FeatureType.CATEGORICAL_ORDINAL,
    "_DRNKWEK": FeatureType.NUMERICAL_DISCRETE,
    "_MISFRTN": FeatureType.CATEGORICAL,
    "_MISVEGN": FeatureType.CATEGORICAL,
    "_FRTRESP": FeatureType.CATEGORICAL,
    "_VEGRESP": FeatureType.CATEGORICAL,
    "_FRUITEX": FeatureType.CATEGORICAL,
    "_VEGETEX": FeatureType.CATEGORICAL,
    "ACTIN11_": FeatureType.CATEGORICAL,
    "ACTIN21_": FeatureType.CATEGORICAL,
    "PADUR1_": FeatureType.NUMERICAL_DISCRETE,
    "PADUR2_": FeatureType.NUMERICAL_DISCRETE,
    "_MINAC11": FeatureType.NUMERICAL_DISCRETE,
    "_MINAC21": FeatureType.NUMERICAL_DISCRETE,
    "PAMISS1_": FeatureType.CATEGORICAL,
    "PAMIN11_": FeatureType.NUMERICAL_DISCRETE,
    "PAMIN21_": FeatureType.NUMERICAL_DISCRETE,
    "PA1MIN_": FeatureType.NUMERICAL_DISCRETE,
    "PAVIG11_": FeatureType.NUMERICAL_DISCRETE,
    "PAVIG21_": FeatureType.NUMERICAL_DISCRETE,
    "PA1VIGM_": FeatureType.NUMERICAL_DISCRETE,
    "_PACAT1": FeatureType.CATEGORICAL_ORDINAL,
    "_PA150R2": FeatureType.CATEGORICAL_ORDINAL,
    "_PA300R2": FeatureType.CATEGORICAL_ORDINAL,
    "_PAREC1": FeatureType.CATEGORICAL_ORDINAL,
    "_LMTACT1": FeatureType.CATEGORICAL_ORDINAL,
    "_LMTWRK1": FeatureType.CATEGORICAL_ORDINAL,
    "_LMTSCL1": FeatureType.CATEGORICAL_ORDINAL,
}


def detect_binary_features(train_dataset: dict[str, np.ndarray]) -> list[str]:
    """
    This method detects which features are binary in the dataset.
    
    After having performed data cleaning, binary features just
    contain two values + Nan, so we remove Nans from the unique values
    and check if the unique_count is 2.
    
    Args:
        train_dataset: Dictionary containing a feature name as key and
        the feature's numpy array as value for every feature

    Returns:
        binary_features: A list containing the names of the binary features
        in the dataset
    """
    binary_features = []
    for feature, feature_col in train_dataset.items():
        unique_values = np.unique(feature_col)
        num_unique_values = len(unique_values)

        # We dont want to count Nan as a value
        if np.isnan(unique_values).any():
            num_unique_values -= 1

        if num_unique_values == 2:
            binary_features.append(feature)

    return binary_features


def detect_numerical_continuous_features(train_dataset: dict[str, np.ndarray]) -> list[str]:
    """
    This method detects which features are continuous in the dataset.
    
    In order to do this, we just check if any element has a fractional
    part (in other words, it is non-integer) in the array.
    
    Args:
        train_dataset: Dictionary containing a feature name as key and
        the feature's numpy array as value for every feature

    Returns:
        cont: A list containing the names of the continuous features
        in the dataset
    """
    cont_features = []
    for feature, feature_col in train_dataset.items():
        unique_values = np.unique(feature_col)
        num_unique_values = len(unique_values)

        # We dont want to count Nan as a value
        if np.isnan(unique_values).any():
            num_unique_values -= 1
            unique_values = unique_values[~np.isnan(unique_values)]

        # Check if there are any non-integer values
        fractional_part, _ = np.modf(unique_values)
        non_integers_exist = np.any(fractional_part != 0)
        if non_integers_exist:
            cont_features.append(feature)

    return cont_features


def detect_feature_types(train_dataset: dict[str, np.ndarray]) -> dict[str, FeatureType]:
    """
    This method performs the complete feature type detection process.
    
    If features are not binary nor continuous, the FEATURE_TYPES data
    structure is used to assess the feature type.
    
    Args:
        train_dataset: Dictionary containing a feature name as key and
        the feature's numpy array as value for every feature

    Returns:
        feat_types: A dictionary containing every feature and its
        respective type in the dataset
    """
    print("Pipeline Stage 5 - Detecting Feature Types...")

    binary_features = detect_binary_features(train_dataset)
    cont_features = detect_numerical_continuous_features(train_dataset)

    feat_types = FEATURE_TYPES
    for feature in train_dataset.keys():
        if feature in binary_features:
            feat_types[feature] = FeatureType.BINARY
        elif feature in cont_features:
            feat_types[feature] = FeatureType.CONTINUOUS

    return feat_types
