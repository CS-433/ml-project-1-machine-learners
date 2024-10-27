import json

import numpy as np

import helpers as hp
import feature_types

def drop_features(x_tr, x_te, feature_names, feat_indexes, features_to_drop):
    remaining_features = [feature for feature in feature_names if feature not in features_to_drop]
    indexes_to_keep = [feat_indexes[feature] for feature in remaining_features]
    x_tr = x_tr[:, indexes_to_keep]
    x_te = x_te[:, indexes_to_keep]
    feature_names = remaining_features

    new_feat_indexes = {}
    for ind, feat in enumerate(feature_names):
        new_feat_indexes[feat] = ind

    return x_tr, x_te, feature_names, new_feat_indexes

def load_original_dataset(data_dir):
    print("Pipeline Stage 1 - Loading Datasets...")
    x_train = np.load(f"{data_dir}/train_dataset.npy")
    x_test = np.load(f"{data_dir}/test_dataset.npy")
    y_train = np.load(f"{data_dir}/train_targets.npy")
    return x_train, x_test, y_train

def merge_landline_cellphone_features(x_train, x_test, feat_indexes):
    print("Pipeline Stage 2 - Merging Landline and Cellphone Features...")
    cstate_idx = feat_indexes["CSTATE"]

    x_train[:, cstate_idx] = np.where(np.isnan(x_train[:, cstate_idx]), 1, x_train[:, cstate_idx])
    x_test[:, cstate_idx] = np.where(np.isnan(x_test[:, cstate_idx]), 1, x_test[:, cstate_idx])

    pvtresd1_idx = feat_indexes["PVTRESD1"]
    pvtresd2_idx = feat_indexes["PVTRESD2"]

    x_train[:, pvtresd1_idx] = np.where(np.isnan(x_train[:, pvtresd1_idx]), x_train[:, pvtresd2_idx], x_train[:, pvtresd1_idx])
    x_test[:, pvtresd1_idx] = np.where(np.isnan(x_test[:, pvtresd1_idx]), x_test[:, pvtresd2_idx], x_test[:, pvtresd1_idx])

    numadult_idx = feat_indexes["NUMADULT"]
    hhadult_idx = feat_indexes["HHADULT"]

    x_train[:, hhadult_idx] = np.where(
        (x_train[:, hhadult_idx] >= 6) & (x_train[:, hhadult_idx] <= 76), 6, x_train[:, hhadult_idx])
    x_test[:, hhadult_idx] = np.where((x_test[:, hhadult_idx] >= 6) & (x_test[:, hhadult_idx] <= 76), 6, x_test[:, hhadult_idx])

    x_train[:, hhadult_idx] = np.where(np.isnan(x_train[:, hhadult_idx]), x_train[:, numadult_idx], x_train[:, hhadult_idx])
    x_test[:, hhadult_idx] = np.where(np.isnan(x_test[:, hhadult_idx]), x_test[:, numadult_idx], x_test[:, hhadult_idx])

    return x_train, x_test

def drop_useless_features(train_dataset, test_dataset):
    print("Pipeline Stage 3 - Dropping Unwanted Features...")
    variables_to_drop = [
        "CTELENUM", "CTELNUM1", # -> they are always yes
        "LADULT", "CADULT",       # they are always yes
        "COLGHOUS", "CCLGHOUS",  # always yes when pvtresidence is no
        "LANDLINE", "IDATE", "FMONTH", "SEQNO", "_PSU", "STATERES", "NUMADULT", "PVTRESD2", "CELLFON3", "CELLFON2", "_STATE"
    ]

    # Remove the corresponding features from our datasets
    for feature in variables_to_drop:
        train_dataset.pop(feature, None)
        test_dataset.pop(feature, None)

    return train_dataset, test_dataset

def replace_weird_values(train_dataset, test_dataset, abnormal_feature_values):
    print("Pipeline Stage 4 - Replacing Abnormal Dataset Values...")
    def replace_values_with_dict(x, replace_dict):
        for value, replacement in replace_dict.items():
            x = np.where(x == value, replacement, x)
        return x
    
    abnormal_feature_values = {feature: {float(key): value for key, value in dic.items()} for feature, dic in abnormal_feature_values.items()}

    for feature, feature_col in train_dataset.items():
        feature_col_test = test_dataset[feature]

        replacement_dict = abnormal_feature_values[feature]
        if len(replacement_dict) == 0:
            # If the replacement dict is empty, just skip to the next value
            continue
        
        # Replace every abnormal value with Nans or 0s to ensure correct treatment
        # in the following steps
        train_dataset[feature] = replace_values_with_dict(feature_col, replacement_dict)
        test_dataset[feature] = replace_values_with_dict(feature_col_test, replacement_dict)
    
    return train_dataset, test_dataset

def fill_nans(train_dataset, test_dataset, feat_types):
    print("Pipeline Stage 6 - Filling Nan Values...")
    def mode(feature_col):
        unique_values, counts = np.unique(feature_col, return_counts=True, equal_nan=False)
        return unique_values[np.argmax(counts)]

    def fill_nans_mode(feature_col, feature_col_test):
        feature_mode = mode(feature_col)
        filled_x_tr = np.where(np.isnan(feature_col), feature_mode, feature_col)
        filled_x_te = np.where(np.isnan(feature_col_test), feature_mode, feature_col_test)
        return filled_x_tr, filled_x_te

    def fill_nans_mean(feature_col, feature_col_test):
        feature_mean = np.nanmean(feature_col)
        filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
        filled_x_te = np.where(np.isnan(feature_col_test), feature_mean, feature_col_test)
        return filled_x_tr, filled_x_te

    def fill_nans_discrete_mean(feature_col, feature_col_test):
        feature_mean = np.round(np.nanmean(feature_col))
        filled_x_tr = np.where(np.isnan(feature_col), feature_mean, feature_col)
        filled_x_te = np.where(np.isnan(feature_col_test), feature_mean, feature_col_test)
        return filled_x_tr, filled_x_te
    
    for feature, feature_col in train_dataset.items():
        feature_col_test = test_dataset[feature]
        type = feat_types[feature]

        fill_func_factory = {
            feature_types.FeatureType.BINARY: fill_nans_mode, # binary
            feature_types.FeatureType.CATEGORICAL: fill_nans_mode, # categorical
            feature_types.FeatureType.CONTINUOUS: fill_nans_mean, # numerical continuous
            feature_types.FeatureType.NUMERICAL_DISCRETE: fill_nans_discrete_mean, # categorical ordinal or numerical discrete
            feature_types.FeatureType.CATEGORICAL_ORDINAL: fill_nans_discrete_mean,
        }
        fill_func = fill_func_factory[type]
        train_dataset[feature], test_dataset[feature] = fill_func(feature_col, feature_col_test)
    
    return train_dataset, test_dataset

def one_hot_categoricals(x_train, x_test, feature_names, feat_indexes, feat_types):
    print("Pipeline Stage 7 - One Hot Encoding Categoricals...")
    categorical_features = [feature for feature in feature_names if feat_types[feature] == feature_types.FeatureType.CATEGORICAL]

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
        return np.concatenate(one_hots_tr, axis=1), np.concatenate(one_hots_test, axis=1)

    one_hot_tr, one_hot_te = one_hot_encode(x_train, x_test, features=categorical_features)
    # remove one-hotted variables from datasets
    x_train, x_test, feature_names, feat_indexes = drop_features(x_train, x_test, feature_names, feat_indexes, features_to_drop=categorical_features)

    x_train = np.hstack((x_train, one_hot_tr))
    x_test = np.hstack((x_test, one_hot_te))
    return x_train, x_test, feature_names, feat_indexes

def standardize(x_train, x_test):
    print("Pipeline Stage 8 - Standardizing Data...")
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test

def list_features(x_train,y_train):

    """
    Returns the list of features selected

    Args : 
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of samples.

    Returns:

    list_features: list, index of selected features 
    """
    
    index_features = np.arange(x_train.shape[1])
    nb_features =x_train.shape[1]

    # We remove the features that are redundant
    correlation_matrix = np.corrcoef(x_train, rowvar=False)
    list_suppr = []
    for i in range(nb_features):
        for j in range(i+1,nb_features):
            if  np.abs(correlation_matrix[i, j]) > 0.9:
                corr_i = np.corrcoef(x_train[:, i], y_train)
                corr_j = np.corrcoef(x_train[:, j], y_train)
                if np.abs(corr_i[0,1]) < np.abs(corr_j[0,1]):
                    list_suppr.append(i)
                else:
                    list_suppr.append(j)
    x = x_train.copy()
    x = np.delete(x, list_suppr, axis = 1)

    index_features = np.delete(index_features, list_suppr)
    # We keep only the features that have a correlation > 0.1 with the target
    list_features =[]
    for i in range(x.shape[1]):
        corr = np.corrcoef(x[:, i], y_train)
        if np.abs(corr[0,1]) > 0:
            list_features.append(i)
    list_features = index_features[list_features]
    return list_features    

def select_features(x_train, y_train, x_test):
    """
    Returns the selected features for the train and test set

    Args : 
    x_train: numpy array of shape (N,D), D is the number of features.
    y_train: numpy array of shape (N,), N is the number of training samples.
    x_test: numpy array of shape (n,D), n is the number of test samples.

    Returns:

    x_train: numpy array of shape (N,d), d is the number of selected features.
    x_test: numpy array of shape (N,d), d is the number of selected features.
    """
    features = list_features(x_train, y_train)
    x_train = x_train[:, features]
    x_test = x_test[:, features]
    return x_train, x_test


def add_bias_feature(x_train, x_test):
    print("Pipeline Stage 9 - Adding Bias Feature...")
    bias = np.ones((x_train.shape[0], 1))
    bias_test = np.ones((x_test.shape[0], 1))

    x_train = np.hstack((x_train, bias))
    x_test = np.hstack((x_test, bias_test))

    return x_train, x_test


def build_dict_representation(x, feature_names):
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

def preprocess_data():
    """This function acts as a pipeline that performs data cleaning, feature selection and standardization,
    among other transformations.

    The resulting dataset is ready for modelling.
    Returns:
    x_train: numpy array of shape (N,d), d is the number of selected features.
    x_test: numpy array of shape (N,d), d is the number of selected features.
    y_train: numpy array of shape (N, 1), containing the labels in the [0, 1] range.
    """
    data_dir = "data/"

    # Define the array of feature names in order to help manipulate our data 
    feature_names = ["_STATE","FMONTH","IDATE","IMONTH","IDAY","IYEAR","DISPCODE","SEQNO","_PSU","CTELENUM","PVTRESD1","COLGHOUS","STATERES","CELLFON3","LADULT","NUMADULT","NUMMEN","NUMWOMEN","CTELNUM1","CELLFON2","CADULT","PVTRESD2","CCLGHOUS","CSTATE","LANDLINE","HHADULT","GENHLTH","PHYSHLTH","MENTHLTH","POORHLTH","HLTHPLN1","PERSDOC2","MEDCOST","CHECKUP1","BPHIGH4","BPMEDS","BLOODCHO","CHOLCHK","TOLDHI2","CVDSTRK3","ASTHMA3","ASTHNOW","CHCSCNCR","CHCOCNCR","CHCCOPD1","HAVARTH3","ADDEPEV2","CHCKIDNY","DIABETE3","DIABAGE2","SEX","MARITAL","EDUCA","RENTHOM1","NUMHHOL2","NUMPHON2","CPDEMO1","VETERAN3","EMPLOY1","CHILDREN","INCOME2","INTERNET","WEIGHT2","HEIGHT3","PREGNANT","QLACTLM2","USEEQUIP","BLIND","DECIDE","DIFFWALK","DIFFDRES","DIFFALON","SMOKE100","SMOKDAY2","STOPSMK2","LASTSMK2","USENOW3","ALCDAY5","AVEDRNK2","DRNK3GE5","MAXDRNKS","FRUITJU1","FRUIT1","FVBEANS","FVGREEN","FVORANG","VEGETAB1","EXERANY2","EXRACT11","EXEROFT1","EXERHMM1","EXRACT21","EXEROFT2","EXERHMM2","STRENGTH","LMTJOIN3","ARTHDIS2","ARTHSOCL","JOINPAIN","SEATBELT","FLUSHOT6","FLSHTMY2","IMFVPLAC","PNEUVAC3","HIVTST6","HIVTSTD3","WHRTST10","PDIABTST","PREDIAB1","INSULIN","BLDSUGAR","FEETCHK2","DOCTDIAB","CHKHEMO3","FEETCHK","EYEEXAM","DIABEYE","DIABEDU","CAREGIV1","CRGVREL1","CRGVLNG1","CRGVHRS1","CRGVPRB1","CRGVPERS","CRGVHOUS","CRGVMST2","CRGVEXPT","VIDFCLT2","VIREDIF3","VIPRFVS2","VINOCRE2","VIEYEXM2","VIINSUR2","VICTRCT4","VIGLUMA2","VIMACDG2","CIMEMLOS","CDHOUSE","CDASSIST","CDHELP","CDSOCIAL","CDDISCUS","WTCHSALT","LONGWTCH","DRADVISE","ASTHMAGE","ASATTACK","ASERVIST","ASDRVIST","ASRCHKUP","ASACTLIM","ASYMPTOM","ASNOSLEP","ASTHMED3","ASINHALR","HAREHAB1","STREHAB1","CVDASPRN","ASPUNSAF","RLIVPAIN","RDUCHART","RDUCSTRK","ARTTODAY","ARTHWGT","ARTHEXER","ARTHEDU","TETANUS","HPVADVC2","HPVADSHT","SHINGLE2","HADMAM","HOWLONG","HADPAP2","LASTPAP2","HPVTEST","HPLSTTST","HADHYST2","PROFEXAM","LENGEXAM","BLDSTOOL","LSTBLDS3","HADSIGM3","HADSGCO1","LASTSIG3","PCPSAAD2","PCPSADI1","PCPSARE1","PSATEST1","PSATIME","PCPSARS1","PCPSADE1","PCDMDECN","SCNTMNY1","SCNTMEL1","SCNTPAID","SCNTWRK1","SCNTLPAD","SCNTLWK1","SXORIENT","TRNSGNDR","RCSGENDR","RCSRLTN2","CASTHDX2","CASTHNO2","EMTSUPRT","LSATISFY","ADPLEASR","ADDOWN","ADSLEEP","ADENERGY","ADEAT1","ADFAIL","ADTHINK","ADMOVE","MISTMNT","ADANXEV","QSTVER","QSTLANG","MSCODE","_STSTR","_STRWT","_RAWRAKE","_WT2RAKE","_CHISPNC","_CRACE1","_CPRACE","_CLLCPWT","_DUALUSE","_DUALCOR","_LLCPWT","_RFHLTH","_HCVU651","_RFHYPE5","_CHOLCHK","_RFCHOL","_LTASTH1","_CASTHM1","_ASTHMS1","_DRDXAR1","_PRACE1","_MRACE1","_HISPANC","_RACE","_RACEG21","_RACEGR3","_RACE_G1","_AGEG5YR","_AGE65YR","_AGE80","_AGE_G","HTIN4","HTM4","WTKG3","_BMI5","_BMI5CAT","_RFBMI5","_CHLDCNT","_EDUCAG","_INCOMG","_SMOKER3","_RFSMOK3","DRNKANY5","DROCDY3_","_RFBING5","_DRNKWEK","_RFDRHV5","FTJUDA1_","FRUTDA1_","BEANDAY_","GRENDAY_","ORNGDAY_","VEGEDA1_","_MISFRTN","_MISVEGN","_FRTRESP","_VEGRESP","_FRUTSUM","_VEGESUM","_FRTLT1","_VEGLT1","_FRT16","_VEG23","_FRUITEX","_VEGETEX","_TOTINDA","METVL11_","METVL21_","MAXVO2_","FC60_","ACTIN11_","ACTIN21_","PADUR1_","PADUR2_","PAFREQ1_","PAFREQ2_","_MINAC11","_MINAC21","STRFREQ_","PAMISS1_","PAMIN11_","PAMIN21_","PA1MIN_","PAVIG11_","PAVIG21_","PA1VIGM_","_PACAT1","_PAINDX1","_PA150R2","_PA300R2","_PA30021","_PASTRNG","_PAREC1","_PASTAE1","_LMTACT1","_LMTWRK1","_LMTSCL1","_RFSEAT2","_RFSEAT3","_FLSHOT6","_PNEUMO2","_AIDTST3"]

    # Load an auxiliary file to help replace weird values in the dataset such as "77" or "99"
    with open("data/abnormal_feature_values.json") as f:
        abnormal_value_replacements = json.load(f)

    # Load the original datasets in .npy format (much faster)
    x_train, x_test, y_train = load_original_dataset(data_dir)
        
    # Convert each dataset to a dict of arrays so we can manipulate features by name
    # This is similar to a dataframe in Pandas, but less efficient
    train_dataset = build_dict_representation(x_train, feature_names)
    test_dataset = build_dict_representation(x_test, feature_names)

    # x_train, x_test = merge_landline_cellphone_features(x_train, x_test, feat_indexes)
    train_dataset, test_dataset = drop_useless_features(train_dataset, test_dataset)
    train_dataset, test_dataset = replace_weird_values(train_dataset, test_dataset, abnormal_value_replacements)
    feat_types = feature_types.detect_feature_types(train_dataset)
    train_dataset, test_dataset = fill_nans(train_dataset, test_dataset, feat_types)
    # train_dataset, test_dataset, feature_names, feat_indexes = one_hot_categoricals(train_dataset, test_dataset, feature_names, feat_indexes, feat_types) # why commented
    
    # Convert the datasets back to a numpy array so we can apply operations over all columns in parallel
    x_train = convert_dict_to_array(train_dataset)
    x_test = convert_dict_to_array(test_dataset)
    
    # Standardize features so we are dealing with variables in the same scale
    x_train, x_test = standardize(x_train, x_test) # for everything ??

    # Analyze correlations between features in order to remove redundant data and help with explainability
    x_train, x_test = select_features(x_train, y_train, x_test)

    # Add bias feature to the dataset so our log regression model is complete
    x_train, x_test = add_bias_feature(x_train, x_test)

    # Convert the targets into 1 or 0 so we can apply the lab loss function
    y_train = (y_train + 1) / 2

    return x_train, x_test, y_train