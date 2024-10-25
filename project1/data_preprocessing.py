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

def drop_useless_features(x_train, x_test, feature_names, feat_indexes):
    print("Pipeline Stage 3 - Dropping Unwanted Features...")
    variables_to_drop = [ # TODO: move to conf file
        "CTELENUM", "CTELNUM1", # -> they are always yes
        "LADULT", "CADULT",       # they are always yes
        "COLGHOUS", "CCLGHOUS",  # always yes when pvtresidence is no
        "LANDLINE", "IDATE", "FMONTH", "SEQNO", "_PSU", "STATERES", "NUMADULT", "PVTRESD2", "CELLFON3", "CELLFON2", "_STATE"
    ]

    x_train, x_test, feature_names, feat_indexes = drop_features(x_train, x_test, feature_names, feat_indexes, features_to_drop=variables_to_drop)
    return x_train, x_test, feature_names, feat_indexes

def replace_weird_values(x_train, x_test, feature_names, feat_indexes):
    print("Pipeline Stage 4 - Replacing Abnormal Dataset Values...")
    def replace_values_with_dict(x, replace_dict):
        for value, replacement in replace_dict.items():
            x = np.where(x == value, replacement, x)
        return x

    with open("projects/project1/var_abnormal_values.json") as f:
        value_replacements = json.load(f)
    
    value_replacements = {feature: {float(key): value for key, value in dic.items()} for feature, dic in value_replacements.items()}

    for feature in feature_names:
        index = feat_indexes[feature]
        feat_col = x_train[:, index]
        feat_col_test = x_test[:, index]

        transdict = value_replacements[feature]
        if len(transdict) == 0:
            continue

        x_train[:, index] = replace_values_with_dict(feat_col, transdict)
        x_test[:, index] = replace_values_with_dict(feat_col_test, transdict)
    
    return x_train, x_test

def detect_feature_types(x_train, feature_names, feat_indexes):
    print("Pipeline Stage 5 - Detecting Feature Types...")
    def detect_binary_features(x_train):
        binary_features = []
        for feature, index in feat_indexes.items():
            feat_col = x_train[:, index]
            unique_values = np.unique(feat_col)
            num_unique_values = len(unique_values)

            # We dont want to count Nan as a value
            if np.isnan(unique_values).any():
                num_unique_values -= 1

            if num_unique_values == 2:
                binary_features.append(feature)
        
        return binary_features

    def detect_numerical_continuous_features(x_train):
        cont_features = []
        for feature, index in feat_indexes.items():
            feat_col = x_train[:, index]
            unique_values = np.unique(feat_col)
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

    binary_features = detect_binary_features(x_train)
    cont_features = detect_numerical_continuous_features(x_train)
    
    feat_types = feature_types.feature_types
    for feature in feature_names:
        if feature in binary_features:
            feat_types[feature] = feature_types.FeatureType.BINARY
        elif feature in cont_features:
            feat_types[feature] = feature_types.FeatureType.CONTINUOUS
    
    return feat_types

def fill_nans(x_train, x_test, feat_indexes, feat_types):
    print("Pipeline Stage 6 - Filling Nan Values...")
    def mode(x):
        unique_values, counts = np.unique(x, return_counts=True, equal_nan=False)
        return unique_values[np.argmax(counts)]

    def fill_nans_mode(x_train, x_test):
        feature_mode = mode(x_train)
        filled_x_tr = np.where(np.isnan(x_train), feature_mode, x_train)
        filled_x_te = np.where(np.isnan(x_test), feature_mode, x_test)
        return filled_x_tr, filled_x_te

    def fill_nans_mean(x_train, x_test):
        feature_mean = np.nanmean(x_train)
        filled_x_tr = np.where(np.isnan(x_train), feature_mean, x_train)
        filled_x_te = np.where(np.isnan(x_test), feature_mean, x_test)
        return filled_x_tr, filled_x_te

    def fill_nans_discrete_mean(x_train, x_test):
        feature_mean = np.round(np.nanmean(x_train))
        filled_x_tr = np.where(np.isnan(x_train), feature_mean, x_train)
        filled_x_te = np.where(np.isnan(x_test), feature_mean, x_test)
        return filled_x_tr, filled_x_te
    
    for feature, type in feat_types.items():
        index = feat_indexes[feature]
        feat_col = x_train[:, index]
        feat_col_test = x_test[:, index]

        fill_func_factory = {
            feature_types.FeatureType.BINARY: fill_nans_mode, # binary
            feature_types.FeatureType.CATEGORICAL: fill_nans_mode, # categorical
            feature_types.FeatureType.CONTINUOUS: fill_nans_mean, # numerical continuous
            feature_types.FeatureType.NUMERICAL_DISCRETE: fill_nans_discrete_mean, # categorical ordinal or numerical discrete
            feature_types.FeatureType.CATEGORICAL_ORDINAL: fill_nans_discrete_mean,
        }
        fill_func = fill_func_factory[type]
        x_train[:, index], x_test[:, index] = fill_func(feat_col, feat_col_test)
    
    return x_train, x_test

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
    non_cat_indexes = [index for feat, index in feat_indexes.items() if feat not in categorical_features]
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

def add_bias_feature(x_train, x_test):
    print("Pipeline Stage 9 - Adding Bias Feature...")
    bias = np.ones((x_train.shape[0], 1))
    bias_test = np.ones((x_test.shape[0], 1))

    x_train = np.hstack((x_train, bias))
    x_test = np.hstack((x_test, bias_test))

    return x_train, x_test


def preprocess_data():
    data_dir = "projects/project1/data/dataset/"
    feature_names = ["_STATE","FMONTH","IDATE","IMONTH","IDAY","IYEAR","DISPCODE","SEQNO","_PSU","CTELENUM","PVTRESD1","COLGHOUS","STATERES","CELLFON3","LADULT","NUMADULT","NUMMEN","NUMWOMEN","CTELNUM1","CELLFON2","CADULT","PVTRESD2","CCLGHOUS","CSTATE","LANDLINE","HHADULT","GENHLTH","PHYSHLTH","MENTHLTH","POORHLTH","HLTHPLN1","PERSDOC2","MEDCOST","CHECKUP1","BPHIGH4","BPMEDS","BLOODCHO","CHOLCHK","TOLDHI2","CVDSTRK3","ASTHMA3","ASTHNOW","CHCSCNCR","CHCOCNCR","CHCCOPD1","HAVARTH3","ADDEPEV2","CHCKIDNY","DIABETE3","DIABAGE2","SEX","MARITAL","EDUCA","RENTHOM1","NUMHHOL2","NUMPHON2","CPDEMO1","VETERAN3","EMPLOY1","CHILDREN","INCOME2","INTERNET","WEIGHT2","HEIGHT3","PREGNANT","QLACTLM2","USEEQUIP","BLIND","DECIDE","DIFFWALK","DIFFDRES","DIFFALON","SMOKE100","SMOKDAY2","STOPSMK2","LASTSMK2","USENOW3","ALCDAY5","AVEDRNK2","DRNK3GE5","MAXDRNKS","FRUITJU1","FRUIT1","FVBEANS","FVGREEN","FVORANG","VEGETAB1","EXERANY2","EXRACT11","EXEROFT1","EXERHMM1","EXRACT21","EXEROFT2","EXERHMM2","STRENGTH","LMTJOIN3","ARTHDIS2","ARTHSOCL","JOINPAIN","SEATBELT","FLUSHOT6","FLSHTMY2","IMFVPLAC","PNEUVAC3","HIVTST6","HIVTSTD3","WHRTST10","PDIABTST","PREDIAB1","INSULIN","BLDSUGAR","FEETCHK2","DOCTDIAB","CHKHEMO3","FEETCHK","EYEEXAM","DIABEYE","DIABEDU","CAREGIV1","CRGVREL1","CRGVLNG1","CRGVHRS1","CRGVPRB1","CRGVPERS","CRGVHOUS","CRGVMST2","CRGVEXPT","VIDFCLT2","VIREDIF3","VIPRFVS2","VINOCRE2","VIEYEXM2","VIINSUR2","VICTRCT4","VIGLUMA2","VIMACDG2","CIMEMLOS","CDHOUSE","CDASSIST","CDHELP","CDSOCIAL","CDDISCUS","WTCHSALT","LONGWTCH","DRADVISE","ASTHMAGE","ASATTACK","ASERVIST","ASDRVIST","ASRCHKUP","ASACTLIM","ASYMPTOM","ASNOSLEP","ASTHMED3","ASINHALR","HAREHAB1","STREHAB1","CVDASPRN","ASPUNSAF","RLIVPAIN","RDUCHART","RDUCSTRK","ARTTODAY","ARTHWGT","ARTHEXER","ARTHEDU","TETANUS","HPVADVC2","HPVADSHT","SHINGLE2","HADMAM","HOWLONG","HADPAP2","LASTPAP2","HPVTEST","HPLSTTST","HADHYST2","PROFEXAM","LENGEXAM","BLDSTOOL","LSTBLDS3","HADSIGM3","HADSGCO1","LASTSIG3","PCPSAAD2","PCPSADI1","PCPSARE1","PSATEST1","PSATIME","PCPSARS1","PCPSADE1","PCDMDECN","SCNTMNY1","SCNTMEL1","SCNTPAID","SCNTWRK1","SCNTLPAD","SCNTLWK1","SXORIENT","TRNSGNDR","RCSGENDR","RCSRLTN2","CASTHDX2","CASTHNO2","EMTSUPRT","LSATISFY","ADPLEASR","ADDOWN","ADSLEEP","ADENERGY","ADEAT1","ADFAIL","ADTHINK","ADMOVE","MISTMNT","ADANXEV","QSTVER","QSTLANG","MSCODE","_STSTR","_STRWT","_RAWRAKE","_WT2RAKE","_CHISPNC","_CRACE1","_CPRACE","_CLLCPWT","_DUALUSE","_DUALCOR","_LLCPWT","_RFHLTH","_HCVU651","_RFHYPE5","_CHOLCHK","_RFCHOL","_LTASTH1","_CASTHM1","_ASTHMS1","_DRDXAR1","_PRACE1","_MRACE1","_HISPANC","_RACE","_RACEG21","_RACEGR3","_RACE_G1","_AGEG5YR","_AGE65YR","_AGE80","_AGE_G","HTIN4","HTM4","WTKG3","_BMI5","_BMI5CAT","_RFBMI5","_CHLDCNT","_EDUCAG","_INCOMG","_SMOKER3","_RFSMOK3","DRNKANY5","DROCDY3_","_RFBING5","_DRNKWEK","_RFDRHV5","FTJUDA1_","FRUTDA1_","BEANDAY_","GRENDAY_","ORNGDAY_","VEGEDA1_","_MISFRTN","_MISVEGN","_FRTRESP","_VEGRESP","_FRUTSUM","_VEGESUM","_FRTLT1","_VEGLT1","_FRT16","_VEG23","_FRUITEX","_VEGETEX","_TOTINDA","METVL11_","METVL21_","MAXVO2_","FC60_","ACTIN11_","ACTIN21_","PADUR1_","PADUR2_","PAFREQ1_","PAFREQ2_","_MINAC11","_MINAC21","STRFREQ_","PAMISS1_","PAMIN11_","PAMIN21_","PA1MIN_","PAVIG11_","PAVIG21_","PA1VIGM_","_PACAT1","_PAINDX1","_PA150R2","_PA300R2","_PA30021","_PASTRNG","_PAREC1","_PASTAE1","_LMTACT1","_LMTWRK1","_LMTSCL1","_RFSEAT2","_RFSEAT3","_FLSHOT6","_PNEUMO2","_AIDTST3"]

    feat_indexes = {}
    for ind, feat in enumerate(feature_names):
        feat_indexes[feat] = ind
    print("Pipeline Stage 1 - Loading Datasets...")
    x_train, x_test, y_train,train_ids, test_ids = hp.load_csv_data(data_dir)
    x_train, x_test = merge_landline_cellphone_features(x_train, x_test, feat_indexes)
    x_train, x_test, feature_names, feat_indexes = drop_useless_features(x_train, x_test, feature_names, feat_indexes)
    x_train, x_test = replace_weird_values(x_train, x_test, feature_names, feat_indexes)
    feat_types = detect_feature_types(x_train, feature_names, feat_indexes)
    x_train, x_test = fill_nans(x_train, x_test, feat_indexes, feat_types)
    #x_train, x_test, feature_names, feat_indexes = one_hot_categoricals(x_train, x_test, feature_names, feat_indexes, feat_types) # why commented
    x_train, x_test = standardize(x_train, x_test) # for everything ??
    x_train, x_test = add_bias_feature(x_train, x_test)
    y_train = (y_train + 1) / 2

    return x_train, x_test, y_train, train_ids, test_ids