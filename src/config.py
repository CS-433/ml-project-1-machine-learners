import numpy as np

DATA_FOLDER = "data/"

# Unwanted features
FEATURES_TO_DROP = [
    "CTELENUM",
    "CTELNUM1",
    "LADULT",
    "CADULT",
    "COLGHOUS",
    "CCLGHOUS",
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

# Define the array of feature names in order to help manipulate our data
FEATURE_NAMES = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "HHADULT",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "CHOLCHK",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "DIABAGE2",
    "SEX",
    "MARITAL",
    "EDUCA",
    "RENTHOM1",
    "NUMHHOL2",
    "NUMPHON2",
    "CPDEMO1",
    "VETERAN3",
    "EMPLOY1",
    "CHILDREN",
    "INCOME2",
    "INTERNET",
    "WEIGHT2",
    "HEIGHT3",
    "PREGNANT",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "SMOKDAY2",
    "STOPSMK2",
    "LASTSMK2",
    "USENOW3",
    "ALCDAY5",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXERANY2",
    "EXRACT11",
    "EXEROFT1",
    "EXERHMM1",
    "EXRACT21",
    "EXEROFT2",
    "EXERHMM2",
    "STRENGTH",
    "LMTJOIN3",
    "ARTHDIS2",
    "ARTHSOCL",
    "JOINPAIN",
    "SEATBELT",
    "FLUSHOT6",
    "FLSHTMY2",
    "IMFVPLAC",
    "PNEUVAC3",
    "HIVTST6",
    "HIVTSTD3",
    "WHRTST10",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "BLDSUGAR",
    "FEETCHK2",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "EYEEXAM",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVLNG1",
    "CRGVHRS1",
    "CRGVPRB1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVMST2",
    "CRGVEXPT",
    "VIDFCLT2",
    "VIREDIF3",
    "VIPRFVS2",
    "VINOCRE2",
    "VIEYEXM2",
    "VIINSUR2",
    "VICTRCT4",
    "VIGLUMA2",
    "VIMACDG2",
    "CIMEMLOS",
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
    "CDDISCUS",
    "WTCHSALT",
    "LONGWTCH",
    "DRADVISE",
    "ASTHMAGE",
    "ASATTACK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTTODAY",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "HPVADSHT",
    "SHINGLE2",
    "HADMAM",
    "HOWLONG",
    "HADPAP2",
    "LASTPAP2",
    "HPVTEST",
    "HPLSTTST",
    "HADHYST2",
    "PROFEXAM",
    "LENGEXAM",
    "BLDSTOOL",
    "LSTBLDS3",
    "HADSIGM3",
    "HADSGCO1",
    "LASTSIG3",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "PSATIME",
    "PCPSARS1",
    "PCPSADE1",
    "PCDMDECN",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTPAID",
    "SCNTWRK1",
    "SCNTLPAD",
    "SCNTLWK1",
    "SXORIENT",
    "TRNSGNDR",
    "RCSGENDR",
    "RCSRLTN2",
    "CASTHDX2",
    "CASTHNO2",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "MISTMNT",
    "ADANXEV",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_LTASTH1",
    "_CASTHM1",
    "_ASTHMS1",
    "_DRDXAR1",
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACE",
    "_RACEG21",
    "_RACEGR3",
    "_RACE_G1",
    "_AGEG5YR",
    "_AGE65YR",
    "_AGE80",
    "_AGE_G",
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5",
    "_BMI5CAT",
    "_RFBMI5",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_SMOKER3",
    "_RFSMOK3",
    "DRNKANY5",
    "DROCDY3_",
    "_RFBING5",
    "_DRNKWEK",
    "_RFDRHV5",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUTSUM",
    "_VEGESUM",
    "_FRTLT1",
    "_VEGLT1",
    "_FRT16",
    "_VEG23",
    "_FRUITEX",
    "_VEGETEX",
    "_TOTINDA",
    "METVL11_",
    "METVL21_",
    "MAXVO2_",
    "FC60_",
    "ACTIN11_",
    "ACTIN21_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMISS1_",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "_PACAT1",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_PASTAE1",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
]

# Auxiliary data structure to help replace weird values in the dataset such as "77" or "99"
ABNORMAL_FEATURE_VALUES = {
    "LANDLINE": {7.0: np.nan, 9.0: np.nan},
    "HHADULT": {77.0: np.nan, 99.0: np.nan},
    "GENHLTH": {7.0: np.nan, 9.0: np.nan},
    "PHYSHLTH": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "MENTHLTH": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "POORHLTH": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "HLTHPLN1": {7.0: np.nan, 9.0: np.nan},
    "PERSDOC2": {7.0: np.nan, 9.0: np.nan},
    "MEDCOST": {7.0: np.nan, 9.0: np.nan},
    "CHECKUP1": {7.0: np.nan, 9.0: np.nan},
    "BPHIGH4": {7.0: np.nan, 9.0: np.nan},
    "BPMEDS": {7.0: np.nan, 9.0: np.nan},
    "BLOODCHO": {7.0: np.nan, 9.0: np.nan},
    "CHOLCHK": {7.0: np.nan, 9.0: np.nan},
    "TOLDHI2": {7.0: np.nan, 9.0: np.nan},
    "CVDINFR4": {7.0: np.nan, 9.0: np.nan},
    "CVDCRHD4": {7.0: np.nan, 9.0: np.nan},
    "CVDSTRK3": {7.0: np.nan, 9.0: np.nan},
    "ASTHMA3": {7.0: np.nan, 9.0: np.nan},
    "ASTHNOW": {7.0: np.nan, 9.0: np.nan},
    "CHCSCNCR": {7.0: np.nan, 9.0: np.nan},
    "CHCOCNCR": {7.0: np.nan, 9.0: np.nan},
    "CHCCOPD1": {7.0: np.nan, 9.0: np.nan},
    "HAVARTH3": {7.0: np.nan, 9.0: np.nan},
    "ADDEPEV2": {7.0: np.nan, 9.0: np.nan},
    "CHCKIDNY": {7.0: np.nan, 9.0: np.nan},
    "DIABETE3": {7.0: np.nan, 9.0: np.nan},
    "DIABAGE2": {98.0: np.nan, 99.0: np.nan},
    "MARITAL": {9.0: np.nan},
    "EDUCA": {9.0: np.nan},
    "RENTHOM1": {7.0: np.nan, 9.0: np.nan},
    "NUMHHOL2": {7.0: np.nan, 9.0: np.nan},
    "NUMPHON2": {7.0: np.nan, 9.0: np.nan},
    "CPDEMO1": {7.0: np.nan, 9.0: np.nan},
    "VETERAN3": {7.0: np.nan, 9.0: np.nan},
    "EMPLOY1": {9.0: np.nan},
    "CHILDREN": {88.0: 0, 99.0: np.nan},
    "INCOME2": {77.0: np.nan, 99.0: np.nan},
    "INTERNET": {7.0: np.nan, 9.0: np.nan},
    "WEIGHT2": {7777.0: np.nan, 9999.0: np.nan},
    "HEIGHT3": {7777.0: np.nan, 9999.0: np.nan},
    "PREGNANT": {7.0: np.nan, 9.0: np.nan},
    "QLACTLM2": {7.0: np.nan, 9.0: np.nan},
    "USEEQUIP": {7.0: np.nan, 9.0: np.nan},
    "BLIND": {7.0: np.nan, 9.0: np.nan},
    "DECIDE": {7.0: np.nan, 9.0: np.nan},
    "DIFFWALK": {7.0: np.nan, 9.0: np.nan},
    "DIFFDRES": {7.0: np.nan, 9.0: np.nan},
    "DIFFALON": {7.0: np.nan, 9.0: np.nan},
    "SMOKE100": {7.0: np.nan, 9.0: np.nan},
    "SMOKDAY2": {7.0: np.nan, 9.0: np.nan},
    "STOPSMK2": {7.0: np.nan, 9.0: np.nan},
    "LASTSMK2": {77.0: np.nan, 99.0: np.nan},
    "USENOW3": {7.0: np.nan, 9.0: np.nan},
    "ALCDAY5": {777.0: np.nan, 999.0: np.nan},
    "AVEDRNK2": {77.0: np.nan, 99.0: np.nan},
    "DRNK3GE5": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "MAXDRNKS": {77.0: np.nan, 99.0: np.nan},
    "FRUITJU1": {777.0: np.nan, 999.0: np.nan},
    "FRUIT1": {777.0: np.nan, 999.0: np.nan},
    "FVBEANS": {777.0: np.nan, 999.0: np.nan},
    "FVGREEN": {777.0: np.nan, 999.0: np.nan},
    "FVORANG": {777.0: np.nan, 999.0: np.nan},
    "VEGETAB1": {777.0: np.nan, 999.0: np.nan},
    "EXERANY2": {7.0: np.nan, 9.0: np.nan},
    "EXRACT11": {77.0: np.nan, 99.0: np.nan},
    "EXEROFT1": {777.0: np.nan, 999.0: np.nan},
    "EXERHMM1": {777.0: np.nan, 999.0: np.nan},
    "EXRACT21": {77.0: np.nan, 99.0: np.nan},
    "EXEROFT2": {777.0: np.nan, 999.0: np.nan},
    "EXERHMM2": {777.0: np.nan, 999.0: np.nan},
    "STRENGTH": {777.0: np.nan, 999.0: np.nan},
    "LMTJOIN3": {7.0: np.nan, 9.0: np.nan},
    "ARTHDIS2": {7.0: np.nan, 9.0: np.nan},
    "ARTHSOCL": {7.0: np.nan, 9.0: np.nan},
    "JOINPAIN": {77.0: np.nan, 99.0: np.nan},
    "SEATBELT": {7.0: np.nan, 9.0: np.nan},
    "FLUSHOT6": {7.0: np.nan, 9.0: np.nan},
    "FLSHTMY2": {777777.0: np.nan, 999999.0: np.nan},
    "IMFVPLAC": {77.0: np.nan, 99.0: np.nan},
    "PNEUVAC3": {7.0: np.nan, 9.0: np.nan},
    "HIVTST6": {7.0: np.nan, 9.0: np.nan},
    "HIVTSTD3": {777777.0: np.nan, 999999.0: np.nan},
    "WHRTST10": {77.0: np.nan, 99.0: np.nan},
    "PDIABTST": {7.0: np.nan, 9.0: np.nan},
    "PREDIAB1": {7.0: np.nan, 9.0: np.nan},
    "INSULIN": {9.0: np.nan},
    "BLDSUGAR": {777.0: np.nan, 999.0: np.nan},
    "FEETCHK2": {777.0: np.nan, 999.0: np.nan},
    "DOCTDIAB": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "CHKHEMO3": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "FEETCHK": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "EYEEXAM": {7.0: np.nan, 9.0: np.nan},
    "DIABEYE": {7.0: np.nan, 9.0: np.nan},
    "DIABEDU": {7.0: np.nan, 9.0: np.nan},
    "CAREGIV1": {7.0: np.nan, 9.0: np.nan},
    "CRGVREL1": {77.0: np.nan, 99.0: np.nan},
    "CRGVLNG1": {7.0: np.nan, 9.0: np.nan},
    "CRGVHRS1": {7.0: np.nan, 9.0: np.nan},
    "CRGVPRB1": {77.0: np.nan, 99.0: np.nan},
    "CRGVPERS": {7.0: np.nan, 9.0: np.nan},
    "CRGVHOUS": {7.0: np.nan, 9.0: np.nan},
    "CRGVMST2": {7.0: np.nan, 9.0: np.nan},
    "CRGVEXPT": {7.0: np.nan, 9.0: np.nan},
    "VIDFCLT2": {7.0: np.nan},
    "VIREDIF3": {7.0: np.nan},
    "VIPRFVS2": {7.0: np.nan, 9.0: np.nan},
    "VINOCRE2": {77.0: np.nan, 99.0: np.nan},
    "VIEYEXM2": {7.0: np.nan, 9.0: np.nan},
    "VIINSUR2": {7.0: np.nan, 9.0: np.nan},
    "VICTRCT4": {7.0: np.nan},
    "VIGLUMA2": {7.0: np.nan},
    "VIMACDG2": {7.0: np.nan},
    "CIMEMLOS": {7.0: np.nan, 9.0: np.nan},
    "CDHOUSE": {7.0: np.nan, 9.0: np.nan},
    "CDASSIST": {7.0: np.nan, 9.0: np.nan},
    "CDHELP": {7.0: np.nan, 9.0: np.nan},
    "CDSOCIAL": {7.0: np.nan, 9.0: np.nan},
    "CDDISCUS": {7.0: np.nan, 9.0: np.nan},
    "WTCHSALT": {7.0: np.nan, 9.0: np.nan},
    "LONGWTCH": {777.0: np.nan, 999.0: np.nan},
    "DRADVISE": {7.0: np.nan, 9.0: np.nan},
    "ASTHMAGE": {98.0: np.nan, 99.0: np.nan},
    "ASATTACK": {7.0: np.nan},
    "ASERVIST": {88.0: 0, 98.0: np.nan},
    "ASDRVIST": {88.0: 0, 98.0: np.nan},
    "ASRCHKUP": {88.0: 0, 98.0: np.nan, 99.0: np.nan},
    "ASACTLIM": {777.0: np.nan, 888.0: 0, 999.0: np.nan},
    "ASYMPTOM": {7.0: np.nan, 9.0: np.nan},
    "ASNOSLEP": {7.0: np.nan, 8.0: 0},
    "ASTHMED3": {7.0: np.nan, 9.0: np.nan},
    "ASINHALR": {7.0: np.nan, 9.0: np.nan},
    "HAREHAB1": {7.0: np.nan, 9.0: np.nan},
    "STREHAB1": {7.0: np.nan, 9.0: np.nan},
    "CVDASPRN": {7.0: np.nan, 9.0: np.nan},
    "ASPUNSAF": {7.0: np.nan, 9.0: np.nan},
    "RLIVPAIN": {7.0: np.nan, 9.0: np.nan},
    "RDUCHART": {7.0: np.nan, 9.0: np.nan},
    "RDUCSTRK": {7.0: np.nan, 9.0: np.nan},
    "ARTTODAY": {7.0: np.nan, 9.0: np.nan},
    "ARTHWGT": {7.0: np.nan, 9.0: np.nan},
    "ARTHEXER": {7.0: np.nan, 9.0: np.nan},
    "ARTHEDU": {7.0: np.nan, 9.0: np.nan},
    "TETANUS": {7.0: np.nan, 9.0: np.nan},
    "HPVADVC2": {3.0: np.nan, 7.0: np.nan, 9.0: np.nan},
    "HPVADSHT": {77.0: np.nan, 99.0: np.nan},
    "SHINGLE2": {7.0: np.nan, 9.0: np.nan},
    "HADMAM": {7.0: np.nan, 9.0: np.nan},
    "HOWLONG": {7.0: np.nan, 9.0: np.nan},
    "HADPAP2": {7.0: np.nan, 9.0: np.nan},
    "LASTPAP2": {7.0: np.nan, 9.0: np.nan},
    "HPVTEST": {7.0: np.nan, 9.0: np.nan},
    "HPLSTTST": {7.0: np.nan, 9.0: np.nan},
    "HADHYST2": {7.0: np.nan, 9.0: np.nan},
    "PROFEXAM": {7.0: np.nan, 9.0: np.nan},
    "LENGEXAM": {7.0: np.nan, 9.0: np.nan},
    "BLDSTOOL": {7.0: np.nan, 9.0: np.nan},
    "LSTBLDS3": {7.0: np.nan, 9.0: np.nan},
    "HADSIGM3": {7.0: np.nan, 9.0: np.nan},
    "HADSGCO1": {7.0: np.nan, 9.0: np.nan},
    "LASTSIG3": {7.0: np.nan, 9.0: np.nan},
    "PCPSAAD2": {7.0: np.nan, 9.0: np.nan},
    "PCPSADI1": {7.0: np.nan, 9.0: np.nan},
    "PCPSARE1": {7.0: np.nan, 9.0: np.nan},
    "PSATEST1": {7.0: np.nan, 9.0: np.nan},
    "PSATIME": {7.0: np.nan, 9.0: np.nan},
    "PCPSARS1": {7.0: np.nan, 9.0: np.nan},
    "PCPSADE1": {9.0: np.nan},
    "PCDMDECN": {7.0: np.nan, 9.0: np.nan},
    "SCNTMNY1": {7.0: np.nan, 9.0: np.nan},
    "SCNTMEL1": {7.0: np.nan, 9.0: np.nan},
    "SCNTPAID": {7.0: np.nan, 9.0: np.nan},
    "SCNTWRK1": {97.0: np.nan, 99.0: np.nan},
    "SCNTLPAD": {7.0: np.nan, 9.0: np.nan},
    "SCNTLWK1": {97.0: np.nan, 99.0: np.nan},
    "SXORIENT": {7.0: np.nan, 9.0: np.nan},
    "TRNSGNDR": {7.0: np.nan, 9.0: np.nan},
    "RCSGENDR": {9.0: np.nan},
    "RCSRLTN2": {7.0: np.nan, 9.0: np.nan},
    "CASTHDX2": {7.0: np.nan, 9.0: np.nan},
    "CASTHNO2": {7.0: np.nan, 9.0: np.nan},
    "EMTSUPRT": {7.0: np.nan, 9.0: np.nan},
    "LSATISFY": {7.0: np.nan, 9.0: np.nan},
    "ADPLEASR": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADDOWN": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADSLEEP": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADENERGY": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADEAT1": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADFAIL": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADTHINK": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "ADMOVE": {88.0: 0, 77.0: np.nan, 99.0: np.nan},
    "MISTMNT": {7.0: np.nan, 9.0: np.nan},
    "ADANXEV": {7.0: np.nan, 9.0: np.nan},
    "_CHISPNC": {9.0: np.nan},
    "_CRACE1": {77.0: np.nan, 99.0: np.nan},
    "_CPRACE": {77.0: np.nan, 99.0: np.nan},
    "_RFHLTH": {9.0: np.nan},
    "_HCVU651": {9.0: np.nan},
    "_RFHYPE5": {9.0: np.nan},
    "_CHOLCHK": {9.0: np.nan},
    "_RFCHOL": {9.0: np.nan},
    "_LTASTH1": {9.0: np.nan},
    "_CASTHM1": {9.0: np.nan},
    "_ASTHMS1": {9.0: np.nan},
    "_PRACE1": {8.0: np.nan, 77.0: np.nan, 99.0: np.nan},
    "_MRACE1": {77.0: np.nan, 99.0: np.nan},
    "_HISPANC": {9.0: np.nan},
    "_RACE": {9.0: np.nan},
    "_RACEG21": {9.0: np.nan},
    "_RACEGR3": {9.0: np.nan},
    "_AGEG5YR": {14.0: np.nan},
    "_AGE65YR": {3.0: np.nan},
    "WTKG3": {99999.0: np.nan},
    "_RFBMI5": {9.0: np.nan},
    "_CHLDCNT": {9.0: np.nan},
    "_EDUCAG": {9.0: np.nan},
    "_INCOMG": {9.0: np.nan},
    "_SMOKER3": {9.0: np.nan},
    "_RFSMOK3": {9.0: np.nan},
    "DRNKANY5": {7.0: np.nan, 9.0: np.nan},
    "DROCDY3_": {900.0: np.nan},
    "_RFBING5": {9.0: np.nan},
    "_DRNKWEK": {99900.0: np.nan},
    "_RFDRHV5": {9.0: np.nan},
    "_FRTLT1": {9.0: np.nan},
    "_VEGLT1": {9.0: np.nan},
    "_TOTINDA": {9.0: np.nan},
    "MAXVO2_": {99900.0: np.nan},
    "FC60_": {99900.0: np.nan},
    "PAFREQ1_": {99000.0: np.nan},
    "PAFREQ2_": {99000.0: np.nan},
    "STRFREQ_": {99000.0: np.nan},
    "PAMISS1_": {9.0: np.nan},
    "_PACAT1": {9.0: np.nan},
    "_PAINDX1": {9.0: np.nan},
    "_PA150R2": {9.0: np.nan},
    "_PA300R2": {9.0: np.nan},
    "_PA30021": {9.0: np.nan},
    "_PASTRNG": {9.0: np.nan},
    "_PAREC1": {9.0: np.nan},
    "_PASTAE1": {9.0: np.nan},
    "_LMTACT1": {9.0: np.nan},
    "_LMTWRK1": {9.0: np.nan},
    "_LMTSCL1": {9.0: np.nan},
    "_RFSEAT2": {9.0: np.nan},
    "_RFSEAT3": {9.0: np.nan},
    "_FLSHOT6": {9.0: np.nan},
    "_PNEUMO2": {9.0: np.nan},
    "_AIDTST3": {9.0: np.nan},
}