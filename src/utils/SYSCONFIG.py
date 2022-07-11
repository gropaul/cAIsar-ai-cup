import multiprocessing
from os.path import exists,join

### PLEASE DEFINE YOUR PLATFORM TYPE AND PROJECT PATH HERE ###

PLATFORM = 'WINDOWS'
PROJECT_PATH = 'C:\\Informatik\\workspaces\\workspace_python\\cAIsar-ai-cup'

### THIS CONFIG IS STATIC ###

# --- PATHS CONFIG ---

if PLATFORM == 'WINDOWS':
    SRC_PATH = PROJECT_PATH + '\\src'
    DATA_PATH = PROJECT_PATH + '\\data'
    TEST_DATA_PATH = PROJECT_PATH + '\\data\\testdata'
    MODELS_PATH = PROJECT_PATH + '\\models'
    LOGS_PATH = PROJECT_PATH + '\\logs'
    TABLES_PATH = DATA_PATH + '\\tables'
    
if PLATFORM == 'UNIX':
    SRC_PATH = PROJECT_PATH + '/src'
    TEST_DATA_PATH = PROJECT_PATH + '/data/testdata'
    MODELS_PATH = PROJECT_PATH + '/models'
    LOGS_PATH = PROJECT_PATH + '/logs'
    TABLES_PATH = DATA_PATH + '/tables'
    print(TABLES_PATH)

AWS_RAY_RESULTS  = join(PROJECT_PATH,'ray_results')
SPLIT_INDICES = join(DATA_PATH, 'split_indices')
SUBMISSIONS_PATH = join(PROJECT_PATH, 'submissions')


# --- DATA SET CONFIG --- 

# - DATASET NAMES - 
DATA_GAIT = 'gait'              # the whole gait data
DATA_SUBMISSION = 'submissions'  # the data merged for the submission prediction
DATA_SETS = [                   # A list of all available datasets
    DATA_SUBMISSION,             
    DATA_GAIT,
    
]

# - SUBSET NAMES - 
DATA_SUBSET_GAIT_TRAINING = 'gait_training'
DATA_SUBSET_GAIT_VALIDATION = 'gait_evaluation'

CROSS_FOLD_VALIDATION_1 = 'cross_fold_val_0'
CROSS_FOLD_VALIDATION_2 = 'cross_fold_val_1'
CROSS_FOLD_VALIDATION_3 = 'cross_fold_val_2'
CROSS_FOLD_VALIDATION_4 = 'cross_fold_val_3'
CROSS_FOLD_VALIDATION_5 = 'cross_fold_val_4'

CROSS_FOLD_TRAINING_1 = 'cross_fold_train_0'
CROSS_FOLD_TRAINING_2 = 'cross_fold_train_1'
CROSS_FOLD_TRAINING_3 = 'cross_fold_train_2'
CROSS_FOLD_TRAINING_4 = 'cross_fold_train_3'
CROSS_FOLD_TRAINING_5 = 'cross_fold_train_4'

# - DATA SUBSETS CONFIG -
DATA_SUBSET_CONFIG = {

    # Gait based subsets 
    DATA_GAIT : {
         'data': DATA_GAIT, 
         'start': 0.0, 'end':1.0
    },
    DATA_SUBSET_GAIT_TRAINING : {
        'data': DATA_GAIT,  
        'start': 0.0, 'end': 0.75
    },
    DATA_SUBSET_GAIT_VALIDATION : {
        'data': DATA_GAIT,  
        'start': 0.75, 'end': 1.0
    },

    # cross validation data

    **{f'cross_fold_val_{i}' : {
        'data': DATA_GAIT,  
        'start': 0.2 * i, 'end':  0.2 * (i + 1)
    } for i in range(0,5)},

    **{f'cross_fold_train_{i}' : {
        'data': DATA_GAIT,  
        'start': 0.2 * i, 'end':  0.2 * (i + 1),
        'inverse': True
    } for i in range(0,5)},
    
    # submission based data
    DATA_SUBMISSION : { 
        'data': DATA_SUBMISSION,  
        'start': 0.0, 'end':1.0
    },
}

# --- Cross validation indices ---

CROSS_FOLD_VALIDATION_SUBSETS = [f'cross_fold_val_{i}' for i in range(0,5)]
CROSS_FOLD_TRAINING_SUBSETS = [f'cross_fold_train_{i}' for i in range(0,5)]

DATA_SUBSETS = [   # A list of all available subsets
    DATA_GAIT,
    DATA_SUBSET_GAIT_TRAINING,
    DATA_SUBSET_GAIT_VALIDATION,
    DATA_SUBMISSION,
    *CROSS_FOLD_VALIDATION_SUBSETS,
    *CROSS_FOLD_TRAINING_SUBSETS,
]

# --- PROCESSING SETTINGS ---

PROCESSOR_CORES = multiprocessing.cpu_count()
ID_MARKER: str = '#ID#'
