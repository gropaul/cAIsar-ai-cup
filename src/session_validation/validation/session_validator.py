
from platform import architecture
from typing import Dict
from data_generator.batch_data_path_generator import get_csv_index_path
from session_validation.validation.validation_session_executer import validation_execution
from session_validation.session_params import SessionParams
from session_validation.validation.validation_bg_configs.splitted_validation_bg_config import create_splitted_batch_generator_configs
from session_validation.callbacks.validation_callback import get_validation_callback_data_path
import pandas as pd
import numpy as np

from utils.value_smoothing import smooth
import matplotlib.pyplot as plt

import os

TRAIN_TEST_SPLIT = 0.2
VALIDATION_RUN_COUNT = 5

SMOOTH_FUNCTION_VALUE = 0.6 # pls use 0.6

def validate_session(session_config, validation_runs: int = 5) -> Dict:

    validation_runs = min(validation_runs,5) # maximum of 5 indexes exist

    ### SESSION INIT ###

    session_id = session_config['session_param_id']
    print(f'Starting validation of session with the id {session_id}')
    batch_generator_configs = create_splitted_batch_generator_configs(session_config=session_config)

    run_index = 0

    best_scores_per_run = []
    validation_dfs_json = []

    for bg_train_config, bg_evaluation_config in batch_generator_configs[:validation_runs]:
                
        ### SESSION APPLY CONFIG ###
        session_config_cp = session_config.copy()
        session_config_cp['bg_train_config'] = bg_train_config
        session_config_cp['bg_evaluation_config'] = bg_evaluation_config

        ### SESSION EXECUTION ###

        """id = f"88{run_index:04d}88{session_id:04d}"
        validation_data_path = get_validation_callback_data_path(int(id))
        validation_df = pd.read_csv(validation_data_path)
        """
        
        session, validation_df = validation_execution(**session_config_cp, run_index=run_index)
        validation_dfs_json.append(validation_df.to_dict())

        ### SESSION ANALYSES ###
        best_scores = get_best_scores(validation_df, session_id, run_index)
        best_scores_per_run.append(best_scores)

        run_index += 1

    summary = create_and_print_summary(best_scores_per_run, session_id)
    validation_data = {}
    validation_data['summary'] = summary
    validation_data['best_scores'] = best_scores_per_run
    validation_data['scores'] = validation_dfs_json

    return validation_data



def create_and_print_summary(analyses, session_id):

    score = []
    epoch = []
    joined_score = []
    joined_epoch = []

    for analysis in analyses:
        max_f_score, max_f_score_epoch = analysis['max_f_score_epoch']
        max_joined_f_score, max_joined_f_score_epoch = analysis['max_joined_f_score_epoch']

        score.append(max_f_score)
        epoch.append(max_f_score_epoch)
        joined_score.append(max_joined_f_score)
        joined_epoch.append(max_joined_f_score_epoch)

    score_avg = round(np.average(score),ndigits=4)
    epoch_avg = round(np.average(epoch),ndigits=4)
    joined_score_avg = round(np.average(joined_score),ndigits=4)
    joined_epoch_avg = round(np.average(joined_epoch),ndigits=4)

    score_std = round(np.std(score),ndigits=4)
    epoch_std = round(np.std(epoch),ndigits=4)
    joined_score_std = round(np.std(joined_score),ndigits=4)
    joined_epoch_std = round(np.std(joined_epoch),ndigits=4)

    analysis_summary = {
        'score_avg': score_avg,
        'score_std': score_std,
        'epoch_avg': epoch_avg,
        'epoch_std': epoch_std,
        'joined_score_avg': joined_score_avg,
        'joined_score_std': joined_score_std,
        'joined_epoch_avg': joined_epoch_avg,
        'joined_epoch_std': joined_epoch_std,
    }

    result = f"""
    _______________________________________________________

    *** RESULT OF THE EVALUATION OF SESSION CONFIG {session_id:04d} ***

    Number of validation runs: {len(analyses)}
        
    - RAW RESULTS - 
    Best Score
        Average:            {score_avg}
        Standard Deviation: {score_std}
    Epoch of best score
        Average:            {epoch_avg}
        Standard Deviation: {epoch_std}

    - POST-PROCESSED RESULTS - 
    Best Score
        Average:            {joined_score_avg}
        Standard Deviation: {joined_score_std}
    Epoch of best score
        Average:            {joined_epoch_avg}
        Standard Deviation: {joined_epoch_std}
    _______________________________________________________
    """
    print(result)

    return analysis_summary



def create_score_plot(
    f_score, smooth_f_score,    
    joined_f_score, smooth_joined_f_score,
    session_id, run_index    
):
    plt.plot(f_score, 'green', alpha=0.7, label='F-Score')
    plt.plot(smooth_f_score, 'green', label='Smoothed F-Score')


    plt.plot(joined_f_score, 'blue', alpha=0.7, label='Joined F-Score')
    plt.plot(smooth_joined_f_score, 'blue', label='Smoothed Joined F-Score')

    plt.legend()
    #plt.savefig(f'{session_id}_{run_index}.png')



def get_best_scores(validation_df: pd.DataFrame, session_id, run_index: int):
    session_analysis = {} 
        
    f_score = validation_df['f_score'].values.tolist()
    smooth_f_score = smooth(f_score, SMOOTH_FUNCTION_VALUE)

    joined_f_score = validation_df['joined_f_score'].values.tolist()
    smooth_joined_f_score = smooth(joined_f_score, SMOOTH_FUNCTION_VALUE)

    create_score_plot(f_score, smooth_f_score, joined_f_score, smooth_joined_f_score, session_id, run_index)

    max_f_score = max(smooth_f_score)
    max_joined_f_score = max(smooth_joined_f_score)

    max_f_score_epoch= smooth_f_score.index(max_f_score)
    max_joined_f_score_epoch= smooth_joined_f_score.index(max_joined_f_score)

    session_analysis['max_f_score_epoch'] = [max_f_score, max_f_score_epoch]
    session_analysis['max_joined_f_score_epoch'] = [max_joined_f_score, max_joined_f_score_epoch]

    return session_analysis






def delete_batch_generator_configs(batch_generator_configs):
     for bg_train_config, bg_evaluation_config in batch_generator_configs:

        normalization_settings = bg_train_config['normalization_settings']
        normalization_function = normalization_settings['normalization_function']
        normalization_mode = normalization_settings['normalization_mode']

        train_index_path = get_csv_index_path(
            data_subset_name= bg_train_config['data_subset_name'],
            normalization_function=normalization_function,
            normalization_mode=normalization_mode
        )

        os.remove(train_index_path)

        val_index_path = get_csv_index_path(
            data_subset_name= bg_evaluation_config['data_subset_name'],
            normalization_function=normalization_function,
            normalization_mode=normalization_mode
        )

        os.remove(val_index_path)

import json
if __name__ == '__main__':

    ### SELECT A VERSION FOR VALIDATION HERE ###
    config = SessionParams.config_6
    NUMBER_OF_VALIDATION_RUNS = 5           # maximum is 5

    # you can override the configs here, f.e.:
    # config['training_config']['epochs'] = 10

    validation = validate_session(config,validation_runs=1)
    with open(f'validation_data.json', 'w', encoding='utf8') as json_file:
        json.dump(validation, json_file, ensure_ascii=False)

