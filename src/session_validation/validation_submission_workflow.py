

from notebooks.submission.submission_workflow import create_submission
from session_validation.hyperopt_analysis.best_run_manager import get_best_run
from session_validation.params.hyper_opt_params_parser import get_session_config_from_hyper_params
from session_validation.session_params import SessionParams
from session_validation.validation.session_validator import validate_session
import traceback
import json
import os

from utils.SYSCONFIG import DATA_SUBMISSION
from utils.util_functions import create_folder_if_not_exists

def run(
    trainable_bucket_name: str,
    submission_bucket_name: str,
    trainable_name: str,
    max_validation_epochs: int,
    submission_reduction_factor : float = 0.8,
    validation_runs = 3

):
    # be exception save
    ids_validated = []

    while True:
        try:
            # get best, not already validated run from hyperoptimization
            print('Searching for the best run.')
            score, iteration, id, hyper_params = get_best_run(
                bucket_name=trainable_bucket_name,
                trainable_name=trainable_name,
                exclude_ids=ids_validated
            ) 
            print(f'\n*** Found run for validation: Starting to process run {id} with score {score} *** \n')

            dir_path = os.path.join(DATA_SUBMISSION,id)
            create_folder_if_not_exists(dir_path)

            ids_validated.append(id)

            # create SessionParams config from hyper_opt for validation
            session_validation_config = get_session_config_from_hyper_params(hyper_params=hyper_params, epochs=max_validation_epochs)
            session_validation_config['session_param_id'] = 1234 + len(ids_validated)
            if 'Inception' in session_validation_config['model_config']['additional_architectures']:
                session_validation_config['model_config']['additional_architectures'].remove('Inception')

            config_json_path = os.path.join(dir_path,f'{id}_config.json')
                
            with open(config_json_path, 'w', encoding='utf8') as json_file:
                json.dump(session_validation_config, json_file, ensure_ascii=False, indent=2, default=str)
            print(f'Session validation config created.')
            
            best_epoch = 35

            if validation_runs != 0:

                # validate this config using the SessionValidation script, get the best model epoch
                validation_data = validate_session(session_validation_config, validation_runs=validation_runs)
                
                validation_json_path = os.path.join(dir_path, f'{id}_validation_data.json')
                with open(validation_json_path, 'w', encoding='utf8') as json_file:
                    json.dump(validation_data, json_file, ensure_ascii=False)
                
                best_epoch = round(validation_data['summary']['joined_epoch_avg'] * 0.85)



            # create a submission using this config and the validation info 
            print("Starting to create a submission")
            session_validation_config['training_config']['epochs'] = best_epoch
            print(f"Training the model for {best_epoch} epochs.")

            create_submission(
                config=session_validation_config,
                run_id=id,
                reduction_factor=submission_reduction_factor,
                submission_path=dir_path,
                visualize=True,
            )

            print("Created submission.")


            # upload results to aws
            os.system(f'aws s3 sync {dir_path} s3://{submission_bucket_name}/{id} --quiet')


        except Exception as e:
            print('Caught exception during validation. Skipping run')
            print(f'Exception: {str(e)}')
            error_str = traceback.format_exc()
            print(error_str)
            with open(f'{id}_error_log.txt', 'w', encoding='utf8') as json_file:
                json.dump(error_str, json_file, ensure_ascii=False, indent=2)
  
            print("\n\n\n")



if __name__ == '__main__':
    TRAINABLE_BUCKET = 'second-ray-bucket'
    SUBMISSION_BUCKET = 'ray-submissions'
    TRAINABLE_NAME = 'MergedTrainable_2022-07-08_22-00-00'
    MAX_VALIDATION_EPOCHS = 30
    SUBMISSION_REDUCTION_FACTOR = 0.8
    N_VALIDATION_RUNS = 0
    run(
        trainable_bucket_name= TRAINABLE_BUCKET, 
        trainable_name= TRAINABLE_NAME, 
        max_validation_epochs=MAX_VALIDATION_EPOCHS,
        submission_bucket_name=SUBMISSION_BUCKET,
        submission_reduction_factor=SUBMISSION_REDUCTION_FACTOR,
        validation_runs=N_VALIDATION_RUNS
    )