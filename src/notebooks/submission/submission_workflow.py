
from typing import Dict
from notebooks.submission.submission_model_trainer import execute_training
from session_validation.session_params import SessionParams
from session.session import Session
from cup_scripts.utils import load_test, write_results, check_result_format
import matplotlib.pyplot as plt
from session_validation.validation.session_validator import get_best_scores


from utils.util_functions import transform_predictions_to_ts, reduce_list, create_folder_if_not_exists
from data_processing.post_processing.post_processing_functions import join_dataset_steps
import os

from utils.SYSCONFIG import SUBMISSIONS_PATH


def create_submission(config: Dict, run_id: str, submission_path, reduction_factor: float = 0.6, visualize: bool = False):

    # train the model
    session, df = execute_training(**config)
    best_scores = get_best_scores(df, run_id, 0)
    best_f_score = best_scores['max_joined_f_score_epoch'][0]
    score_str = str(round(best_f_score,ndigits=5)).replace('.','-')
    # predict the time series
    predictions_as_ts , _   = session.get_predictions_as_ts()
    print(f"Get get_predictions_as_ts result predictions: {len(predictions_as_ts)}")
    print(f"Number of files of validation generator: {len(session.validation_generator.file_list)}")
    # Should match

    # cast predictions and check format
    predictions = transform_predictions_to_ts(predictions_as_ts)
    check_result_format(predictions)

    # apply post processing
    joined_predictions = join_dataset_steps(predictions)

    # create reduced submissions
    reduced_joined_predictions = []
    for prediction in joined_predictions:
        reduced_prediction = reduce_list(prediction, reduction_factor)
        reduced_joined_predictions.append(reduced_prediction)
    check_result_format(reduced_joined_predictions)

    create_folder_if_not_exists(submission_path)

    write_results(
        reduced_joined_predictions,
        filename=os.path.join(submission_path,f"{score_str}_{run_id}_submission_joined_reduced-{reduction_factor}.zip")
    )
    write_results(
        joined_predictions,
        filename=os.path.join(submission_path,f"{score_str}_{run_id}_submission_joined_FULL.zip")
    )

    if visualize:
        visualize_path = os.path.join(submission_path,'plots')
        create_folder_if_not_exists(visualize_path)
        create_submission_plots(
            predictions, joined_predictions, reduced_joined_predictions,
            visualize_path
        )

  


def create_submission_plots(predictions, joined_predictions, reduced_joined_predictions, path):
    
    # load test data for visualization

    X_test, metadata_test = load_test()
    print('Downloaded data for submission for visualization')

    max_x = 10000
    plt.rcParams["figure.figsize"] = (20,5)
    line_args = {"linestyle": "--", "color": "k"}
    for index in range(96,len(X_test),14):

        plt.title(f'File number {str(index)}')

        x = X_test[index]
        subset = x.to_numpy()[:max_x,:8]

        # Cup data
        plots = plt.plot(subset, label="Data from cup")

        for x in range(len(plots)):
            plot = plots[x]
            alpha = 256 - round(128 / len(plots) * x +1)
            alpha = format(alpha, '02x')
            alpha = "80"
            #plot.set(color="#0000FF" + alpha)
        

        # Model predictions

        if True:
            for (start, end) in reduced_joined_predictions[index]:

                if start > max_x : break

                plt.axvline(start, **line_args)
                plt.axvline(end, **line_args)
                plt.axvspan(start, end, facecolor="g", alpha=0.3)

        if True:
            for (start, end) in joined_predictions[index]:
                plt.axvline(start, **line_args)
                plt.axvline(end, **line_args)
                plt.axvspan(start, end, facecolor="g", alpha=0.1)

        if True:
            for (start, end) in predictions[index]:
                plt.axvspan(start, end, facecolor="b", alpha=0.05)

        plt.legend(loc="upper left")
        #plt.ylim([-2,2])
        plt.savefig(os.path.join(path,f'run_{index}.png'))
        plt.clf()



if __name__ == '__main__':

    ### ENTER YOUR CONFIGURATION HERE ###

    CONFIG = SessionParams.config_6

    SUBMISSION_PATH = os.path.join(SUBMISSIONS_PATH,str(CONFIG['session_param_id']))
    REDUCTION_FACTOR = 1.0
    RUN_ID = "manual_run_1"
    CREATE_VISUALIZATIONS = True

    ### POSSIBLY override config
    #CONFIG['training_config']['epochs'] = 5

    create_submission(
        config=CONFIG,
        submission_path=SUBMISSION_PATH,
        run_id=RUN_ID,
        reduction_factor=REDUCTION_FACTOR,
        visualize=CREATE_VISUALIZATIONS
    )