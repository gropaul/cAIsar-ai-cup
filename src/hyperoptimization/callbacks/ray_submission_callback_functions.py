
import os
from typing import List 

from cup_scripts.utils import write_results, check_result_format
from utils.SYSCONFIG import SUBMISSIONS_PATH

from data_processing.post_processing.post_processing_functions import join_dataset_steps
from utils.util_functions import transform_predictions_to_ts, printc, create_folder_if_not_exists, reduce_list

import boto3
s3 = boto3.client("s3")

SUBMISSION_BUCKET = 'ray-submissions'


def _send_message(message: str, **kwargs) -> None:
    printc(source='[RaySubmission]', message=message, **kwargs)


def create_and_save_submission(session, reduction_factor: float, id: str, dir: str, submission_bucket: dir):

    predictions_as_ts , _   = session.get_predictions_as_ts()
    predictions = transform_predictions_to_ts(predictions_as_ts)
    check_result_format(predictions)

    joined_predictions = join_dataset_steps(predictions)

    reduce_joined_predictions = _reduce_submissions(joined_predictions, reduction_factor=reduction_factor)
    check_result_format(reduce_joined_predictions)

    _save_submission(
        reduced_joined_predictions=reduce_joined_predictions,
        joined_predictions=joined_predictions,
        id=id,
        reduction_factor=reduction_factor,
        dir=dir
    )

    _send_message(f"Submission successfully saved for run {id}")



def _reduce_submissions(predictions, reduction_factor: float):
    # reduced
    reduced_predictions = []
    for prediction in predictions:
        reduced_prediction = reduce_list(prediction, reduction_factor)
        reduced_predictions.append(reduced_prediction)

    return reduced_predictions

    

def _save_submission(reduced_joined_predictions, joined_predictions, id: str, reduction_factor: float, dir: str):

    create_folder_if_not_exists(SUBMISSIONS_PATH)

    joined_filename = os.path.join(SUBMISSIONS_PATH,f"submission_joined_id={id}_reduced-{reduction_factor}.zip")
    joined_key = os.path.join(dir, os.path.basename(joined_filename))

    write_results(
        reduced_joined_predictions,
        filename=joined_filename
    )

    full_filename = os.path.join(SUBMISSIONS_PATH,f"submission_joined_id={id}_FULL.zip")
    full_key = os.path.join(dir, os.path.basename(full_filename))

    write_results(
        joined_predictions,
        filename=full_filename
    )

    s3.upload_file(
        Filename=joined_filename,
        Bucket=SUBMISSION_BUCKET,
        Key=joined_key
    )

    s3.upload_file(
        Filename=full_filename,
        Bucket=SUBMISSION_BUCKET,
        Key=full_key
    )
