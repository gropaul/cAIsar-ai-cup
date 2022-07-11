from typing import Tuple
import numpy as np
from cup_scripts.metric import fscore_step_detection
from utils.util_functions import convert_float_to_binary_mask, convert_mask_to_cup_format

def cup_f1_score(y_pred: np.array, y_true: np.array) -> Tuple[float, float, float]:
    """takes two arrays containg (poss. multi-channel) time series of predictions and the respective ground truth;
    converts them from the mask to the cup format and calculates the score and metrics

    Args:
        y_pred (np.array): array of predicted masks on time-series
        y_true (np.array): array of actual masks on time-series

    Returns:
        Tuple[float, float, float]: Tuple of mean f-score, precision, recall
    """
    
    def process_channel(mask: np.array) -> list:
        # process one mask of shape (length,)
        binary_mask = convert_float_to_binary_mask(mask)
        steps = convert_mask_to_cup_format(binary_mask)
        steps = np.array(steps).tolist()
        return steps

    predictions = []
    ground_truth = []

    for mask_pred, mask_truth in zip(y_pred, y_true):
        multi_channel: bool = (len(y_pred[0].shape) > 1)
        channels = y_pred[0].shape[1] if multi_channel else 1

        for channel in range(channels):
            channel_pred = mask_pred[:, channel] if multi_channel else mask_pred
            channel_truth = mask_truth[:, channel] if multi_channel else mask_truth
            processed_pred = process_channel(mask=channel_pred)
            processed_truth = process_channel(mask=channel_truth)
            predictions.append(processed_pred)
            ground_truth.append(processed_truth)

    fscore, precision, recall = fscore_step_detection(y_pred=predictions, y_true=ground_truth)
    return fscore, precision, recall
