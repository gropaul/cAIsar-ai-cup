from typing import List
import numpy as np
from copy import copy

def get_metrics(steps: List[List[int]]):
    """_summary_

    Args:
        steps (List[List[int]]): The list of predicted steps per time series [[1,4],[8,21],[213,231]]
        contains list with start and end index
    """

    number_of_detected_steps = len(steps)
    step_lengths = [None] * number_of_detected_steps

    step_offsets = [None] * (number_of_detected_steps - 1)

    for i in range(number_of_detected_steps):
        step_start, step_end = steps[i]
        length = step_end - step_start
        step_lengths[i] = length

        # get the step before to calculate the last offset
        if i != 0 :
            step_before_start, step_before_end = steps[i - 1]
            step_offset = step_start - step_before_end
            step_offsets[i -1] = step_offset

    return step_lengths, step_offsets


def join_with_before(steps,i):

    (start,end) = steps[i]
    (start_before,end_before) = steps[i - 1]

    joined_start, joined_end = start_before, end
    steps[i] = [joined_start, joined_end]
    del steps[i-1]
    return steps


def join_steps(steps: List[List[int]], 
        offset_quantile = 0.25, 
        length_quantile = 0.25, 
        small_offset_quantile = 0.2, 
        small_length_quantile = 0.2
):
    step_lengths, step_offsets = get_metrics(steps=steps)
    steps_length_avg = np.average(step_lengths)
    steps_offset_avg = np.average(step_offsets)

    step_length_quantile, small_step_length_quantile= steps_length_avg * length_quantile, steps_length_avg * small_length_quantile
    step_offset_quantile, small_step_offset_quantile = steps_offset_avg * offset_quantile, steps_offset_avg * small_offset_quantile
    
    for i in range(len(steps)):

        (start,end) = steps[i]
        step_length = end - start

        if i == 0: continue
        (start_before,end_before) = steps[i - 1]

        step_offset = start - end_before
        step_before_length = end_before - start_before

        # two smaller steps get merged to one
        if (step_offset < step_offset_quantile 
            and step_before_length < step_length_quantile
            and step_length < step_length_quantile
        ):
            return join_with_before(steps=steps,i=i), True

        # the first step is very small, the second is normal size
        if (step_offset < small_step_offset_quantile 
            and step_before_length < small_step_length_quantile
        ):
            return join_with_before(steps=steps,i=i), True

        # the first step is normal, the second is very small
        if (step_offset < small_step_offset_quantile 
            and step_length < small_step_length_quantile
        ):
            return join_with_before(steps=steps,i=i), True

            
    return steps, False

# Für 1. run small parameters: 
#length_quantile=0.4564, offset_quantile=0.6944
#From: 0.9512374047012664 to 0.9528609966524334, offset = 0.1624 % Punkte

# Für 2. run big parameters: 
#length_quantile=0.586667, offset_quantile=1.101333
#From: 0.9512374047012664 to 0.9526462546517871, offset = 0.1409 % Punkte

def join_dataset_steps(
    y_pred: List[List[List[int]]], 
    length_quantile: float = 0.5, 
    offset_quantile:float = 0.5, 
    small_offset_quantile:float=0.6944, 
    small_length_quantile:float=0.4564
):
    y_pred_new = [None] * len(y_pred)

    for i in range(len(y_pred)):
        #print(i)
        y = y_pred[i]
        ys = copy(y)

        while True:
            ys, change = join_steps(ys, 
                length_quantile=length_quantile, 
                offset_quantile=offset_quantile,
                small_offset_quantile=small_offset_quantile,
                small_length_quantile=small_length_quantile,


            ) 
            if not change: break

        y_pred_new[i] = ys

    return y_pred_new