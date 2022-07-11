
from ray import tune
import os
from ray.tune import Callback
from ray.tune.trial_runner import TrialRunner, Trial
from typing import List, Dict
from hyperoptimization.callbacks.ray_submission_callback_functions import create_and_save_submission

from utils.util_functions import printc





class RaySubmissionCallback(Callback):
    


    def __init__(self, trainable, reduction_factor = 0.4, ) -> None:
        self.trainable = trainable
        self.reduction_factor = reduction_factor



    def on_trial_complete(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
    
        runner = trial.runner
        print("******* RUNNER: ")
        print(vars(runner))

        print("*********Tuneable?")
        print(vars(trial.get_trainable_cls().get_current_session()))
        trial_map = vars(trial)

        # 'trainable_name': 'UeberTrainable', 'trial_id': 'c232e71b'
        id = trial_map['trial_id']
        
        training_dir_name = os.path.basename(trial_map['local_dir'])

        if hasattr(self.trainable,'session'):
            self.send_message(f'Starting to create a submission for {id}.')
            create_and_save_submission(session=self.trainable.session, reduction_factor=self.reduction_factor, id=id, dir=training_dir_name)
        else:
            self.send_message(f'Creating no submission for {id} as the trainable got no valid session attribute.')
        return super().on_trial_complete(iteration, trials, trial, **info)


    def send_message(self, message: str, **kwargs) -> None:
        printc(source='[RaySubmission]', message=message, **kwargs)