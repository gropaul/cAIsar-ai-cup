import pickle
from hyperoptimization.trainables.merged_trainable import MergedTrainable
from hyperoptimization.trainables.ueber_trainable import UeberTrainable
from hyperoptimization.trainables.preprocessing_trainable import PreprocessingTrainable
from hyperoptimization.callbacks.ray_submission_callback import RaySubmissionCallback

import pandas as pd
from ray.tune import run, ExperimentAnalysis, SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperoptimization.trainables.base_trainable import BaseTrainable
from hyperoptimization.trainables.ueber_trainable import UeberTrainable
from utils.duration import Duration

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class RayTuner:
    def __init__(self, trainable: BaseTrainable, scheduler, searcher) -> None:
        self.trainable = trainable
        self.scheduler = scheduler
        self.searcher = searcher
        
    def run(self) -> ExperimentAnalysis:
        analysis: ExperimentAnalysis = run(
            self.trainable,
            scheduler=self.scheduler,
            search_alg=self.searcher,
            config=self.trainable.hyperopt_space,

            num_samples=10000,
            time_budget_s=Duration(hours=10).in_seconds(),

            mode=self.trainable.mode,
            metric=self.trainable.metric,

            resources_per_trial={'cpu' : 6, 'gpu' : 1},
            max_concurrent_trials=None,

            checkpoint_score_attr = f'max-{self.trainable.metric}',
            keep_checkpoints_num = 10,
            name = "PreprocessingTrainable_2022-07-05_22-00-00",
            resume = False, # "AUTO"
            callbacks=[
                RaySubmissionCallback(trainable=self.trainable, reduction_factor=0.5)
            ],
        )
        return analysis
    

if __name__ == '__main__':
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=25,
        grace_period=4,
        reduction_factor=3, 
        brackets=1
    )
    
    searcher = HyperOptSearch()

    tuner = RayTuner(trainable=MergedTrainable, scheduler=scheduler, searcher=searcher)
    analysis: ExperimentAnalysis = tuner.run()

    results_df: pd.DataFrame = analysis.results_df
    with open('results_tune_att.csv', 'w') as f:
        results_df.to_csv(f, index=False)
    with open('analysis_tune_att.pkl', 'wb') as f:
        pickle.dump(analysis, f)
    

