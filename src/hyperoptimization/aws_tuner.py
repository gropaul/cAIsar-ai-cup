import pickle
import pandas as pd

from ray.tune import run, ExperimentAnalysis, SyncConfig, CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperoptimization.trainables.base_trainable import BaseTrainable
from hyperoptimization.trainables.merged_trainable import MergedTrainable

from hyperoptimization.callbacks.ray_submission_callback import RaySubmissionCallback
# from hyperoptimization.trainables.ueber_trainable import UeberTrainable

from utils.duration import Duration


# This tuner is configured to run as distributed notes

class AWSTuner:
    def __init__(self, trainable: BaseTrainable, scheduler, searcher) -> None:
        self.trainable = trainable
        self.scheduler = scheduler
        self.searcher = searcher
        
    def run(self) -> ExperimentAnalysis:

        # configure how checkpoints are sync'd to the scheduler/sampler
        # we recommend cloud storage checkpoints as it survives the cluster when
        # instances are terminated, and has better performance
        sync_config = SyncConfig(
            upload_dir="s3://ray-results-bucket/synced_folder/",  # requires AWS credentials
        )

        #print(local_dir)
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

            progress_reporter= CLIReporter(max_report_frequency=60), # log every 60s, default is every 5s

           
            checkpoint_score_attr = f'max-{self.trainable.metric}',
            keep_checkpoints_num = 10,
            name = "PreprocessingTrainable_2022-06-17_15-00-00",

            resume=True,
            callbacks=[
                RaySubmissionCallback(trainable=self.trainable, reduction_factor=0.5)
            ],
            resume="AUTO",
            sync_config = sync_config,
        )
        return analysis
    

if __name__ == '__main__':
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=30,
        grace_period=5,
        reduction_factor=3, 
        brackets=1
    )
    
    searcher = HyperOptSearch()

    tuner = AWSTuner(trainable=MergedTrainable, scheduler=scheduler, searcher=searcher)
    analysis: ExperimentAnalysis = tuner.run()

    results_df: pd.DataFrame = analysis.results_df
    with open('results_tune_att.csv', 'w') as f:
        results_df.to_csv(f, index=False)
    with open('analysis_tune_att.pkl', 'wb') as f:
        pickle.dump(analysis, f)
    

