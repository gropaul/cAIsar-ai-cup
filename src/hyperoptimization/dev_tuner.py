import pickle
from hyperoptimization.trainables.dev_trainable import DevTrainable
import pandas as pd
from ray.tune import run, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperoptimization.trainables.base_trainable import BaseTrainable

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class DevTuner:
    """Development Tuner class to facilitate the usage of ray (tune);
    intended to be used with DevTrainable
    """
    def __init__(self, trainable: BaseTrainable, scheduler, searcher) -> None:
        self.trainable = trainable
        self.scheduler = scheduler
        self.searcher = searcher
        
    def run(self) -> ExperimentAnalysis:
        analysis: ExperimentAnalysis = run(
            self.trainable,
            scheduler=self.scheduler,
            search_alg=self.searcher,
            config=self.trainable.bayes_space,
            time_budget_s=500,
            mode=self.trainable.mode,
            metric=self.trainable.metric,
            num_samples=10,
            max_concurrent_trials=1,
            )
        return analysis
    

if __name__ == '__main__':
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=1,
        grace_period=1,
        reduction_factor=3, 
        brackets=1)
    
    searcher = BayesOptSearch()

    tuner = DevTuner(trainable=DevTrainable, scheduler=scheduler, searcher=searcher)
    analysis: ExperimentAnalysis = tuner.run()

    results_df: pd.DataFrame = analysis.results_df
    print(results_df)
    # with open('results_tune_1st.csv', 'w') as f:
    #     results_df.to_csv(f, index=False)
    # with open('analysis_tune_1st.pkl', 'wb') as f:
    #     pickle.dump(analysis, f)
    

