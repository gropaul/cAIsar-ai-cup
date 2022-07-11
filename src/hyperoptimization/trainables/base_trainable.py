from typing import Any, Dict
from ray import tune

class BaseTrainable(tune.Trainable):
    hyperopt_space: Dict[str, Any] = None
    bayes_space: Dict[str, Any] = None
    mode: str = 'min'
    metric: str = 'metric'

    