from time import sleep
from typing import Any, Dict, List
from hyperoptimization.trainables.base_trainable import BaseTrainable
import numpy as np
from utils.errors import WrongDataTypeInSearchSpaceError
from ray import tune
from utils.util_functions import printc


class DevTrainable(BaseTrainable):
    """Trainable for development purposes, that only aims at optimizing
    a simple mathematical problem

    Raises:
        WrongDataTypeInSearchSpaceError: Thrown if the data type of the config is unexpected 
    """
    
    metric = 'objective'
    mode = 'min'

    def objective(self, x: float, c: str) -> float:
        """simple mathematical objective to minimize

        Args:
            x (float): function variable
            c (str): function variable

        Returns:
            float: value of function
        """
        # f(x) : Min. at (-3, -5)
        f_x = (x + 2) ** 2 + 2 * x
        # g(x) = Min. at (3.688, -4.598)
        g_x = (x - 3) ** 2 - 1.375 * x
        
        sleep(2)
        return f_x if c == 'f' else g_x
    
    hyperopt_space = {
            'x' : tune.uniform(-10, 10),
            'c' : tune.choice(['f', 'g']),
        }
    
    bayes_space = {
            'x' : tune.uniform(-10, 10),
            # ['f', 'g']
            'c' : tune.uniform(0.0, 2.0),
        }
    
    def parse_config(self, config: Dict[str, Any]):
        '''
        parses the config: adapted to deal with all 
        available spaces and their specifics
        NOTE: for understandability the order of parsed args
              should match the order of definition in the search space
        NOTE: for readability parse/check for types in 
              the order str > float > np.float64 (Bayes) > bool > int
        '''
        
        c = config['c']
        if type(c) == str:
            pass
        elif type(c) == np.float64:
            config['c'] = 'f' if (c < 1.0) else 'g'
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='c',
                value=c,
                data_type=type(c),
                expected_types=[str, np.float64]
            )
            
        return config


    def setup(self, config) -> None:
        self.send_message(f'setup() was called')
        self.config = self.parse_config(config)

    def step(self):
        self.send_message(f'step() was called')
        score = self.objective(x=self.config['x'], c=self.config['c'])
        return {DevTrainable.metric : score}
    
    def save_checkpoint(self, checkpoint_dir):
        self.send_message(f'save_checkpoint() was called with arg checkpoint_dir={checkpoint_dir}')
        return checkpoint_dir
    
    def load_checkpoint(self, path):
        self.send_message(f'load_checkpoint() was called with arg path={path}')
        pass
    
    def reset_config(self, new_config) -> bool:
        self.send_message(f'reset_config() was called with arg new_config={new_config}')
        self.config = self.parse_config(new_config)
        return True
    
    def send_message(self, message: str, **kwargs) -> None:
        printc(source='[DevTrainable]', message=message, **kwargs)
