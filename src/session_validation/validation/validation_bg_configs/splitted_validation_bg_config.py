from utils.SYSCONFIG import  CROSS_FOLD_TRAINING_SUBSETS, CROSS_FOLD_VALIDATION_SUBSETS
from data_generator.batch_generator_params import BatchGeneratorParams

from typing import List, Tuple

def create_splitted_batch_generator_configs(session_config) -> List[Tuple[BatchGeneratorParams,BatchGeneratorParams]]:

    bg_train_config = session_config['bg_train_config']
    bg_val_config = session_config['bg_evaluation_config']

    new_configs = []

    print(CROSS_FOLD_TRAINING_SUBSETS)

    for i in range(len(CROSS_FOLD_TRAINING_SUBSETS)): 

        new_bg_train_config = bg_train_config.copy()
        new_bg_train_config.update({
            'data_subset_name' : CROSS_FOLD_TRAINING_SUBSETS[i],
        })

        new_bg_val_config = bg_val_config.copy()
        new_bg_val_config.update({
            'data_subset_name' : CROSS_FOLD_VALIDATION_SUBSETS[i],
        })

        new_configs.append(
            (new_bg_train_config, new_bg_val_config)
        )

    return new_configs

if __name__ == '__main__': 
    from session_validation.session_params import SessionParams
    bgs = create_splitted_batch_generator_configs(SessionParams.config_1)
    print(bgs)
    print(len(bgs))