from data_processing.data_normalization_params import DataNormalizationParams
from utils.SYSCONFIG import DATA_SETS
from data_processing.data_normalization_utils import normalize


def normalize_all_datasets(exclude_datasets = []):
    
    for dataset_name in DATA_SETS:
        if dataset_name in exclude_datasets : continue
        for normalization_mode in DataNormalizationParams.options['normalization_mode']:
            for normalization_function in DataNormalizationParams.options['normalization_function']:
                normalize(
                    dataset_name = dataset_name,
                    normalization_mode = normalization_mode,
                    normalization_function = normalization_function
                )

                

# only if executed directly!
if __name__ == '__main__':
    normalize_all_datasets()
