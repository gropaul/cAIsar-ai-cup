

from utils.SYSCONFIG import DATA_SUBSET_CONFIG, DATA_SUBSETS
from utils.util_functions import printc, parse_index, save_index
import data_generator.batch_data_path_generator as pg


def create_index_files(exclude_datasets = []):

    def send_message(message: str, **kwargs) -> None:
        printc(source='[INDEX FILE CREATOR]', message=message, **kwargs)
    
    from utils.SYSCONFIG import DATA_SETS

    for dataset_name in DATA_SETS:
        if dataset_name in exclude_datasets : continue
        for data_subset_name in DATA_SUBSETS:

            data_subset_config = DATA_SUBSET_CONFIG[data_subset_name]
            subset_dataset_name = data_subset_config['data']

            if subset_dataset_name != dataset_name: continue
            if data_subset_name == dataset_name: continue

            send_message(f"Splitting the index under the following parameters:\n{dataset_name}, {data_subset_name}")
            
            original_index_path = pg.get_csv_index_path(data_subset_name=dataset_name)
            subset_index_path = pg.get_csv_index_path(data_subset_name=data_subset_name)

            create_index_for_subset(data_subset_config, original_index_path, subset_index_path)

            original_index_path = pg.get_json_index_path(data_subset_name=dataset_name)
            subset_index_path = pg.get_json_index_path(data_subset_name=data_subset_name)

            create_index_for_subset(data_subset_config, original_index_path, subset_index_path)

            original_index_path = pg.get_norm_meta_index_path(data_subset_name=dataset_name)
            subset_index_path = pg.get_norm_meta_index_path(data_subset_name=data_subset_name)

            create_index_for_subset(data_subset_config, original_index_path, subset_index_path)


def create_index_for_subset(data_subset_config,  original_index_path, subset_index_path  ):
    
    start, end = data_subset_config['start'], data_subset_config['end']
    inverse = data_subset_config['inverse'] if 'inverse' in data_subset_config else False
    print(data_subset_config)
    original_index = parse_index(original_index_path)
    print('Creating new Subset index:')
    print(f'original path: {original_index_path},')
    print(f'subset path: {subset_index_path},')
    print(f'inverse: {inverse}, start: {start} end: {end}')

    split_index_start = max(round(len(original_index) * start),0)
    print(f'{round(len(original_index) * end)} - {len(original_index)}')

    split_index_end = min(round(len(original_index) * end),len(original_index))

    if not inverse:
        index_subset = original_index[split_index_start:split_index_end]
    else:
        index_subset = original_index[:split_index_start] + original_index[split_index_end:]

    save_index(subset_index_path, index_subset)


if __name__ == '__main__':
    create_index_files()