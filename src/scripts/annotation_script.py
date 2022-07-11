from data_processing.data_processing import annotate_all_files
from utils.SYSCONFIG import DATA_PATH

CSV_INDEX_FILE = f'{DATA_PATH}\\csv_index.txt'
JSON_INDEX_FILE = f'{DATA_PATH}\\json_index.txt'
annotate_all_files(CSV_INDEX_FILE, JSON_INDEX_FILE)