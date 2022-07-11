
from data_processing.data_processing_functions import annotate_all_files
from utils.SYSCONFIG import DATA_PATH
from typing import List, Tuple
import urllib.request
import os
from utils.SYSCONFIG import DATA_PATH, PLATFORM, DATA_GAIT
from utils.util_functions import printc, CustomBar, get_folder_size
from utils.path_utils import create_index_for_dir
import zipfile
import shutil
import data_generator.batch_data_path_generator as pg

DOWNLOAD_URL = "http://dev.ipol.im/~truong/GaitData.zip"
DOWNLOAD_URL = "https://gait-data.s3.eu-central-1.amazonaws.com/GaitData.zip"

def create_gait_dataset():

    def send_message(message: str, **kwargs) -> None:
        printc(source='[GAIT DATASET CREATOR]', message=message, **kwargs)

    
    ORIGINAL_DATA_DIR = pg.get_original_folder_path(data_subset_name=DATA_GAIT)
    ORIGINAL_INDEX_CSV = pg.get_original_csv_index_path(data_subset_name=DATA_GAIT)
    ORIGINAL_INDEX_JSON = pg.get_original_json_index_path(
        data_subset_name=DATA_GAIT)
    
        # region Find or Download original data
    ZIP_FILE_PATH = os.path.join(DATA_PATH, "GaitData.zip")
    zip_exists = os.path.exists(ZIP_FILE_PATH)

    download_necessary = False

    if zip_exists:
        send_message(f"The original data was found under {ZIP_FILE_PATH}")
        size = os.path.getsize(ZIP_FILE_PATH)
        if size == 201650885:
            send_message(f"The .zip file found has the proper size: {size} bytes")
            download_necessary = False
        else:
            send_message(
                f"The .zip file size does not match with the expected file size: {size} bytes. Redownloading the file is necessary")
            download_necessary = True
    else:
        download_necessary = True

    if download_necessary:

        bar = CustomBar('Download in progress: ', max=1000)

        def download_progress_hook(count, blockSize, totalSize):
            percent = int(count * blockSize * 1000 / totalSize)
            bar.goto(percent)

        send_message(
            f"There is no file found at {ZIP_FILE_PATH}. Starting to download the data via {DOWNLOAD_URL}")
        urllib.request.urlretrieve(
            DOWNLOAD_URL, filename=ZIP_FILE_PATH, reporthook=download_progress_hook)
        send_message(f"Successfully downloaded {ZIP_FILE_PATH}")

    # endregion

    # region Unzip the downloaded file

    # extracted size:             665243558
    # extracted & annotated size: 480469617

    extraction_necessary = True

    if os.path.exists(ORIGINAL_DATA_DIR):
        folder_size = get_folder_size(ORIGINAL_DATA_DIR)
        if folder_size == 665243558 or folder_size == 480469617:
            extraction_necessary = False
            send_message(
                f"The folder found under {ORIGINAL_DATA_DIR} has the proper size: {folder_size}")
        else:
            send_message(
                f"The folder under {ORIGINAL_DATA_DIR} has NOT the expected size ({folder_size}). (Re-)Extraction is necessary")
    else:
        send_message(
            f"There is no folder under {ORIGINAL_DATA_DIR}. (Re-)Extraction is necessary")

    if extraction_necessary:

        if os.path.exists(ORIGINAL_DATA_DIR):
            shutil.rmtree(ORIGINAL_DATA_DIR)

        send_message(f"Starting to extract files from zip file.")

        with zipfile.ZipFile(ZIP_FILE_PATH, "r") as zip_ref:
            send_message(f"Starting to unzip files")
            zip_ref.extractall(DATA_PATH)
            extracted_path = DATA_PATH + \
                ('\\' if PLATFORM == 'WINDOWS' else '/') + f'GaitData'
            os.rename(extracted_path, ORIGINAL_DATA_DIR)
    else:
        send_message(
            f"No data extraction is necessary, Data is ready under {ORIGINAL_DATA_DIR}")

        # Create index for original data

    index: List[str] = create_index_for_dir(
        dir_path=ORIGINAL_DATA_DIR, index_path=ORIGINAL_INDEX_CSV, file_ending=".csv")

    send_message(f"Created index for {ORIGINAL_DATA_DIR} at {ORIGINAL_INDEX_CSV}")

    create_index_for_dir(dir_path=ORIGINAL_DATA_DIR,
                        index_path=ORIGINAL_INDEX_JSON, file_ending=".json")
    send_message(f"Created index for {ORIGINAL_DATA_DIR} at {ORIGINAL_INDEX_JSON}")

    # endregion
    480469617
    # region annotation:

    folder_size = get_folder_size(ORIGINAL_DATA_DIR)
    if folder_size == 480469617:
        send_message(
            f"No annotation necessary as folder already got expected size for annotated files")
    else:
        send_message(
            f"Annotating all files at {ORIGINAL_INDEX_CSV} with the json index at {ORIGINAL_INDEX_JSON}")
        annotate_all_files(ORIGINAL_INDEX_CSV, ORIGINAL_INDEX_JSON)

