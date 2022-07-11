from utils.util_functions import create_folder_if_not_exists
from utils.SYSCONFIG import DATA_PATH, AWS_RAY_RESULTS, DATA_SUBMISSION
from typing import List
from utils.SYSCONFIG import DATA_PATH, LOGS_PATH, MODELS_PATH, DATA_GAIT, DATA_SUBMISSION
from utils.util_functions import printc, create_folder_if_not_exists

from data_processing.normalization_metadata_setup import create_normalization_metadata

"""
This script unifies the complete data setup for both UNIX and Windows Systems.

Parameters: 
DOWNLOAD_URL: If none, the GaitData.zip file must be in the data directory 


"""

def data_setup_workflow(exclude_datasets: List[str] = []):
   
    def send_message(message: str, **kwargs) -> None:
        printc(source='[DATA SETUP WORKFLOW]', message=message, **kwargs)
    
    
    create_folder_if_not_exists(DATA_PATH)
    create_folder_if_not_exists(LOGS_PATH)
    create_folder_if_not_exists(MODELS_PATH)
    create_folder_if_not_exists(AWS_RAY_RESULTS)
    create_folder_if_not_exists(DATA_SUBMISSION)

    
    if not DATA_GAIT in exclude_datasets:
        from scripts.data_setup_helper.create_gait_dataset import create_gait_dataset
        create_gait_dataset()
    

    if not DATA_SUBMISSION in exclude_datasets:
        from scripts.data_setup_helper.create_submission_dataset import create_submission_dataset
        create_submission_dataset()
    

    # region normalization
    send_message(f"Starting adding normalization meta data...")
    create_normalization_metadata()


    send_message(f"Data normalization meta data setup complete.")
    send_message(f"Creating split index files for configs for both training and evaluation")


    # endregion
    #"""
    from scripts.data_setup_helper.index_file_creator import create_index_files
    create_index_files(exclude_datasets=exclude_datasets)


    send_message(f"Data Setup completed")


if __name__ == "__main__":
    data_setup_workflow()
