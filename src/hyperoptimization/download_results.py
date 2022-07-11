from utils.SYSCONFIG import AWS_RAY_RESULTS
from utils.util_functions import create_folder_if_not_exists
import os

bucket_name = 'ray-results-bucket'
bucket_name = 'second-ray-bucket'
bucket_path = '/synced_folder'

create_folder_if_not_exists(AWS_RAY_RESULTS)

os.system(f'aws s3 sync s3://{bucket_name} {AWS_RAY_RESULTS} --quiet')
