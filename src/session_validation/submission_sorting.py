from utils.SYSCONFIG import AWS_RAY_RESULTS, SUBMISSIONS_PATH
from utils.util_functions import create_folder_if_not_exists
import os
import shutil
local_path_raw = os.path.join(SUBMISSIONS_PATH,"FinalSubmissions","raw")
local_path_sorted = os.path.join(SUBMISSIONS_PATH,"FinalSubmissions","sorted")

create_folder_if_not_exists(local_path_raw)
create_folder_if_not_exists(local_path_sorted)

os.system(f'aws s3 sync s3://ray-submissions-v2 {local_path_raw}/ --quiet')

from os import walk

subfolders = [ f.path for f in os.scandir(local_path_raw) if f.is_dir() ]


for folder in subfolders:
    paths = next(walk(folder), (None, None, []))[2]  # [] if no file
    paths = [os.path.join(folder,x) for x in paths]
    for file_path in paths:
        if not 'FULL' in file_path: continue

        filename = os.path.basename(file_path)
        target_path = os.path.join(local_path_sorted,filename)

        shutil.move(file_path,target_path) 