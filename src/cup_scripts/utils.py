import copy
import json
import os
import shutil
from pathlib import Path
from shutil import unpack_archive
from zipfile import ZipFile

import pandas as pd
import requests
from loadmydata.load_human_locomotion import (
    get_code_list,
    load_human_locomotion_dataset,
)
from loadmydata.utils import get_local_data_path
from tqdm import tqdm
from yarl import URL

SUBMISSION_FILE_NAME = "submission.txt"
TEST_DATA_URL = URL("https://plmbox.math.cnrs.fr/f/a78c07c38a864deeb85d/?dl=1")
TEST_DATA_NAME = "HumanLocomotionTest"
TEST_DATAFILE_NAME = "test.tar.gz"
N_TEST = 690


def load_json(filename: Path) -> dict:
    with open(filename, "r") as fp:
        res_dict = json.load(fp)
    return res_dict


def load_train():
    """Return the training data.

    Training data consist of 3 lists of same length:

    - the list of signals (each element is a Pandas dataframe),
    - the list of detected steps (each element is a list),
    - the list of metadata for each signal (each element is a dictionary).

    Returns:
        tuple: a tuple of three lists (X_train, y_train, metadata_train)
    """
    # This wil download the data on the first run
    _ = load_human_locomotion_dataset("1-1")

    # better data format
    X_train = list()
    y_train = list()
    metadata_train = list()
    for code in get_code_list():
        sensor_data = load_human_locomotion_dataset(code)
        signal = sensor_data["signal"]
        info_left = copy.deepcopy(sensor_data["metadata"])
        info_right = copy.deepcopy(sensor_data["metadata"])

        # left sensor
        X_train.append(
            signal.filter(regex="L[AR][XYZV]", axis="columns").rename(
                columns=lambda x: x[1:]
            )
        )
        y_train.append(sensor_data["left_steps"].tolist())
        info_left["SensorLocation"] = "Left"
        metadata_train.append(info_left)

        # right sensor
        X_train.append(
            signal.filter(regex="R[AR][XYZV]", axis="columns").rename(
                columns=lambda x: x[1:]
            )
        )
        y_train.append(sensor_data["right_steps"].tolist())
        info_right["SensorLocation"] = "Right"
        metadata_train.append(info_right)

    return X_train, y_train, metadata_train


def download_test_from_remote_human_locomotion() -> None:
    """Download and uncompress the (test) human locomotion data set."""
    local_cache_data = get_local_data_path(TEST_DATA_NAME)
    local_archive_path = local_cache_data / TEST_DATAFILE_NAME

    if not local_cache_data.exists():
        local_cache_data.mkdir(exist_ok=True, parents=True)

        # get archive's url
        remote_archive_path = TEST_DATA_URL
        response = requests.get(remote_archive_path, stream=True)
        # handle the download progress bar
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        # actual download
        with open(local_archive_path, "wb") as handle:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                handle.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"ERROR: the download of {TEST_DATAFILE_NAME} went wrong.")
        # uncompress the data in the data folder
        unpack_archive(filename=local_archive_path, extract_dir=local_cache_data)
        # remove zip file
        os.remove(local_archive_path)
        # Check if the extracted directory contains a single sub-directory and
        # no other file.
        directory_list = [x for x in local_cache_data.iterdir() if x.is_dir()]
        non_directory_list = [x for x in local_cache_data.iterdir() if not x.is_dir()]
        if len(directory_list) == 1 and len(non_directory_list) == 0:
            sub_dir = directory_list[0]
            for element in sub_dir.iterdir():
                shutil.move(str(element), str(sub_dir.parent))
            os.rmdir(str(sub_dir))


def load_test():
    """Return the testing data.

    Testing data consist of two lists of same length:

    - the list of signals (each element is a Pandas dataframe),
    - the list of metadata (each element is a dictionary).

    Returns:
        tuple: a tuple of two lists (X_test, metadata_test)
    """
    # download the test data on the first run
    download_test_from_remote_human_locomotion()
    # read and return the data
    local_cache_data = get_local_data_path(TEST_DATA_NAME)
    X_test = list()
    metadata_test = list()
    for ind in range(N_TEST):
        filename = local_cache_data / f"{ind}"
        X_test.append(pd.read_csv(filename.with_suffix(".csv"),index_col=False))
        metadata_test.append(load_json(filename.with_suffix(".json")))
    return (X_test, metadata_test)


def check_result_format(result_list) -> None:
    """Raise an exception if the result is badly formatted."""

    err_msg = f"The result should be a list, not {type(result_list)}."
    assert isinstance(result_list, list), err_msg

    err_msg = f"The result should be of length {N_TEST}, not {len(result_list)}."
    assert len(result_list) == N_TEST, err_msg

    for pred_steps in result_list:
        err_msg = (
            f"Each prediction should be a **list** of steps, "
            f"not {type(pred_steps)}."
        )
        assert isinstance(pred_steps, list), err_msg

        for steps in pred_steps:
            err_msg = (
                f"Each detected step should be a list [start_index, end_index], "
                f"not {steps}."
            )
            assert isinstance(steps, list) and len(steps) == 2, err_msg


def write_results(result_list, filename="submission.zip") -> None:
    """Write the results to a file."""

    check_result_format(result_list=result_list)

    # Write the results to a text file
    with open(SUBMISSION_FILE_NAME, "w") as res_file:
        json.dump(result_list, fp=res_file)

    # Compress the text file
    with ZipFile(filename, "w") as myzip:
        myzip.write(SUBMISSION_FILE_NAME)

    Path(SUBMISSION_FILE_NAME).unlink()


def read_results(filename="submission.zip") -> list:
    """Read the results from a zip file."""

    with ZipFile(filename) as myzip:
        with myzip.open(SUBMISSION_FILE_NAME) as myfile:
            result_list = json.load(myfile)
    
    # Check the results' format
    check_result_format(result_list=result_list)

    return result_list
