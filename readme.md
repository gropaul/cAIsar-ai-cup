# Introduction

The human gait is a complex mechanism that is subject to alteration by many pathologies that lead quickly to a loss of autonomy and an increased risk of falls. Analyzing and understanding human gait could lead to applications in early detection and harm prevention for patients. An integral part of understanding the gait itself is being able to detect it in unlabelled time series sensor data to efficiently generate data for subsequent developments.	The aim of this project & paper is to demonstrate the feasibility of accurately labeling steps in multi-variate time-series data recorded from foot-worn inertia measurement units using a state-of-the-art deep learning approach based on CNNs. 

The report can be found in the root directory of the repository.

# Setup
The following steps should lead you through the setup process for this repository. In case of any troubles during the setup and validation scripts, please feel free to contact the owner of the repository under paul.gross(at)philomatech.com.

## Configure project path 

Set the project path at `src\utils\SYSCONFIG.py`. This path must point at the directory wrapping the `src` folder. For example, if the src folder is at `your\path\ai-cup-caisar\src` please enter the following config and also the  type. 
```python
PLATFORM = 'WINDOWS' # Choose from "UNIX" or "WINDOWS"
PROJECT_PATH = 'your\path\ai-cup-caisar'
```

## Install dependencies
Install the necessary dependencies by running the following command. Your terminals current working directory must be the `PROJECT_PATH`:
```shell
pipenv shell
pip install -r src/requirements.txt
cd src
pip install -e .
```
Please use this virtual environment for the future commands.  
## Data setup
To run the data setup script, you can take the `src\scripts\data_setup_workflow.py` script. This script will create the necessary folder structure and also download and transform the training and submission data. Therefore, please run the following command:
```python
python3 src\scripts\data_setup_workflow.py
```

The setup should now be completed. 

# Training, Validation and Submission 
The main workflow of validation and submission is the `validation_submission_workflow.py` found at `src\session_validation\validation_submission_workflow.py`. As this process requires access to an AWS S3 bucket for retrieving hyperoptimization results, it is not publicly executable. 
## Model configurations
The best configurations retrieved from hyperoptimization can be found under `src\session_validation\session_params.py`. The dictionaries found under `SessionParams.` wrap all possible configuration options through multiple nested lists. The parameter used for the best submission during the ai-cup is the `SessionParams.config_10`. However, this version is quiet unstable and not always produces good results. The best stable version used for submissions is the version `SessionParams.config_6`. 
## Model validation
You can start a cross fold validation with this parameters by running the validation script found under `src\session_validation\validation\session_validator.py`: 

```python
python3 src\session_validation\validation\session_validator.py
```
You can change the version you want to validate at line 206 of the `src\session_validation\validation\session_validator.py` script. This script will produces and console print with the most important data as well as a json file with more detailed information.

## Create Submissions
Submissions are created using the `src\notebooks\submission\submission_workflow.py` script. You can change the configuration for the submission in the line 126 of the script. The submissions will be saved to the `submissions` folder on the same layer as the `src` folder. 
If the `CREATE_VISUALIZATIONS` option is set to `True`, the will be an folder including some visualizations of the predictions for further analysis. 

# Hyperoptimization
To start local hyperoptimization, run the script found under `src\hyperoptimization\ray_tuner.py`. Using this script, you can configure the hyperoptimization parameters and also define the search space. To start the hyperoptimization, run the following line:
```python
python3 src\hyperoptimization\ray_tuner.py
```
Please keep in mind, that hyperoptimization is very resource intensive. 
