
# Setup
The following steps will lead you through the setup process for this repository. In case of any troubles during the setup and validation scripts, please feel free to contact the owner of the repository under paul.gross(at)philomatech.com.

## Configure project path 

Set the project path in `src\utils\SYSCONFIG.py`. This path must point to the directory wrapping the `src` folder. For example, if the src folder is at `your\path\ai-cup-caisar\src` please enter the following config. Specify your operating system type as well.
```python
PLATFORM = 'WINDOWS' # Choose from "UNIX" or "WINDOWS"
PROJECT_PATH = 'your\path\ai-cup-caisar'
```

## Install dependencies
Install the necessary dependencies by running the following commands. Your terminal's current working directory must be the `PROJECT_PATH`:
```shell
pipenv shell
pip install -r src/requirements.txt
pip install -e .
```
Please execute all future commands in the automatically created virtual environment.  
## Data setup
To run the data setup, execute the `src\scripts\data_setup_workflow.py` script. This script will create the necessary folder structure and also download and transform the training and submission data to suit the project. Therefore, please run the following command:
```python
python3 src\scripts\data_setup_workflow.py
```

The setup is now complete. 

# Training, Validation and Submission 
The main workflow of validation and submission is the `validation_submission_workflow.py` found at `src\session_validation\validation_submission_workflow.py`. Please note, that this process requires access to an AWS S3 bucket and is therfore not executable out of the box. 
## Model configurations
The best configurations retrieved from hyperoptimization can be found under `src\session_validation\session_params.py`. The dictionaries found under `SessionParams.` wrap all possible configuration options through multiple nested dictionaris. The parameters used for the best submission during the ai-cup is the `SessionParams.config_10`. However, this version is quite unstable in respect to weights initialization and does not always produce good results. The best stable version used for submissions is the version `SessionParams.config_6`. 
## Model validation
You can start a cross fold validation with this parameters by running the validation script found under `src\session_validation\validation\session_validator.py`: 

```python
python3 src\session_validation\validation\session_validator.py
```
You can change the version you want to validate at line 206 of the `src\session_validation\validation\session_validator.py` script. This script will produce and print the most important data as well as a json file with more detailed information.

## Create Submissions
Submissions are created using the `src\notebooks\submission\submission_workflow.py` script. You can change the configuration for the submission in line 126 of the script. The submissions will be saved to the `submissions` folder on the same layer as the `src` folder. 
If the `CREATE_VISUALIZATIONS` option is set to `True`, there will be an additional folder including some visualizations of the predictions for further analysis. 

# Hyperoptimization
To start local hyperoptimization, run the script found under `src\hyperoptimization\ray_tuner.py`. Using this script, you can configure the hyperoptimization parameters and also define the search space. To start the hyperoptimization, run the following line:
```python
python3 src\hyperoptimization\ray_tuner.py
```
Please keep in mind, that executing the hyperoptimization is resource-intensive. 
