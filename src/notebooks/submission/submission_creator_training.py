from notebooks.submission.submission_model_trainer import execute_training
from session_validation.session_params import SessionParams
from session.session import Session


if __name__ == '__main__':    
    session, df = execute_training(**SessionParams.config_231)
