
from cProfile import run
from typing import Dict, List, Tuple
import os
import json
import collections.abc
from utils.SYSCONFIG import AWS_RAY_RESULTS

# only public method
def get_best_run(bucket_name: str, trainable_name: str, exclude_ids: List[str] = []) -> Tuple[float,int,str,Dict]:
    """_summary_

    Args:
        bucket_name (str): The name of the aws bucket
        trainable_name (str): The name of the trainable
        exclude_ids (List[str], optional): The ids to be excluded, for example ids that already have been analyzed

    Returns:
        Tuple[float,int,str]: [f-score, best_iteration, run_id, ]
    """
    target_dir = os.path.join(AWS_RAY_RESULTS,trainable_name)
    print(f'Starting to download synchronizing  ...')
    os.system(f'aws s3 sync s3://{bucket_name}/{trainable_name} {target_dir} --quiet')
    print(f'Sync completed')

    runs = load_runs(target_dir)
    score, iteration, id, params = get_run_with_best_score(runs, exclude_ids=exclude_ids)

    return score, iteration, id, params


def load_runs(runs_dir_path: str) -> List[Dict]:
    runs = []

    for x in os.walk(runs_dir_path):
        run_folder = x[0]

        params_path = os.path.join(run_folder,"params.json")
        result_path = os.path.join(run_folder,"result.json")

        if os.path.exists(params_path) and os.path.exists(result_path):
        
            p = open(params_path, encoding='utf-8')
            params = json.load(p)
            results =  [json.loads(line) for line in open(result_path, encoding='utf-8')]

            new_run = {
                'params': flatten(params),
                'results': results,
            }

            new_run_score, iteration, id, params  = get_score_of_run(new_run)

            if len(runs) == 0:
                runs.append(new_run)
                continue

            for i in range(len(runs)):
                run = runs[i]
                run_score, iteration, id, params = get_score_of_run(run)

                if run_score >= new_run_score:
                    runs.insert(i,new_run)
                    break

                if i == len(runs) - 1: 
                    runs.append(new_run)
    return runs


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_score_of_run(run) -> Tuple[float,int,str, Dict]:
    scores = [[x['f1score'], x['iterations_since_restore'], x['trial_id'], run['params']] for x in run['results']]

    if len(scores) == 0: return 0, 0, "NONE", {}

    # sort by first element: score, descending
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return scores[0]


def get_run_with_best_score(runs: List[Dict], exclude_ids: List[str] = []) -> Tuple[float,int,str, Dict]:
    run_scores = []

    for run in runs:
    
        score, iteration, id, params = get_score_of_run(run)

        if id in exclude_ids: continue

        run_scores.append([score, iteration, id, params])

    run_scores = sorted(run_scores, key=lambda x: x[0], reverse=True)
    return run_scores[0]


if __name__ == '__main__':
    SUBMISSION_BUCKET = 'second-ray-bucket'
    TRAINABLE_NAME = 'MergedTrainable_2022-07-08_22-00-00'

    get_best_run(bucket_name= SUBMISSION_BUCKET, trainable_name= TRAINABLE_NAME)

"""s3
s3.upload_file(
    Filename=full_filename,
    Bucket=SUBMISSION_BUCKET,
    Key=full_key
)"""