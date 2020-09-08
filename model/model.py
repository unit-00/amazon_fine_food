import pandas as pd
import numpy as np

import surprise
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import dump

import ast
import os
import joblib
from pathlib import Path
from typing import Dict, List

base_dir = Path(__file__).resolve(strict=True).parent

def train(dump_filename: str) -> None:
    """
    Train SVD model based on options using utility matrix,
    then dump model for future usage.
    """

    options = dict()
    
    try:
        options = input("Enter options for SVD: (Read Surprise doc for listed options)\ne.g.{'n_epochs': 5 ,'verbose':True}\n")
        options = ast.literal_eval(options)

    except:

        print('Malformed options, moving forward with options found through hyperparameter search\n')
        options = {'n_factors': 87,
        'n_epochs': 22,
        'lr_all': 0.06,
        'reg_all': 0.028888888888888888}

    filename = ''

    while not os.path.isfile(filename):
        filename = input('Enter filename to train on:\n')
    
    try:
        if filename.endswith('.gz'):
            utility_matrix = pd.read_csv(filename, 
                                         header=0,
                                         usecols=['review/userId', 'product/productId', 'review/score'],
                                         compression='gzip')

        else:
            utility_matrix = pd.read_csv(filename,
                                         header=0,
                                         usecols=['review/userId', 'product/productId', 'review/score'])
    except:
        raise Exception('File does not exist.')

    utility_matrix = utility_matrix[['review/userId', 'product/productId', 'review/score']]
   
    algo = SVD(**options)
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(utility_matrix, reader)

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    dump.dump(base_dir.joinpath(dump_filename), algo=algo)

def predict(dump_filename: str, raw_item_id: str, k: int) -> List[str]:
    """
    Return k most similar items through latent factors
    """

    _, loaded_algo = dump.load(base_dir.joinpath(dump_filename))

    inner_id = loaded_algo.trainset.to_inner_iid(raw_item_id)
    sorted_items = np.argsort(loaded_algo.qi[:, inner_id])[-1:-k-1:-1]

    return [loaded_algo.trainset.to_raw_iid(iid) for iid in sorted_items]