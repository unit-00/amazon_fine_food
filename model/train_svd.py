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
from typing import Dict

base_dir = Path(__file__).resolve(strict=True).parent

def train_and_dump_model(options: Dict, utility_matrix: pd.DataFrame) -> None:
    algo = SVD(**options)
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(utility_matrix, reader)

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    dump.dump(base_dir.joinpath('svd_algo.dump'), algo=algo)


if __name__ == '__main__':
    options = None
    
    while not isinstance(options, dict):
        options = input('Enter options for SVD: (Read Surprise doc for listed options)\ne.g.{\'n_epochs\': 5 ,\'verbose\':True}\n')
        options = ast.literal_eval(options)

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
   
    train_and_dump_model(options, utility_matrix)
    

