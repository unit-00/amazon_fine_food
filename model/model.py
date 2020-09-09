import pandas as pd
import numpy as np

from surprise import SVD, KNNBaseline
from surprise import Dataset, Reader
from surprise import prediction_algorithms, AlgoBase, Trainset
from surprise import dump

import ast
import joblib
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Type

base_dir = Path(__file__).resolve(strict=True).parent
base_data_dir = base_dir.joinpath('data')

random.seed(42)
np.random.seed(42)

def train(data: pd.DataFrame):
    """
    Train SVD model based on options using utility matrix,
    then dump prediction and algorithm for future usage.
    """
    # getting data in dataframe and then filtering the data down
    filtered_items = get_items_with_n_ratings(data, 100)
    mask = data['itemId'].isin(filtered_items)
    utility_matrix = data.loc[mask, ['userId', 'itemId', 'rating']]
    
    svd_options = {'n_factors': 82, 'n_epochs': 33, 'lr_all': 0.115, 'reg_all': 0.02}
    knn_options = {'sim_options': {'name': 'pearson', 
                                   'min_support': 4, 
                                   'user_based': False},
                    'k': 35,
                    'min_k': 1}
    # setup the algorithm
    svd_algo = SVD(**svd_options)
    knn_algo = KNNBaseline(**knn_options)
    reader = Reader(line_format='user item rating', rating_scale=(1,5))
    ds = Dataset.load_from_df(utility_matrix, reader)

    # data building
    trainset = ds.build_full_trainset()
    testset = trainset.build_anti_testset()
    print('Dataset build completed')


    # train and dump
    _train_and_dump('svd', svd_algo, trainset, testset)
    _train_and_dump('knn', knn_algo, trainset)
    print('Training and dumping completed')

def _train_and_dump(name: str, algo: Type[AlgoBase], train: Trainset, test: List[Tuple[str, str, float]] = None):
    """
    Helper function for training to deallocate memory
    """
    preds = None

    # train
    algo.fit(train)
    print("Training in helper function completed")

    # anti test, user-item pairs that do not exist in original dataset
    if name == 'svd':
        preds = test
        print('Predictions completed')

    dump.dump(base_dir.joinpath(name + '.dump'), predictions=preds, algo=algo)

def predict(raw_id: str, n: int = 10):
    """
    Return top n recommendations for user
    """

    if len(raw_id) == 10:
        dump_filename = 'knn.dump'

        loaded_predictions, loaded_algo = dump.load(base_dir.joinpath(dump_filename))

        predictions = get_item_rec(loaded_algo, raw_id, n)

    else:
        dump_filename = 'svd.dump'
    
        loaded_predictions, loaded_algo = dump.load(base_dir.joinpath(dump_filename))
    
        predictions = get_user_rec(loaded_algo, loaded_predictions, raw_id, n)

    return predictions

def get_user_rec(algo: Type[AlgoBase], predictions: List[Tuple[str, str, float]], raw_uid: str, n: int = 10):
    """
    Returns top n recommendations for all users
    """

    user_check = lambda u: u[0] == raw_uid

    top_n = []

    for uid, iid, def_r in filter(user_check, predictions):
        preds = algo.predict(uid, iid)
        top_n.append((preds.iid, preds.est))

    top_n.sort(key=lambda x: x[1], reverse=True)

    return top_n[:n]

def get_item_rec(algo: Type[AlgoBase], raw_iid: int, n: int = 10):
    iid = algo.trainset.to_inner_iid(raw_iid)
    return [algo.trainset.to_raw_iid(good_neighbor) for good_neighbor in algo.get_neighbors(iid, n)]
    

def get_data(filename: str):
    """
    Prepare data into dataframe for model,
    Returns data as a dataframe with userId, itemId, and rating columns
    """
    user_col = 'review/userId'
    item_col = 'product/productId'
    rating_col = 'review/score'
    
    try:
        if filename.endswith('.gz'):
            utility_matrix = pd.read_csv(base_data_dir.joinpath(filename), 
                                         header=0,
                                         usecols=[user_col, item_col, rating_col],
                                         compression='gzip')

        else:
            utility_matrix = pd.read_csv(base_data_dir.joinpath(filename),
                                         header=0,
                                         usecols=[user_col, item_col, rating_col])
    except:
        raise Exception('File does not exist.')

    return utility_matrix.rename(columns={
                                   user_col:'userId',
                                   item_col: 'itemId',
                                   rating_col: 'rating'})

def get_items_with_n_ratings(u_mat: pd.DataFrame, n: int):
    """
    Filter down items with less than n number of ratings,
    Returns items with greater than n ratings
    """

    # Filter down data due to lack of resources on computer
    item_rating_count = u_mat['itemId'].value_counts()
    greater_than_n_ratings = item_rating_count[item_rating_count >= n].index

    return greater_than_n_ratings