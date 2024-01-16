"""
    Some handy functions for pytroch model training ...
"""
import logging

import numpy as np
import scipy.sparse as sp
import torch


# Checkpoints
def saveCheckPoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resumeCheckPoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(
                                device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def initLogging(log_file_name):
    """Init for logging"""
    import logging
    import coloredlogs

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=log_file_name,
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    coloredlogs.install()


def setSeed(seed=0):
    """Set all random seeds"""

    import random
    import numpy as np
    import torch

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def datasetFilter(ratings, min_items=5):
    """
            Only keep the data useful, which means:
                - all ratings are non-zeros
                - each user rated at least {self.min_items} items
            :param ratings: pd.DataFrame
            :param min_items: the least number of items user rated
            :return: filter_ratings: pd.DataFrame
            """

    # filter unuseful data
    ratings = ratings[ratings['rating'] > 0]

    # only keep users who rated at least {self.min_items} items
    user_count = ratings.groupby('uid').size()
    user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
    filter_ratings = ratings[user_subset].reset_index(drop=True)

    del ratings

    return filter_ratings


def loadData(path, dataset, config, file_name='ratings.dat'):
    import os
    import pandas as pd

    dataset_file = os.path.join(path, dataset, file_name)

    min_rates = 10

    if dataset == "movielens":
        ratings = pd.read_csv(dataset_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')
    elif dataset == "amazon":
        if "product_reviews" not in file_name:
            ratings = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')
        else:
            ratings = pd.read_csv(dataset_file, sep=",", header=None, usecols=[0, 1, 17],
                                  names=['uid', 'mid', 'rating'], engine='python')

    elif dataset == "books":

        min_rates = 5

        ratings = pd.read_csv(dataset_file, sep=",", header=1, usecols=[3, 4, 6], names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    elif dataset == "last.fm":
        min_rates = 10

        ratings = pd.read_csv(dataset_file, sep="\t", header=None, usecols=[0, 1, 2], names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    elif dataset == "tmall":
        ratings = pd.read_csv(dataset_file, sep=",", header=None, usecols=[0, 2, 4, 5],
                              names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

    elif dataset == "kgrec":
        ratings = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    elif dataset == "user-behavior":
        chunks = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'cid', 'behavior', 'timestamp'],
                             engine='python', chunksize=1000000)

        all_chunks = []
        for chunk in chunks:
            chunk
            chunk.loc[chunk['behavior'] == 'pv', 'rating'] = 1
            chunk.loc[chunk['behavior'] == 'cart', 'rating'] = 2
            chunk.loc[chunk['behavior'] == 'fav', 'rating'] = 3
            chunk.loc[chunk['behavior'] == 'buy', 'rating'] = 4
            all_chunks.append(chunk)

        ratings = pd.concat(all_chunks)

    elif dataset == "tenrec":

        chunks = pd.read_csv(dataset_file, sep=",", header=1, usecols=[0, 1, 2],
                             names=['uid', 'mid', 'rating'],
                             engine='python', chunksize=1000000)

        all_chunks = []
        for chunk in chunks:
            all_chunks.append(chunk)

        ratings = pd.concat(all_chunks)

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    else:
        ratings = pd.DataFrame()

    # for synthetic analysis
    # you need to pre-process data
    # ratings = top_N_items(ratings, 1000)
    if config['what'] == 'user':
        ratings = top_N_users(ratings, config['num'], config['start'])
    elif config['what'] == 'item':
        ratings = top_N_items(ratings, config['num'], config['start'])

    ratings = datasetFilter(ratings, min_rates)

    # Reindex user id and item id
    user_id = ratings[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

    item_id = ratings[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

    ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)

    num_users, num_items = print_statistics(ratings)

    return ratings, num_users, num_items


def top_N_items(rating, n, start_idx=0):
    rated_num = rating.groupby(['mid'])['uid'].count()

    sorted_idx = rated_num.sort_values(ascending=False).index.tolist()

    first_n_item = sorted_idx[start_idx:start_idx + n]

    return rating.loc[rating['mid'].isin(first_n_item)]


def top_N_users(rating, n, start_idx=0):
    rated_num = rating.groupby(['uid'])['mid'].count()

    sorted_idx = rated_num.sort_values(ascending=False).index.tolist()

    first_n_item = sorted_idx[start_idx:start_idx + n]

    return rating.loc[rating['uid'].isin(first_n_item)]


def print_statistics(ratings):
    """print the statistics of the dataset, and return the number of users and items"""
    maxs = ratings.max()
    num_interactions = len(ratings)
    sparsity = 1 - num_interactions / ((maxs['userId'] + 1) * (maxs['itemId'] + 1))

    logging.info('The number of users: {}, and of items: {}.'.format(int(maxs['userId'] + 1), int(maxs['itemId'] + 1)))
    logging.info('There are total {} interactions, the sparsity is {:.2f}%.'.format(num_interactions, sparsity * 100))

    return int(maxs['userId'] + 1), int(maxs['itemId'] + 1)


def get_inter_matrix(df, args, form='coo'):
    '''
    get the whole sparse interaction matrix
    '''
    print("get the whole sparse interaction matrix")
    user_num, item_num = args['num_users'], args['num_items']

    src, tar = df['userId'].values, df['itemId'].values
    data = df['rating'].values

    mat = sp.coo_matrix((data, (src, tar)), shape=(user_num, item_num))

    if form == 'coo':
        return mat
    elif form == 'csr':
        return mat.tocsr()
    else:
        raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented...')
