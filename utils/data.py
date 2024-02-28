import random
from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings

        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)

        # get the index pool od user_id and item_id
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        # create negative item samples for model learning
        # 99 negatives for each user's test item
        self.negatives = self._sample_negative(ratings)

        # divide all ratings into train and test two dataframes, which consit of userId, itemId and rating three columns.
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        data = deepcopy(ratings)
        max_rating = ratings.ratings.max()
        data['rating'] = data.ratings * 1.0 / max_rating
        return data

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        data = deepcopy(ratings)
        # data['rating'][data['rating'] > 0] = 1.0
        data.loc[data['rating'] > 0, 'rating'] = 1.0
        return data

    def _split_loo(self, ratings):
        """leave one out train/test split """

        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]

        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)

        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[
            ['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """store all the train data as a list including users, items and ratings. each list consists of all users'
        information, where each sub-list stores a user's positives and negatives"""
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')

        # include userId, itemId, rating, negative_items and negatives five columns.
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))

        # split train_ratings into groups according to userId.
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        users, items, ratings = [], [], []
        single_user, user_item, user_rating = [], [], []
        for userId, user_train_ratings in grouped_train_ratings:
            train_users.append(userId)
            user_length = len(user_train_ratings)
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # negative samples get 0 rating
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
            single_user = []
            user_item = []
            user_rating = []
        # assert len(users) == len(items) == len(ratings) == len(self.user_pool)
        assert train_users == sorted(train_users)
        return [users, items, ratings]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def validate_data(self):
        """create validation data"""
        val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [], []
        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        assert len(val_users) == len(val_items)
        assert len(negative_users) == len(negative_items)
        assert 99 * len(val_users) == len(negative_users)
        assert val_users == sorted(val_users)
        return [torch.LongTensor(val_users), torch.LongTensor(val_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

    @property
    def test_data(self):
        """create evaluate data"""
        # return four lists, which consist userId or itemId.
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        assert len(test_users) == len(test_items)
        assert len(negative_users) == len(negative_items)
        assert 99 * len(test_users) == len(negative_users)
        assert test_users == sorted(test_users)
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
