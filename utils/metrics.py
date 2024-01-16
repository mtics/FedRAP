import math

import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        all_data = pd.DataFrame({'user': neg_users + test_users,
                                 'item': neg_items + test_items,
                                 'score': neg_scores + test_scores})
        all_data = pd.merge(all_data, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        all_data['rank'] = all_data.groupby('user')['score'].rank(method='first', ascending=False)
        all_data.sort_values(['user', 'rank'], inplace=True)

        top_k = all_data[all_data['rank'] <= self._top_k]
        self.test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        self.user_num = all_data['user'].nunique()

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""

        return len(self.test_in_top_k) * 1.0 / self.user_num

    def cal_ndcg(self):
        """NDCG @ top_K"""

        ndcg = self.test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        self.test_in_top_k['ndcg'] = ndcg

        return self.test_in_top_k['ndcg'].sum() * 1.0 / self.user_num
