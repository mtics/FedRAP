import copy
import math
import random

import torch
from torch.utils.data import DataLoader

from utils.data import UserItemRatingDataset
from utils.metrics import MetronAtK


class Engine(object):
    """
    Meta Engine for training & evaluating our model

    Note: Subclass should implement self.model!

    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.server_model_param = {}
        self.client_model_params = {}
        self.sparsity = None
        self._metron = MetronAtK(top_k=self.config['top_k'])

        if config['vary_param'] == 'fixed':
            self.lam = config['lambda']
            self.mu = config['mu']
        else:
            self.lam = 0
            self.mu = 0

    def instanceUserTrainLoader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def federatedTrainSingleBatch(self, model_client, batch_data, optimizer, scheduler):
        """train a batch and return an updated model."""
        from model.loss import Loss

        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        model_loss = Loss(self.config)
        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            model_loss = model_loss.cuda()

        # optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred, item_personality, item_commonality = model_client(items)
        loss = model_loss(ratings_pred.view(-1), ratings, item_personality, item_commonality)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if self.config['use_cuda']:
            loss = loss.cpu()

        return model_client, loss.item()

    def aggregateParamsFromClients(self, client_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.

        t = 0
        for user in client_params.keys():
            # load a user's parameters.
            user_params = client_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                self.server_model_param['item_commonality.weight'].data += user_params[
                    'item_commonality.weight'].data
            t += 1

        self.server_model_param['item_commonality.weight'].data = self.server_model_param[
                                                                      'item_commonality.weight'].data / len(
            client_params)

        data = copy.deepcopy(self.server_model_param['item_commonality.weight'].data)
        sparsity = 1 - torch.count_nonzero(torch.abs(data) > 1e-4) / data.numel()

        return data, sparsity.item()

    def federatedTrainOneRound(self, train_data, item_commonality, iteration):
        """train a round."""
        # sample users participating in single round.

        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)

        # store users' model parameters of current round.
        participant_params = {}
        # store all the users' train loss/
        losses = {}

        # perform model update for each participated user.
        for user in participants:
            loss = 0

            # for the first round, client models copy initialized parameters directly.
            # copy the client model architecture from self.model
            client_model = copy.deepcopy(self.model)
            client_model.setItemCommonality(item_commonality)

            # for other rounds, client models receive updated item embedding and score function from server.
            if iteration != 0:
                client_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        client_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
                client_param_dict['item_commonality.weight'] = copy.deepcopy(
                    self.server_model_param['item_commonality.weight'].data)

                if self.config['use_cuda']:
                    for key in client_param_dict.keys():
                        client_param_dict[key] = client_param_dict[key].cuda()

                client_model.load_state_dict(client_param_dict)

            # Defining optimizers
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.SGD([
                {'params': client_model.affine_output.parameters(), 'lr': self.config['lr_network']},
                {'params': client_model.item_personality.parameters(), 'lr': self.config['lr_args']},
                {'params': client_model.item_commonality.parameters(), 'lr': self.config['lr_args']},
            ],
                weight_decay=self.config['l2_regularization']
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

            # load current user's training data and instance a train loader.
            user_train_data = [train_data[0][user], train_data[1][user], train_data[2][user]]
            user_dataloader = self.instanceUserTrainLoader(user_train_data)

            client_model.train()
            sample_num = 0
            # update client model.
            client_losses = []
            for epoch in range(self.config['local_epoch']):

                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    client_model, client_loss = self.federatedTrainSingleBatch(client_model, batch, optimizer,
                                                                               scheduler)
                    loss += client_loss * len(batch[0])
                    sample_num += len(batch[0])

                losses[user] = loss / sample_num
                client_losses.append(loss / sample_num)

                # check convergence
                if epoch > 0 and abs(client_losses[epoch] - client_losses[epoch - 1]) / abs(
                        client_losses[epoch - 1]) < self.config['tol']:
                    break

            # obtain client model parameters,
            # and store client models' local parameters for personalization.
            self.client_model_params[user] = copy.deepcopy(client_model.state_dict())
            if self.config['use_cuda']:
                for key in self.client_model_params[user].keys():
                    self.client_model_params[user][key] = self.client_model_params[user][key].cpu()

            # store client models' local parameters for global update.
            participant_params[user] = copy.deepcopy(self.client_model_params[user])

            # delete all user-related data
            del participant_params[user]['affine_output.weight']
            del participant_params[user]['affine_output.bias']
            del participant_params[user]['item_personality.weight']

        # aggregate client models in server side.
        data, sparsity = self.aggregateParamsFromClients(participant_params)

        # update global learning rates
        self.config['lr_network'] = self.config['lr_network'] * self.config['decay_rate']
        self.config['lr_args'] = self.config['lr_args'] * self.config['decay_rate']

        if self.config['vary_param'] == 'tanh':
            self.lam = math.tanh(iteration / 10) * self.config['lambda']
            self.mu = math.tanh(iteration / 10) * self.config['mu']
        elif self.config['vary_param'] == 'sin':
            self.lam = (math.sin(iteration / 10) + 1) / 2 * self.config['lambda']
            self.mu = (math.sin(iteration / 10) + 1) / 2 * self.config['mu']
        elif self.config['vary_param'] == 'square':
            if iteration % 5 == 0:
                self.lam = 0 if self.lam == self.config['lambda'] else self.config['lambda']
                self.mu = 0 if self.mu == self.config['mu'] else self.config['mu']
        elif self.config['vary_param'] == 'frac':
            self.lam = 1 / (iteration + 1) * self.config['lambda']
            self.mu = 1 / (iteration + 1) * self.config['mu']

        return losses, sparsity

    @torch.no_grad()
    def federatedEvaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()

        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None

        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)

            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data
            else:
                user_param_dict = copy.deepcopy(self.model.state_dict())

            user_param_dict['item_commonality.weight'] = copy.deepcopy(
                self.server_model_param['item_commonality.weight'].data)

            user_model.load_state_dict(user_param_dict)

            user_model.eval()

            # obtain user's positive test information.
            test_item = test_items[user: user + 1]
            # obtain user's negative test information.
            negative_item = negative_items[user * 99: (user + 1) * 99]
            # perform model prediction.
            test_score, _, _ = user_model(test_item)
            negative_score, _, _ = user_model(negative_item)

            if user == 0:
                test_scores = test_score
                negative_scores = negative_score
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))

        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()

        return hr, ndcg
