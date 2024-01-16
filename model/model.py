import copy

import torch

from model.engine import Engine


class FedRAP(torch.nn.Module):
    def __init__(self, config):
        super(FedRAP, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def forward(self, item_indices):
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        logits = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(logits)

        return rating, item_personality, item_commonality


class FedRAPEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = FedRAP(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(FedRAPEngine, self).__init__(config)
        print(self.model)
