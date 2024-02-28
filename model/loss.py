import torch


class Loss(torch.nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config

        self.crit = torch.nn.BCELoss()
        self.independency = torch.nn.MSELoss()

        if self.config['regular'] == 'l2':
            self.reg = torch.nn.MSELoss()
        elif self.config['regular'] == 'l1':
            self.reg = torch.nn.L1Loss()
        else:
            self.reg = torch.nn.MSELoss()

    def forward(self, predictions, truth, item_personality, item_commonality, ):

        if self.config['regular'] == 'l2':
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)
        elif self.config['regular'] == 'l1':
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)
        elif self.config['regular'] == 'none':
            self.config['mu'] = 0
            dummy_target = item_commonality
            third = self.reg(item_commonality, dummy_target)
        elif self.config['regular'] == 'nuc':
            third = torch.norm(item_commonality, p='nuc')
        elif self.config['regular'] == 'inf':
            third = torch.norm(item_commonality, p=float('inf'))
        else:
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)

        loss = self.crit(predictions, truth) \
               - self.config['lambda'] * self.independency(item_personality, item_commonality) \
               + self.config['mu'] * third

        return loss
