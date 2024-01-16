import argparse
import datetime
import logging
import os
import time

import numpy as np
import requests
import torch

from utils.data import SampleGenerator
from utils.utils import setSeed, initLogging, loadData, get_inter_matrix


def loadEngine(configuration, df=None):
    # Load engine according to the alias
    if configuration['alias'] == 'FedRAP':
        from model.model import FedRAPEngine

        load_engine = FedRAPEngine(configuration)
    elif configuration['alias'] == 'PFedRec':
        from compare.PFedRec.pfedrec import PFedRecEngine

        load_engine = PFedRecEngine(configuration)

    elif config['alias'] == 'FedMF':
        from compare.FedMF.mf import FedMFEngine

        load_engine = FedMFEngine(config)

    elif configuration['alias'] == 'FedNCF':
        from compare.FedNCF.neumf import NeuMFEngine

        configuration['latent_dim_mf'] = 32  # 8
        configuration['latent_dim_mlp'] = 32  # 8
        configuration['layers'] = [64, 128, 64, 32, 16, 8]
        configuration['optimizer'] = 'adam'
        configuration['adam_lr'] = configuration['lr_network']
        configuration['pretrain'] = False

        load_engine = NeuMFEngine(configuration)

    elif configuration['alias'] == 'NCF':
        from compare.NCF.neumf import NeuMFEngine

        configuration['latent_dim_mf'] = 32
        configuration['latent_dim_mlp'] = 32
        configuration['layers'] = [64, 128, 64, 32, 16, 8]
        configuration['optimizer'] = 'adam'
        configuration['adam_lr'] = configuration['lr_network']
        configuration['pretrain'] = False

        load_engine = NeuMFEngine(configuration)

    elif configuration['alias'] == 'MF':
        from compare.NCF.gmf import GMFEngine

        configuration['latent_dim'] = 32
        configuration['optimizer'] = 'adam'
        configuration['adam_lr'] = configuration['lr_network']
        configuration['pretrain'] = False

        load_engine = GMFEngine(configuration)

    elif configuration['alias'] == 'CFedMF':
        from compare.CFedMF.CFedMF import CFedMFEngine

        load_engine = CFedMFEngine(configuration)

    elif configuration['alias'] == 'DFedMF':
        from compare.DFedMF.DFedMF import DFedMFEngine

        load_engine = DFedMFEngine(configuration)

    elif configuration['alias'] == 'FedRAP-L2':

        from model.model import FedRAPEngine

        configuration['regular'] == 'l2'

        load_engine = FedRAPEngine(configuration)

    elif configuration['alias'] == 'FedRAP-No':
        from model.model import FedRAPEngine

        configuration['mu'] == 0

        load_engine = FedRAPEngine(configuration)

    elif configuration['alias'] == 'lightGCN':
        from compare.lightGCN.lightGCN import LightGCNEngine

        configuration['optimizer'] = 'adam'

        load_engine = LightGCNEngine(configuration, df)

    return load_engine


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--alias', type=str, default='FedRAP')
    parser.add_argument('--dataset', type=str, default='kgrec')
    parser.add_argument('--data_file', type=str, default='music.csv')
    parser.add_argument('--model_dir', type=str, default='results/checkpoints/{}/{}/[{}]Epoch{}.model')
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--decay_rate', type=float, default=0.97)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr_network', type=float, default=1e-1)
    parser.add_argument('--lr_args', type=float, default=1e2)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--num_negative', type=int, default=4)
    parser.add_argument('--lambda', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--regular', type=str, default='l1')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--type', type=str, default='seed')
    parser.add_argument('--comment', type=str, default='default')
    parser.add_argument('--on_server', type=bool, default=False)
    parser.add_argument('--vary_param', type=str, default='tanh')
    parser.add_argument('--num', type=int, default='100')
    parser.add_argument('--start', type=int, default='0')
    parser.add_argument('--what', type=str, default='None')

    args = parser.parse_args()

    # Config
    config = vars(args)

    # Set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_id'])

    torch.cuda.set_device(0)

    # Set random seed
    setSeed(config['seed'])

    # Logging.
    path = 'logs/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file_name = os.path.join(path,
                                 '[{}]-[{}.{}]-[{}.{}]-[{}].txt'.format(config['alias'], config['dataset'],
                                                                        config['data_file'].split('.')[0],
                                                                        config['type'], config['comment'],
                                                                        current_time))
    initLogging(log_file_name)

    # Load Data
    ratings, config['num_users'], config['num_items'] = loadData('../datasets', config['dataset'], config,
                                                                 config['data_file'])

    # create folder to save checkpoints
    checkpoint_path = 'results/checkpoints/{}/{}/'.format(config['alias'], config['dataset'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ratings)

    if config['alias'] not in ['NCF', 'MF', 'lightGCN']:
        train_data = sample_generator.store_all_train_data(config['num_negative'])
    else:
        train_data = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        if config['alias'] == 'lightGCN':
            config['inter_matrix'] = get_inter_matrix(ratings, config)

    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data

    # Load engine
    engine = loadEngine(config, ratings)

    logging.info(str(config))

    # Initialize for training
    test_hrs, test_ndcgs, val_hrs, val_ndcgs, train_losses = [], [], [], [], []
    best_test_hr, final_test_round = 0, 0
    sparsity = []

    item_commonality = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['latent_dim'])
    if config['use_cuda']:
        item_commonality = item_commonality.cuda()

    times = []

    for iteration in range(config['num_round']):

        logging.info('--------------- Round {} starts ! ---------------'.format(iteration + 1))

        # 1. Train Phase
        start_time = time.perf_counter()
        train_loss, sparse_value = engine.federatedTrainOneRound(train_data, item_commonality, iteration)
        end_time = time.perf_counter()

        times.append((end_time - start_time))

        logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                                 config['data_file'],
                                                                 config['alias'],
                                                                 (end_time - start_time)))

        loss = sum(train_loss.values()) / len(train_loss.keys())
        train_losses.append(loss)
        sparsity.append(sparse_value)

        logging.info(
            '[Epoch {}/{}][Train] Loss = {:.4f}, Sparsity = {:.4f}'.format(iteration + 1, config['num_round'], loss,
                                                                           sparse_value))

        # 2. Evaluations on Test set
        hr, ndcg = engine.federatedEvaluate(test_data)

        logging.info(
            '[Epoch {}/{}][Test] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['num_round'],
                                                                          config['top_k'], hr, config['top_k'], ndcg))

        test_hrs.append(hr)
        test_ndcgs.append(ndcg)

        # Choose the model has the best performances
        if hr >= best_test_hr:
            best_test_hr = hr
            final_test_round = iteration

        # 3. Evaluations on Validation set
        val_hr, val_ndcg = engine.federatedEvaluate(validate_data)

        logging.info(
            '[Epoch {}/{}][Validation] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['num_round'],
                                                                                config['top_k'], val_hr,
                                                                                config['top_k'],
                                                                                val_ndcg))

        val_hrs.append(val_hr)
        val_ndcgs.append(val_ndcg)

    logging.info('--------------- The model training is finished ---------------')

    logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                             config['data_file'],
                                                             config['alias'],
                                                             sum(times)))

    # use a dict format to save results
    content = config.copy()

    # delete some unuseful key-value
    del content['device_id']
    del content['use_cuda']
    del content['model_dir']

    logging.info(str(content))

    # add some useful key-value
    content['finish_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content['hr'] = val_hrs[final_test_round]
    content['ndcg'] = val_ndcgs[final_test_round]

    # save useful data
    save_path = 'results/{}/{}/{}'.format(content['alias'], content['dataset'], content['type'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result_file = save_path + '/{}.{}.txt'.format(config['dataset'], config['data_file'].split('.')[0])

    with open(result_file, 'a') as file:
        file.write(str(content) + '\n')

    data_file = '{}/[{}]-[{}-{:.2e}-{:.2e}]-[HR{:.4f}-NDCG{:.4f}]-[{}].npz'.format(save_path,
                                                                                   content['data_file'].split('.')[0],
                                                                                   config['regular'], content['lambda'],
                                                                                   content['mu'], content['hr'],
                                                                                   content['ndcg'],
                                                                                   content['comment'])

    np.savez(data_file, test_hrs=test_hrs, test_ndcgs=test_ndcgs, val_hrs=val_hrs, val_ndcgs=val_ndcgs,
             train_losses=train_losses, sparsity=sparsity)

    logging.info('hit_list: {}'.format(test_hrs))
    logging.info('ndcg_list: {}'.format(test_ndcgs))

    notice = 'Best test hr: {:.4f}, ndcg: {:.4f} at round {}'.format(test_hrs[final_test_round],
                                                                     test_ndcgs[final_test_round], final_test_round)

    logging.info(notice)

    # Use WeChat to notice
    if config['on_server']:
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                             json={
                                 "token": str("e91a9b7f2c1c"),
                                 "title": str("NOTICE FROM EXPeriment"),
                                 "name": str("[{}] {}-{}: {}-{}".format(config['alias'], config['dataset'],
                                                                        config['data_file'].split('.')[0],
                                                                        config['type'], config['comment'])),
                                 "content": str(notice)
                             })
        print(resp.content.decode())
