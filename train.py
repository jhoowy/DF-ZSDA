import sys
import os
import glob
import time
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from datasets.datasets import *
from models import ZSDAModel
from config import NetConfig
from utils.logger import Logger
from optparse import OptionParser
from tqdm import tqdm

parser = OptionParser()
parser.add_option('--config', type=str, help='path to the experiment configuration file', default='configs/base_config.yaml')
parser.add_option('--name', type=str)
parser.add_option('--batch_size', type=int)
parser.add_option('--max_class', type=int)
parser.add_option('--match_sampling', default=False, action='store_true')

domain_dir = {
    'G' : 'original',
    'C' : 'color',
    'E' : 'edge',
    'N' : 'original',
}

def create_dataset(cfg):
    s, t = cfg.s_domain, cfg.t_domain
    for domain in [s, t]:
        if domain not in domain_dir:
            domain_dir[domain] = domain
            
    root = cfg.data_root
    irt_name = cfg.irt_data
    rt_name = cfg.rt_data
    resize_dim = 28
    rt_class_bias = 0
    irt_class_bias = 0

    # EMNIST class label starts from 1
    if rt_name == 'EMNIST':
        rt_class_bias = 1
        cfg.rt_classes -= 1
    if irt_name == 'EMNIST':
        irt_class_bias = 1

    s_transform = get_transform(s, resize_dim=resize_dim)
    t_transform = get_transform(t, resize_dim=resize_dim)
    irt_s_path = os.path.join(root, irt_name, domain_dir[s])
    irt_t_path = os.path.join(root, irt_name, domain_dir[t])
    rt_s_path = os.path.join(root, rt_name, domain_dir[s])

    irs_dataset = SingleDataset(irt_s_path, irt_s_path + "_labels.txt", s_transform, cfg.max_class, class_bias=irt_class_bias)
    irt_dataset = SingleDataset(irt_t_path, irt_t_path + "_labels.txt", t_transform, cfg.max_class, class_bias=irt_class_bias)
    rs_dataset = SingleDataset(rt_s_path, rt_s_path + "_labels.txt", s_transform, class_bias=rt_class_bias)

    if cfg.match_sampling:
        irs_pair_dataset = PairDataset(irs_dataset, irt_dataset)
        irt_pair_dataset = PairDataset(irt_dataset, irs_dataset)

    # Test set
    rt_test_path = os.path.join(root, rt_name + "_test", domain_dir[t])
    rs_test_path = os.path.join(root, rt_name + "_test", domain_dir[s])
    irt_test_path = os.path.join(root, irt_name + "_test", domain_dir[t])
    irs_test_path = os.path.join(root, irt_name + "_test", domain_dir[s])

    rt_test_dataset = SingleDataset(rt_test_path, rt_test_path + "_labels.txt", t_transform, class_bias=rt_class_bias)
    rs_test_dataset = SingleDataset(rs_test_path, rs_test_path + "_labels.txt", s_transform, class_bias=rt_class_bias)
    irt_test_dataset = SingleDataset(irt_test_path, irt_test_path + "_labels.txt", t_transform, cfg.max_class, class_bias=irt_class_bias)
    irs_test_dataset = SingleDataset(irs_test_path, irs_test_path + "_labels.txt", s_transform, cfg.max_class, class_bias=irt_class_bias)

    if cfg.match_sampling:
        return irs_pair_dataset, irt_pair_dataset, irs_dataset, irt_dataset, rs_dataset, \
               rt_test_dataset, rs_test_dataset, irt_test_dataset, irs_test_dataset

    return irs_dataset, irt_dataset, rs_dataset, rt_test_dataset, rs_test_dataset, irt_test_dataset, irs_test_dataset


def get_loader(dataset, cfg, batch_size=None, shuffle=True, drop_last=False, sampler=None):
    if batch_size is None:
        batch_size = cfg.batch_size

    if sampler is None:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.workers,
            drop_last=drop_last,
            pin_memory=False
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=cfg.workers,
            pin_memory=False
        )
    return loader

def main(argv):
    # Config
    (opts, args) = parser.parse_args(argv)
    assert isinstance(opts, object)
    cfg = NetConfig(opts.config)
    if opts.name is not None:
        cfg.name = '_'.join([cfg.name, opts.name])
    if opts.batch_size is not None:
        cfg.batch_size = opts.batch_size
    cfg.max_class = opts.max_class
    if cfg.max_class != None:
        cfg.irt_classes = cfg.max_class
    match_sampling = cfg.match_sampling

    save_dir = os.path.join(cfg.checkpoints_dir, cfg.name)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(cfg)

    # Dataset
    if match_sampling:
        irs_pair_dataset, irt_pair_dataset, irs_dataset, irt_dataset, rs_dataset, \
        rt_test_dataset, rs_test_dataset, irt_test_dataset, irs_test_dataset = create_dataset(cfg)
    else:
        irs_dataset, irt_dataset, rs_dataset, rt_test_dataset, rs_test_dataset, irt_test_dataset, irs_test_dataset = create_dataset(cfg)

    loaders = {}
    single_loaders = {}
    domains = ['irs', 'irt', 'rs']
    if match_sampling:
        loaders['irs'] = get_loader(irs_pair_dataset, cfg, batch_size=cfg.batch_size // 2, drop_last=True)
        loaders['irt'] = get_loader(irt_pair_dataset, cfg, batch_size=cfg.batch_size // 2, drop_last=True)
    loaders['rs'] = get_loader(rs_dataset, cfg, drop_last=True)
    single_loaders['irs'] = get_loader(irs_dataset, cfg, drop_last=True)
    single_loaders['irt'] = get_loader(irt_dataset, cfg, drop_last=True)
    single_loaders['rs'] = loaders['rs']

    test_loaders = {}
    test_loaders['rt'] = get_loader(rt_test_dataset, cfg)
    test_loaders['rs'] = get_loader(rs_test_dataset, cfg)
    test_loaders['irt'] = get_loader(irt_test_dataset, cfg)
    test_loaders['irs'] = get_loader(irs_test_dataset, cfg)
    
    main_dom = 'rs'
    dataset_size = len(loaders['rs'])

    dataset_size *= cfg.batch_size
    print('\nDataset size : %d' % dataset_size)
    
    save_train_image = False
    save_test_image = False

    model = ZSDAModel(cfg)
    n_epochs = cfg.n_epochs

    # Training the Model
    print('start training...')
    best_rt_accuracy = 0
    best_accuracy = 0
    total_iters = 0
    for epoch in range(n_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        loader_iters = {}
        single_loader_iters = {}
        for d in domains:
            if d == main_dom:
                continue
            if match_sampling:
                loader_iters[d] = iter(loaders[d])
            else:
                single_loader_iters[d] = iter(single_loaders[d])

        iters = 0
        for sample in tqdm(loaders[main_dom]):
            samples = {}
            samples[main_dom] = sample
            if match_sampling:
                for d, loader_iter in loader_iters.items():
                    try:
                        samples[d] = next(loader_iter)
                    except StopIteration:
                        loader_iters[d] = iter(loaders[d])
                        samples[d] = next(loader_iters[d])
            else:
                for d, loader_iter in single_loader_iters.items():
                    try:
                        samples[d] = next(loader_iter)
                    except StopIteration:
                        single_loader_iters[d] = iter(single_loaders[d])
                        samples[d] = next(single_loader_iters[d])

            iter_start_time = time.time()   # timer for computation per iteration
            if total_iters % cfg.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += cfg.batch_size
            epoch_iter += cfg.batch_size

            if match_sampling:
                model.set_pair_input(samples['irs'], samples['irt'], samples['rs'])
            else:
                model.set_input(samples['irs'], samples['irt'], samples['rs'])
            
            # match_sampling = not match_sampling
            
            if save_train_image:
                save_image(make_grid(samples['irs'][0], nrow=8), os.path.join(save_dir, 'irs.png'))
                save_image(make_grid(samples['irt'][0], nrow=8), os.path.join(save_dir, 'irt.png'))
                save_image(make_grid(samples['rs'][0], nrow=8), os.path.join(save_dir, 'rs.png'))
                save_train_image = False

            # calculate loss, get gradients, update network weights
            model.update()

            losses = model.get_current_loss()

            if cfg.print_loss and total_iters % cfg.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / cfg.batch_size
                logger.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            iter_data_time = time.time()
            
        logger.log_message('\nEnd of epoch %d / %d \t Time Taken: %d sec' % (epoch, cfg.n_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()

        accuracy_sum = 0
        rt_accuracy = 0
        for task in test_loaders:
            if save_test_image:
                for x, y in test_loaders[task]:
                    save_image(make_grid(x, nrow=8), os.path.join(save_dir, task + '_test.png'))
                    break

            accuracy, di_accuracy = model.test(test_loaders[task], task)
            logger.print_evaluation_result(task, accuracy, di_accuracy)

            if task != 'rt':
                accuracy_sum += accuracy
            else:
                rt_accuracy = accuracy
        save_test_image = False

        if accuracy_sum >= best_accuracy:
            best_accuracy = accuracy_sum
            best_rt_accuracy = rt_accuracy
            print('saving the best model at epoch %d' % epoch)
            model.save_networks('best')

    logger.print_end_of_training(best_rt_accuracy)

    if not cfg.save_weight:
        for f in glob.glob(os.path.join(save_dir, '*.pth')):
            os.remove(f)

if __name__ == '__main__':
    main(sys.argv)
