import os
import torch
import numpy as np
import ray
import tqdm
import math

from .dataset import LBCDataset, data_loader
from .logger import Logger
from .lbc import LBC


def main(args):
    
    lbc = LBC(args)
    logger = Logger(args)
    dataset = LBCDataset(lbc.main_data_dir, args.config_path, jitter=True)
    
    data = data_loader(dataset, args.batch_size)
    
    global_it = 0
    for epoch in range(args.num_epochs):
        for rgbs, lbls, sems, dlocs, spds, cmds in tqdm.tqdm(data, desc=f'Epoch {epoch}'):
            info = lbc.train(rgbs, lbls, sems, dlocs, spds, cmds, train='bev')
            global_it += 1

            if global_it % args.num_iters_per_log == 0:
                logger.log_bev(global_it, lbls, info)
    
    
        # Save model
        torch.save(lbc.bev_model.state_dict(), os.path.join(logger.log_dir, 'bev_model_{}.th'.format(epoch+1)))
        torch.save(lbc.bev_model.state_dict(), os.path.join(args.save_path, 'bev_model_{}.th'.format(epoch+1)))


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--project', default='carla_lbc_bev')
    parser.add_argument('--config-path', default='expirements/config/config_lbc.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--save_path', default="./expirements/models/bev_model/e30")

    # Training data config
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-epochs', type=int, default=30)

    # Logging config
    parser.add_argument('--num-iters-per-log', type=int, default=100)

    args = parser.parse_args()

    main(args)
