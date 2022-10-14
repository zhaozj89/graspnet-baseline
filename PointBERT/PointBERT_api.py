from .models import Point_BERT
from .utils import misc
from .utils.parser import *
from .utils.logger import *
from .utils.config import *
from .tools import builder
import torch
import torch.nn as nn
import time
import os
from tensorboardX import SummaryWriter


def GetPointBERT():
    args = get_args()
    args.use_gpu = torch.cuda.is_available()

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    args.distributed = False

    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, name=args.log_name)

    # if not args.test:
    #     train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    #     val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    config = get_config(args)#, logger = logger)

    config.dataset.train.others.bs = config.total_bs
    if config.dataset.get('extra_train'):
        config.dataset.extra_train.others.bs = config.total_bs * 2
    config.dataset.val.others.bs = config.total_bs * 2
    if config.dataset.get('test'):
        config.dataset.test.others.bs = config.total_bs

    # log_args_to_file(args, 'args', logger = logger)
    # log_config_to_file(config, 'config', logger = logger)


    if args.seed is not None:
        # logger.info(f'Set random seed to {args.seed}, '
        #             f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation

    base_model = builder.model_builder(config.model)

    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)

    # print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    return base_model, optimizer, scheduler

if __name__=='__main__':
    GetPointBERT()
