import os
import argparse
from collections import defaultdict
from pathlib import Path

def get_args():
    argsd = defaultdict()
    argsd['config'] = '/home/amax/zzhaoao/Grasp/Point-BERT/cfgs/ModelNet_models/PointTransformer.yaml'
    argsd['launcher'] = 'none'
    argsd['local_rank'] = 0
    argsd['num_workers'] = 4
    argsd['seed'] = 0
    argsd['deterministic'] = True
    argsd['sync_bn'] = False
    argsd['exp_name'] = 'test'
    argsd['start_ckpts'] = None
    argsd['ckpts'] = '/home/amax/zzhaoao/Grasp/Point-BERT/Point-BERT.pth'
    argsd['val_freq'] = 1
    argsd['resume'] = False
    argsd['test'] = False
    argsd['finetune_model'] = True
    argsd['scratch_model'] = False
    argsd['label_smoothing'] = False
    argsd['mode'] = None
    argsd['way'] = -1
    argsd['shot'] = -1
    argsd['fold'] = -1

    def dict2obj(obj,dict):
        obj.__dict__.update(dict)
        return obj
    
    class A(object):
        # def __init__(self):
        #         self.name = name

        def __setitem__(self, k, v):
                self.k = v

        def __str__(self):
                return "name:%s, %s" % (self.name, self.k)
                
    if argsd['test'] and argsd['resume']:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if argsd['resume'] and argsd['start_ckpts'] is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if argsd['test'] and argsd['ckpts'] is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if argsd['finetune_model'] and argsd['ckpts'] is None:
        raise ValueError(
            'ckpts shouldnt be None while finetune_model mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(argsd['local_rank'])

    if argsd['test']:
        argsd['exp_name'] = 'test_' + argsd['exp_name']
    if argsd['mode'] is not None:
        argsd['exp_name'] = argsd['exp_name'] + '_' +argsd['mode']
    argsd['experiment_path'] = os.path.join('./experiments', Path(argsd['config']).stem, Path(argsd['config']).parent.stem, argsd['exp_name'])
    argsd['tfboard_path'] = os.path.join('./experiments', Path(argsd['config']).stem, Path(argsd['config']).parent.stem,'TFBoard' ,argsd['exp_name'])
    argsd['log_name'] = Path(argsd['config']).stem
    
    args = A()
    dict2obj(args, argsd)
    
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

