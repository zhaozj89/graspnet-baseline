# from .models import Point_BERT
# from .utils import misc
# from .utils.parser import *
from .model import stratified_transformer
from .util import config
# import os
# import torch
# import collections

def get_parser():
    config_path = '/home/amax/zzhaoao/Grasp/graspnet-baseline/StratifiedTransformer/config/s3dis/s3dis_stratified_transformer.yaml'
    cfg = config.load_cfg_from_cfg_file(config_path)
    cfg = config.merge_cfg_from_list(cfg, [])
    return cfg

def GetStratifiedTransformer():
    args = get_parser()

    args.grid_size = 0.005
    args.patch_size = args.grid_size * args.patch_size
    args.window_size = [args.patch_size * args.window_size * (2**i) / 10 for i in range(args.num_layers)]
    args.grid_sizes = [args.patch_size * (2**i) / 10 for i in range(args.num_layers)]
    args.quant_sizes = [args.quant_size * (2**i) / 10 for i in range(args.num_layers)]

    model = stratified_transformer.Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
        args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
        rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=256, \
        ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    # model_path = '/home/amax/zzhaoao/Grasp/graspnet-baseline/StratifiedTransformer/pre-trained/s3dis_model_best.pth'
    # if os.path.isfile(model_path):
    #     checkpoint = torch.load(model_path)
    #     state_dict = checkpoint['state_dict']
    #     new_state_dict = collections.OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]
    #         new_state_dict[name.replace("item", "stem")] = v
    #     current_model = model.state_dict()
    #     new_state_dict={k:v if v.size()==current_model[k].size() else current_model[k] for k,v in zip(current_model.keys(),new_state_dict.values())}
    #     model.load_state_dict(new_state_dict, strict=True)
    #     # args.epoch = checkpoint['epoch']

    return model

if __name__=='__main__':
    print(GetStratifiedTransformer())
