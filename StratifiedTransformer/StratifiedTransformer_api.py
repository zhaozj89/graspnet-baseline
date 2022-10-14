# from .models import Point_BERT
# from .utils import misc
# from .utils.parser import *
from .model import stratified_transformer
from .util import config

def get_parser():
    config_path = '/home/amax/zzhaoao/Grasp/graspnet-baseline/StratifiedTransformer/config/s3dis/s3dis_stratified_transformer.yaml'
    cfg = config.load_cfg_from_cfg_file(config_path)
    cfg = config.merge_cfg_from_list(cfg, [])
    return cfg

def GetStratifiedTransformer():
    args = get_parser()

    args.patch_size = args.grid_size * args.patch_size
    args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
    args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
    args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

    model = stratified_transformer.Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
        args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
        rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=256, \
        ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    return model

if __name__=='__main__':
    print(GetStratifiedTransformer())
