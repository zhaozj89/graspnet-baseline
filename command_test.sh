# CUDA_VISIBLE_DEVICES=2 python test.py --dump_dir logs_transformer/dump_rs --checkpoint_path logs_transformer/log_rs/checkpoint.tar --camera realsense --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
# CUDA_VISIBLE_DEVICES=1 python test.py --dump_dir logs_stratified/dump_rs --checkpoint_path logs_stratified/log_rs/checkpoint.tar --camera realsense --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
# CUDA_VISIBLE_DEVICES=2 python test.py --dump_dir logs_stratified/dump_kt --checkpoint_path logs_stratified/log_kt/checkpoint.tar --camera kinect --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=1 python test.py --dump_dir logs_stratified_pretrained/dump_rs --checkpoint_path logs_stratified_pretrained/log_rs/checkpoint.tar --camera realsense --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
# CUDA_VISIBLE_DEVICES=2 python test.py --dump_dir logs_stratified/dump_kt --checkpoint_path logs_stratified/log_kt/checkpoint.tar --camera kinect --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet

# CUDA_VISIBLE_DEVICES=1 python test.py --dump_dir logs_stratified_new/dump_rs --checkpoint_path logs_stratified_new/log_rs/checkpoint_17.tar --camera realsense --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
CUDA_VISIBLE_DEVICES=1 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint_2.tar --camera realsense --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet