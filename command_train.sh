# CUDA_VISIBLE_DEVICES=3 python train.py --camera realsense --log_dir logs_debug/log_rs --batch_size 2 --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
# CUDA_VISIBLE_DEVICES=0 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet
CUDA_VISIBLE_DEVICES=2 python train.py --camera kinect --log_dir logs_stratified/log_kt --batch_size 2 --dataset_root /home/data/zzhaoao/Transformer/Grasp/Graspnet
