""" PointNet2 backbone for feature learning.
    Author: Charles R. Qi
"""
import os
import sys
from turtle import forward
from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
import torch.nn as nn
import torch_points_kernels as tp

# sys.path.append('..')
# from PointBERT.PointBERT_api import GetPointBERT
from StratifiedTransformer.StratifiedTransformer_api import GetStratifiedTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from PointBERT.datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class StratifiedTransformerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GetStratifiedTransformer()

    def forward(self, points: torch.cuda.FloatTensor, end_points=None):
        if end_points is None:
            raise ValueError("end_points is needed")

        point_clouds = end_points['point_clouds']
        cloud_colors = end_points['cloud_colors']

        npoints = 1024
        if npoints == 1024:
            point_all = 1200
        else:
            raise NotImplementedError()

        end_points['input_xyz'] = point_clouds
        fps_idx = pointnet2_utils.furthest_point_sample(point_clouds, point_all)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        point_clouds = pointnet2_utils.gather_operation(point_clouds.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
        cloud_colors = pointnet2_utils.gather_operation(cloud_colors.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

        offset, count = [], 0
        max_batch_points = 140000
        k = 0
        for item in point_clouds:
            # print("item shape:",item.shape)
            count += item.shape[0]
            if count > max_batch_points:
                break
            k += 1
            offset.append(count)

        coord = torch.cat([ele for ele in point_clouds[:k]])
        feat = torch.cat([ele for ele in cloud_colors[:k]])
        offset = torch.IntTensor(offset[:k])

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()

        grid_size = 0.04
        max_num_neighbors = 34
        sigma = 1.0
        radius = 2.5 * grid_size * sigma
        neighbor_idx = tp.ball_query(radius, max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

        coord, feat, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        feat = torch.cat([feat, coord], 1)

        output = self.model(feat, coord, offset, batch, neighbor_idx)
        
        offset_new = torch.cat([torch.tensor([0]).cuda(), offset]).tolist()
        features = torch.stack([output[offset_new[i]:j] for i,j in enumerate(offset_new[1:])])
        features = features.permute(0, 2, 1)
        end_points['fp2_xyz'] = point_clouds
        end_points['fp2_inds'] = fps_idx
        return features, point_clouds, end_points
    
class TransformerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model, self.optimizer, self.scheduler = GetPointBERT()

    def forward(self, points: torch.cuda.FloatTensor, end_points=None):
        npoints = 1024
        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        end_points['input_xyz'] = points
        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
        # # import pdb; pdb.set_trace()
        points = train_transforms(points)
        features, positions = self.base_model(points)
        end_points['fp2_xyz'] = positions
        end_points['fp2_inds'] = fps_idx
        return features, positions, end_points

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.04,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        return features, end_points['fp2_xyz'], end_points