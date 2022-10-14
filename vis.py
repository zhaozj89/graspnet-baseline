import os, sys, time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from torch.utils.data import DataLoader
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
import numpy as np
import open3d as o3d
import cv2

dataset_root = '/home/data/zzhaoao/Transformer/Grasp/Graspnet'
camera = 'realsense'
num_point = 20000
batch_size = 2

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

valid_obj_idxs, grasp_labels = load_grasp_labels(dataset_root)

# TRAIN_DATASET = GraspNetDataset(dataset_root, valid_obj_idxs, grasp_labels, camera=camera, split='train', num_points=num_point, remove_outlier=True, augment=True)
TEST_DATASET = GraspNetDataset(dataset_root, valid_obj_idxs, grasp_labels, camera=camera, split='test_seen', num_points=num_point, remove_outlier=True, augment=False)

# print(len(TRAIN_DATASET), len(TEST_DATASET))

# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
# TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

# print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))


end_points = TEST_DATASET[30]

print(end_points.keys())
point_clouds = end_points['point_clouds']
cloud_colors = end_points['cloud_colors']


# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_clouds)
pcd.colors = o3d.utility.Vector3dVector(cloud_colors)

o3d.visualization.draw_geometries([pcd])

# # Visualize Point Cloud
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)

# # Read camera params
# # param = o3d.io.read_pinhole_camera_parameters('cameraparams.json')
# # ctr = vis.get_view_control()
# # ctr.convert_from_pinhole_camera_parameters(param)

# # Updates
# vis.update_geometry(pcd)
# vis.poll_events()
# vis.update_renderer()

# # Capture image
# time.sleep(10)
# # vis.capture_screen_image('cameraparams.png')
# image = vis.capture_screen_float_buffer()
# # scale and convert to uint8 type
# o3d_screenshot_mat = (255.0 * np.asarray(image)).astype(np.uint8)

# cv2.imwrite('test.png', o3d_screenshot_mat)

# # use as required
# # cv2.imshow("screenshot", o3d_screenshot_mat) 

# # Close
# vis.destroy_window()