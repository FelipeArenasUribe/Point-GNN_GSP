import open3d
import numpy as np

from os.path import join

from dataset.kitti_dataset import KittiDataset
from Demo_Utils import PointCloud_Visualization
from Demo_Utils.Data_Fetcher import fetch_data_and_labels

import pdb

voxel_size = 0.8

color_dict = {  0: [0, 0, 0],
                1: [1, 0, 0],
                2: [0, 1, 0],
                3: [0.7, 0.7, 0]}

config = {
    "box_encoding_method": "classaware_all_class_box_encoding",
    "downsample_by_voxel_size": True,
    "eval_is_training": True,
    "graph_gen_kwargs": {
        "add_rnd3d": True,
        "base_voxel_size": 0.8,
        "downsample_method": "random",
        "level_configs": [
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 1.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 0,
                "graph_scale": 1
            },
            {
                "graph_gen_kwargs": {
                    "num_neighbors": 256,
                    "radius": 4.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 1,
                "graph_scale": 1
            }
        ]
    },
    "graph_gen_method": "multi_level_local_graph_v3",
    "input_features": "i",
    "label_method": "Car",
    "loss": {
        "cls_loss_type": "softmax",
        "cls_loss_weight": 0.1,
        "loc_loss_weight": 10.0
    },
    "model_kwargs": {
        "layer_configs": [
            {
                "graph_level": 0,
                "kwargs": {
                    "output_MLP_activation_type": "ReLU",
                    "output_MLP_depth_list": [
                        300,
                        300
                    ],
                    "output_MLP_normalization_type": "NONE",
                    "point_MLP_activation_type": "ReLU",
                    "point_MLP_depth_list": [
                        32,
                        64,
                        128,
                        300
                    ],
                    "point_MLP_normalization_type": "NONE"
                },
                "scope": "layer1",
                "type": "scatter_max_point_set_pooling"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer2",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer3",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer4",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "activation_type": "ReLU",
                    "normalization_type": "NONE"
                },
                "scope": "output",
                "type": "classaware_predictor"
            }
        ],
        "regularizer_kwargs": {
            "scale": 5e-07
        },
        "regularizer_type": "l1"
    },
    "model_name": "multi_layer_fast_local_graph_model_v2",
    "nms_overlapped_thres": 0.01,
    "num_classes": 4,
    "runtime_graph_gen_kwargs": {
        "add_rnd3d": False,
        "base_voxel_size": 0.8,
        "level_configs": [
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 1.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 0,
                "graph_scale": 0.5
            },
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 4.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 1,
                "graph_scale": 0.5
            }
        ]
    }
}

train_config = {
    "NUM_GPU": 2,
    "NUM_TEST_SAMPLE": -1,
    "batch_size": 4,
    "capacity": 1,
    "checkpoint_path": "model",
    "config_path": "config",
    "data_aug_configs": [
        {
            "method_kwargs": {
                "expend_factor": [
                    1.0,
                    1.0,
                    1.0
                ],
                "method_name": "normal",
                "yaw_std": 0.39269908169872414
            },
            "method_name": "random_rotation_all"
        },
        {
            "method_kwargs": {
                "flip_prob": 0.5
            },
            "method_name": "random_flip_all"
        },
        {
            "method_kwargs": {
                "appr_factor": 10,
                "expend_factor": [
                    1.1,
                    1.1,
                    1.1
                ],
                "max_overlap_num_allowed": 100,
                "max_overlap_rate": 0.01,
                "max_trails": 100,
                "method_name": "normal",
                "xyz_std": [
                    3,
                    0,
                    3
                ]
            },
            "method_name": "random_box_shift"
        }
    ],
    "decay_factor": 0.1,
    "decay_step": 400000,
    "gpu_memusage": -1,
    "initial_lr": 0.125,
    "load_dataset_every_N_time": 0,
    "load_dataset_to_mem": True,
    "max_epoch": 1718,
    "max_steps": 1400000,
    "num_load_dataset_workers": 16,
    "optimizer": "sgd",
    "optimizer_kwargs": {},
    "save_every_epoch": 20,
    "train_dataset": "train_car.txt",
    "train_dir": "./checkpoints/car_auto_T3_train",
    "unify_copies": True,
    "visualization": False
}

# Create dataset object for easier data manipulation
dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '/media/felipearur/ZackUP/dataset/kitti/labels/training/label_2',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    num_classes=config['num_classes'])

def custom_draw_geometry_load_option(geometry_list):
                vis = open3d.Visualizer()
                vis.create_window()
                for geometry in geometry_list:
                    vis.add_geometry(geometry)
                ctr = vis.get_view_control()
                ctr.rotate(0.0, 3141.0, 0)
                vis.run()
                vis.destroy_window()

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        cam_rgb_points, downsampled_PointCloud, last_layer_points_xyz, last_layer_points_xyz_full, cls_labels_full, cls_labels = fetch_data_and_labels(dataset, frame_idx, voxel_size, config)

        colors_GT = []
        colors_D = []

        for i in range(0,len(cls_labels_full)):
            colors_GT.append(color_dict[int(cls_labels_full[i])])

        for i in range(0,len(cls_labels)):
            colors_D.append(color_dict[int(cls_labels[i])])

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(last_layer_points_xyz_full)
        pcd.colors = open3d.Vector3dVector(colors_GT)

        print(pcd)
        custom_draw_geometry_load_option([pcd])

        pcd_D = open3d.PointCloud()
        pcd_D.points = open3d.Vector3dVector(last_layer_points_xyz)
        pcd_D.colors = open3d.Vector3dVector(colors_D)

        print(pcd_D)
        custom_draw_geometry_load_option([pcd_D])

        input('Hold on...')