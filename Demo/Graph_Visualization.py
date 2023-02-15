import numpy as np
import open3d

import sys
from os.path import join
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from dataset.kitti_dataset import KittiDataset, Points
from models import graph_gen

# Create dataset object for easier data manipulation
dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

level_configs: a dict of 'level', 'graph_gen_method','graph_gen_kwargs', 'graph_scale'

level_configs = {
    'level': 3,
    'graph_gen_method': '',
    'graph_gen_kwargs': kwargs,
    'graph_scale', voxel_size
}

voxel_size = 0.8

def fetch_data(frame_idx):
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx, voxel_size)
    #box_label_list = dataset.get_label(frame_idx)

    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_gen.gen_multi_level_local_graph_v3(cam_rgb_points.xyz, voxel_size, level_configs='3')
    
    input_v = cam_rgb_points.attr

    input_v = input_v.astype(np.float32)

    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)

def Visualize_Point_Cloud(geometry_list, image_file_path, show_image=False):
        print('Image file path: ',image_file_path)
        
        # Visualize image:
        if show_image:
            plt.title("Point Cloud Image")
            plt.xlabel("")
            plt.ylabel("")
            image = mpimg.imread(image_file_path)
            print('Close image to continue.')
            plt.imshow(image)
            plt.show()
            

        # Visualize point cloud
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        print(geometry_list)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close point cloud to continue.')
        vis.run()  
        vis.destroy_window()

def show_graph(src_points, des_points, edges):
    """
    :param src_points: [N, 3] src_points.
    :param des_points: [N, 3] des_points.
    :param edges: [M, 2],M pairs of connections src_points[edges[0]] -> des_points[edges[1]]
    :return: None
    """
    points = np.concatenate([src_points, des_points])
    edges[:, 1] += src_points.shape[0]
    line_set = open3d.LineSet()
    line_set.points = open3d.Vector3dVector(points)
    line_set.lines = open3d.Vector2iVector(edges)

    # add color if wanted
    # line_set.colors = np.zeros((edges.shape[0], 3), dtype=np.float32)
    
    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        vis.run()
        vis.destroy_window()

    custom_draw_geometry_load_option([line_set])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        results = fetch_data(frame_idx)
        print(results)