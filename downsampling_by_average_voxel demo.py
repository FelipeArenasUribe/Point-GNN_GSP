import numpy as np
import open3d

import sys
from os.path import join
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from dataset.kitti_dataset import KittiDataset, Points

dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

def downsample_by_average_voxel(points, voxel_size):
    """Voxel downsampling using average function.

    points: a Points namedtuple containing "xyz" and "attr".
    voxel_size: the size of voxel cells used for downsampling.
    """
    # create voxel grid
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    xyz_idx = (points.xyz - xyz_offset) // voxel_size
    xyz_idx = xyz_idx.astype(np.int32)
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
    order = np.argsort(keys)
    keys = keys[order]
    points_xyz = points.xyz[order]
    unique_keys, lens = np.unique(keys, return_counts=True)
    indices = np.hstack([[0], lens[:-1]]).cumsum()
    downsampled_xyz = np.add.reduceat(
        points_xyz, indices, axis=0)/lens[:,np.newaxis]
    print(lens[:,np.newaxis])
    include_attr = points.attr is not None
    if include_attr:
        attr = points.attr[order]
        downsampled_attr = np.add.reduceat(
            attr, indices, axis=0)/lens[:,np.newaxis]
    if include_attr:
        return Points(xyz=downsampled_xyz,
                attr=downsampled_attr)
    else:
        return Points(xyz=downsampled_xyz,
                attr=None)

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

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":

    voxel_size=0.8 #Sample value extracted from a sample config file

    for frame_idx in range(0, dataset.num_files):
        ## Read the point cloud file:
        velo_points = dataset.get_velo_points(frame_idx)
        filename = dataset.get_filename(frame_idx)
        xyz = velo_points.xyz        

        ## Visualize initial point cloud:
        image_file=join(dataset._image_dir,dataset._file_list[frame_idx]+'.png')

        color_attr = velo_points.attr
        color_attr = NormalizeData(color_attr)
        colors = np.zeros((len(velo_points.xyz), 3))
        for i in range(0,len(color_attr)):
            if color_attr[i] < 0.33:
                colors[i][0] = color_attr[i]
            if color_attr[i] < 0.66 and color_attr[i] < 0.33:
                colors[i][1] = color_attr[i]
            else:
                colors[i][2] = color_attr[i]

        print('Original Point Cloud visualization:')
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(xyz)
        pcd.colors = open3d.Vector3dVector(colors)
        Visualize_Point_Cloud([pcd], image_file, show_image=True)

        #Downsample
        downsampled_xyz = downsample_by_average_voxel(velo_points, 0.8)

        color_attr = downsampled_xyz.attr
        color_attr = NormalizeData(color_attr)
        colors = np.zeros((len(downsampled_xyz.xyz), 3))
        for i in range(0,len(color_attr)):
            if color_attr[i] < 0.33:
                colors[i][0] = color_attr[i]
            if color_attr[i] < 0.66 and color_attr[i] < 0.33:
                colors[i][1] = color_attr[i]
            else:
                colors[i][2] = color_attr[i]

        print('Downsampled Point Cloud visualization:')
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(downsampled_xyz.xyz)
        pcd.colors = open3d.Vector3dVector(colors)
        Visualize_Point_Cloud([pcd], image_file)