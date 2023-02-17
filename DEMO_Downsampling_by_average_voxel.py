import numpy as np
import open3d

from os.path import join

from dataset.kitti_dataset import KittiDataset
from Demo_Utils.PointCloud_Downsampling import downsample_by_average_voxel
from Demo_Utils.PointCloud_Visualization import NormalizeData, Visualize_Point_Cloud


dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)


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

        print('------------------Original Point Cloud visualization------------------')
        print()
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(xyz)
        pcd.colors = open3d.Vector3dVector(colors)
        Visualize_Point_Cloud([pcd], image_file)
        print()

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

        print('------------------Downsampled Point Cloud visualization------------------')
        print()
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(downsampled_xyz.xyz)
        pcd.colors = open3d.Vector3dVector(colors)
        Visualize_Point_Cloud([pcd], image_file)
        print()

        holder = False
        while holder==False:
            if input('Enter q to continue: ') == 'q':
                holder=True