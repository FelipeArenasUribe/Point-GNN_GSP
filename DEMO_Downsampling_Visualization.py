import numpy as np
import open3d
import os
from os.path import join
from dataset.kitti_dataset import KittiDataset
from sklearn.cluster import KMeans
from tqdm import tqdm

from Demo_Utils.PointCloud_Visualization import NormalizeData, Visualize_Point_Cloud

## Create dataset object to manipulate dataset with ease. While creating the object, some parameters are given.
# Dataset file path: /media/felipearur/ZackUP/dataset/kitti
dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

## Downsampling parameter: defines the denominator by which the point will be reduced (N' = N/downsample_rate)
downsample_rate = 2
output_dir = '/media/felipearur/ZackUP/dataset/kitti/velodyne/training_downsampled_%d/velodyne/' % downsample_rate

## For each point cloud...
for frame_idx in tqdm(range(0, dataset.num_files)):

    ## Read the point cloud file:
    velo_points = dataset.get_velo_points(frame_idx) # Read the points and store them in a Point object with two attributes: xyz and attr
    filename = dataset.get_filename(frame_idx) # Gets the filename of the point cloud (i.e. '000001.bin')
    xyz = velo_points.xyz # Store point's xyz values in a different variable

    ## Calculate the cosine angle for every point:
    xyz_norm = np.sqrt(np.sum(xyz * xyz, axis=1, keepdims=True)) # Calculate the point's euclidean norm for every point in the point cloud
    z_axis = np.array([[0], [0], [1]]) # Create z_axis vector representation.
    cos = xyz.dot(z_axis) / xyz_norm # Normalize z values.
    
    ## Visualize initial point cloud:
    image_file=join(dataset._image_dir,dataset._file_list[frame_idx]+'.png')

    # Uncomment if you want to color points according to cosine value
    color_cos = NormalizeData(cos)
    colors = np.zeros((len(cos), 3))
    for i in range(0,len(color_cos)):
        if color_cos[i] < 0.33:
            colors[i][0] = color_cos[i]
        if color_cos[i] < 0.66 and color_cos[i] < 0.33:
            colors[i][1] = color_cos[i]
        else:
            colors[i][2] = color_cos[i]
    
    # Uncomment if you want to color points according to attributes
    #color_attr = velo_points.attr
    #color_attr = NormalizeData(color_attr)
    #colors = np.zeros((len(cos), 3))
    #for i in range(0,len(color_attr)):
    #    if color_attr[i] < 0.33:
    #        colors[i][0] = color_attr[i]
    #    if color_attr[i] < 0.66 and color_attr[i] < 0.33:
    #        colors[i][1] = color_attr[i]
    #    else:
    #        colors[i][2] = color_attr[i]

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(xyz)
    pcd.colors = open3d.Vector3dVector(colors)
    Visualize_Point_Cloud([pcd], image_file)

    ## Fit a KMeans model on the cosines:
    # kmeans = KMeans(n_clusters=64, n_jobs=-1).fit(cos) # Doesnt work as it involves a parameter used in older versions of sklearn.
    kmeans = KMeans(n_clusters=64).fit(cos) # Trains a K-means clustering model to fit the cos data in 64 clusters.
    centers = np.sort(np.squeeze(kmeans.cluster_centers_)) # Sorts centroids from the K-means model in ascending order.
    centers = [-1, ] + centers.tolist() + [1, ] # Parse Centers from np.array to list and inclues a -1 and a 1 as the first and las elements of the list.
    cos = np.squeeze(cos)


    ## Compares every point to the average of two consecutive clusters
    point_total_mask = np.zeros(len(velo_points.xyz), dtype=bool) # Create boolean array of size N (N = total number of points in point cloud).
    for i in range(0, len(centers) - 2, downsample_rate): # For every 2 centroids
        lower = (centers[i] + centers[i + 1]) / 2
        #print('Lower: ',lower)
        higher = (centers[i + 1] + centers[i + 2]) / 2
        #print('higher: ',higher)
        point_mask = (cos > lower) * (cos < higher)
        #print('point_mask: ',len(point_mask))
        point_total_mask += point_mask
        #print('point_total_mask: ',len(point_total_mask))
     
        # Visualization
        #pcd = open3d.PointCloud()
        #pcd.points = open3d.Vector3dVector(velo_points.xyz[point_total_mask, :])
        #Visualize_Point_Cloud([pcd], image_file)

    # Visualize downsampled point cloud:
    clusters = np.linspace(0,1,len(cos))
    clusters = clusters[point_total_mask]
    color_clusters = np.zeros((len(clusters), 3))

    for i in range(0,len(clusters)):
            color_clusters[i][0] = clusters[i]

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(velo_points.xyz[point_total_mask, :])
    pcd.colors = open3d.Vector3dVector(color_clusters)
    Visualize_Point_Cloud([pcd], image_file)

    holder = False
    while holder==False:
        if input('Enter q to continue: ') == 'q':
            holder=True

    # Save output point_cloud
    #output = np.hstack([velo_points.xyz[point_total_mask, :], velo_points.attr[point_total_mask, :]])
    #point_file = output_dir + filename + '.bin'
    #os.makedirs(os.path.dirname(point_file), exist_ok=True)
    #output.tofile(point_file)
