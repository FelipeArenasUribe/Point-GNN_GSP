'''
Point downsampling algorithms.
'''
import numpy as np

from dataset.kitti_dataset import Points

def downsample_by_average_voxel(points, voxel_size):
    """Voxel downsampling using average function.

    points: a dataset.Points namedtuple containing "xyz" and "attr".
    voxel_size: the size of voxel cells used for downsampling (i.e. 0.8)
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