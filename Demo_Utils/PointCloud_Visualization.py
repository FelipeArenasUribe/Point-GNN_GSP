import numpy as np
import open3d
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def Visualize_Point_Cloud(geometry_list, image_file_path='', show_image=False):
        
        if show_image:
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

def Visualize_Graph(nodes, edges):
    """
    :param src_points: [N, 3] src_points.
    :param des_points: [N, 3] des_points.
    :param edges: [M, 2],M pairs of connections src_points[edges[0]] -> des_points[edges[1]]
    :return: None
    """

    src_points = np.zeros([len(edges),3])
    for i in range(0,len(edges)):
        src_points[i][:] = nodes[edges[i][0]]

    des_points = np.zeros([len(edges),3])
    for i in range(0,len(edges)):
        des_points[i][:] = nodes[edges[i][0]]

    points = np.concatenate([src_points, des_points])
    edges[:, 1] += src_points.shape[0]
    line_set = open3d.LineSet()
    line_set.points = open3d.Vector3dVector(points)
    line_set.lines = open3d.Vector2iVector(edges)
    line_set.paint_uniform_color([0.7, 0.8, 0.7])

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(src_points)
    pcd.paint_uniform_color([0.5, 1, 0.8])
    
    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([line_set, pcd])