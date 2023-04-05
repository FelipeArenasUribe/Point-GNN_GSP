import numpy as np
import open3d
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random as rnd

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
        opt.show_coordinate_frame = False
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close point cloud to continue.')
        vis.run()  
        vis.destroy_window()

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def Visualize_Graph(nodes, edges):
    colors = [[0.7, 0.8, 0.7] for i in range(len(edges))]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(nodes)
    line_set.lines = open3d.utility.Vector2iVector(edges)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(nodes)
    pcd.paint_uniform_color([0.5, 1, 0.8])
    print(pcd)

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        #opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([line_set, pcd])

def Visualize_Partitioned_Graph(nodes, edges, pcds):
    colors = [[0.7, 0.8, 0.7] for i in range(len(edges))]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(nodes)
    line_set.lines = open3d.utility.Vector2iVector(edges)
    line_set.colors = open3d.utility.Vector3dVector(colors)

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

    custom_draw_geometry_load_option([line_set, *pcds])

def Visualize_Graphs(nodes, edges):
    pcd = []
    for i in range(0, len(nodes)):
        pcd.append(open3d.PointCloud())
        pcd[i].points = open3d.Vector3dVector(nodes[i])
        pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

    lines = []
    for i in range(0, len(edges)):
        lines.append(open3d.geometry.LineSet())
        lines[i].points = open3d.utility.Vector3dVector(nodes[i])
        lines[i].lines = open3d.utility.Vector2iVector(edges[i])
        lines[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        #opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([*lines, *pcd])

def Visualize_VaC_Point_Clouds(Points):
    pcd = []
    #for i in range(0, len(Points)):
    #    pcd.append(open3d.PointCloud())
    #    pcd[i].points = open3d.Vector3dVector(Points[i])
    #    pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])
    
    # Visualize original Point Cloud
    pcd.append(open3d.PointCloud())
    pcd[0].points = open3d.Vector3dVector(Points[0])
    pcd[0].paint_uniform_color([0, 0, 1])

    # Visualize downsampled Point Cloud
    pcd.append(open3d.PointCloud())
    pcd[1].points = open3d.Vector3dVector(Points[1])
    pcd[1].paint_uniform_color([1, 0, 0])

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        #opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([*pcd])

def Visualize_Point_Clouds(Point_clouds):
    pcd = []
    for i in range(0, len(Point_clouds)):
        pcd.append(open3d.PointCloud())
        pcd[i].points = open3d.Vector3dVector(Point_clouds[i])
        pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

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

    custom_draw_geometry_load_option([*pcd])