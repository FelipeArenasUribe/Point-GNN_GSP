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