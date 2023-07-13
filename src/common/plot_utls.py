
import matplotlib.pyplot as plt
import numpy as np
from common.A_star import Env, AStar
from common.config import argparser
import matplotlib.patches as patches
from PIL import Image

class result_plot(AStar):
    def __init__(self, fp_grid, s_start, s_goal, heuristic_type, Env, color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,0],4:[255,0,0],5:[255,255,255], 6:[128,128,128], 7:[0,0,204]}):
        super(result_plot, self).__init__(s_start, s_goal, heuristic_type, Env)
        self.fp_grid = fp_grid
        self.fp_shape = fp_grid.shape
        # self.obstacle = obstacle
        # self.start_end_pair_list = start_end_pair_list
        self.color_map = color_map
        self.img_show = np.zeros((self.fp_shape[0], self.fp_shape[1], 3))

    def reload_fp_grid(self, fp_grid):
        self.Env = fp_grid  # class Env
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.doorway = self.Env.doorway # position of doorways

    def plot_fp(self):
        for i in range(self.fp_shape[0]):
            for j in range(self.fp_shape[1]):
                self.img_show[i,j] = self.color_map[self.fp_grid[i,j]]
    
    def add_path(self, doorway_penalty,wall_penalty, path_color):
        path,_ = self.searching(doorway_penalty, wall_penalty)
        for i in path:
            self.img_show[i] = path_color
        # print(path)
    
    def change_s_start_goal(self, new_s_start, new_s_goal):
        self.s_start = new_s_start
        self.s_goal = new_s_goal

    def add_s_start_goal_plot(self):
        point_list = [self.s_start, self.s_goal]
        for i in range(2):
            neighbour = self.get_neighbor(point_list[i])
            point_list.extend(neighbour)
        for p in point_list:
            self.img_show[p] = [255,0,0]

    def add_obs(self, obs, obs_color):
        for i in obs:
            self.img_show[i] = obs_color
            
    def get_neighbor(self, s):
        return super().get_neighbor(s)

    def re_initialize(self):
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def plot_show(self, file_name=None):
        plt.imshow(self.img_show.astype(np.int))
        plt.axis('off')
        if file_name != None:
            plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    
def plot_sensor_placement(fp_grid, grid_width, fp_img_name, sensor_placement):
    
    # read floor plan image
    img = Image.open(fp_img_name)

    # create figure and axes
    fig, ax = plt.subplots()

    ax.imshow(img)

    img = np.array(img)
    # compute the scale
    x_scale = img.shape[0]/fp_grid.shape[0]
    y_scale = img.shape[1]/fp_grid.shape[1]
    
    # compute rectangle size (cm)
    pixel_width = grid_width/((x_scale + y_scale)/2)
    rectangle_size = int(210 / pixel_width)

    # compute the sensor pixel position
    # point_list = []
    for i in sensor_placement:
        x = int(i[0] * x_scale - 0.5*rectangle_size)
        y = int(i[1] * y_scale - 0.5*rectangle_size)
        # point_list.append([x,y])
    
        rect = patches.Rectangle((y,x), rectangle_size,rectangle_size,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    fig.show()
    plt.show()
   

