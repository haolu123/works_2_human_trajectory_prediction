import numpy as np
from common.config import argparser
from common.engine import Get_BaseColor, Label_gridFloorPlan
import os,pickle
import matplotlib.pyplot as plt

class f_abs:
    Feature_dims = 7

    def __init__(self, grid_width,baseColor_fn, fp_zone_fn, fp_code_fn) -> None:
        self.features = np.zeros(self.Feature_dims)
        self.fp = self.get_fp(baseColor_fn, fp_zone_fn, fp_code_fn, grid_width)
        

    def get_fp(self, baseColor_fn, fp_zone_fn, fp_code_fn, grid_width):
        args = argparser.parse_args()
        base_color_fig_name = args.data_dir + baseColor_fn
        floor_plan_zone_fig_name = args.data_dir + fp_zone_fn
        floor_plan_code_fig_name = args.data_dir + fp_code_fn

        room_width = args.room_width
        room_length = args.room_length

        flag = os.path.exists(args.temp_result_dir+'baseColor.pickle')
        if args.use_saved_base_color and flag:
            with open(args.temp_result_dir+'baseColor.pickle', 'rb') as f:
                baseColor = pickle.load(f)
        else:
            baseColor = Get_BaseColor(base_color_fig_name, args.temp_result_dir + 'baseColor.pickle', 16)

        flag = os.path.exists(args.temp_result_dir + 'fp_grid.pickle')
        if args.use_save_temp_result and flag:
            with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
                fp_grid = pickle.load(f)
            with open(args.temp_result_dir + 'fp_grid_groundtruth.pickle','rb') as f:
                fp_grid_gt = pickle.load(f)
        else:
            fp_grid, fp_grid_gt =  Label_gridFloorPlan(floor_plan_code_fig_name, floor_plan_zone_fig_name, room_width, room_length, grid_width, args.temp_result_dir, baseColor)
        
        
        return fp_grid
            
    def get_features(self, s):
        '''
            feature map
            Args:
                s : (x,y) position
            Returns:
                features : change self.features
        '''
        material = self.fp[s]
        b = np.zeros(6)
        b[material] = 1
        self.features[:6] = b
        self.features[6] = self.min_distance_to_wall(s)

    def min_distance_to_wall(self,point):
        """
            get the minimum distance to the wall or obstacles

        """
        detect_dir = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for i in range(10):
            for j in detect_dir:
                x = i*j[0] + point[0]
                y = i*j[1] + point[1]
                if self.features(x,y) == 1 or self.features(x,y) == 2:
                    return i
        return 10


def run():
    grid_width = 10
    baseColor_fn = 'baseColor.png'
    fp_zone_fn = 'fp1_zone.png'
    fp_code_fn = 'fp1_code.png'


