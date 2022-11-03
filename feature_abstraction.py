import numpy as np
from common.config import argparser
from common.engine import Get_BaseColor,Label_gridFloorPlan,get_grid_width
import os,pickle

class f_abs:
    def __init__(self, grid_width,baseColor_fn, fp_zone_fn, fp_code_fn) -> None:
        self.features = None
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
            
