import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
from common.config import argparser
from common.A_star import Env, AStar
import tqdm

def Get_BaseColor(img_name, pickle_name, baseColor_num):
    """This function is used to get the base color for color board
        Args: 
            img_name: the file name of the base color image
            pickle_name: the file name of the output file, which save the base colors.
            baseColor_num : the number of base colors.
            
        Returns: 
            baseColor: the list of the base color (bgr)
    """
    baseColor_img = cv2.imread(img_name)
    bgr_get = []
    def getposBgr(event, x, y, flags, param):
        nonlocal bgr_get
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr_get = baseColor_img[y, x]
    
    baseColor = []
    for i in range(baseColor_num):
        cv2.imshow("baseColor",baseColor_img)
        cv2.setMouseCallback("baseColor",getposBgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        baseColor.append(bgr_get)
    
    with open(pickle_name,'wb') as f:
        pickle.dump(baseColor, f)
    return baseColor

def get_grid_width(width, length):
    return int(min(40, math.sqrt(width*length/10000)))


def Label_gridFloorPlan(img_file, img_zone_file, width, length, grid_width, temp_result_folder, baseColor):
    """
    This function is used to label the colored floor plan

    Parameters
    ----------
    img_file : strings
        file name of the floor plan image
        
    width : int or float
        the width of the floor plan (cm)
    length : int or float
        the length of the floor plan (cm)
    grid_width : int or float
        the edge size of a grid (cm)
    temp_result_folder: str
        the direction of the temp_reulst folder
    baseColor: list (bgr color)
        the list of base color

    Returns
    -------
    fp_grid : np.array
        the grided floor plan 
        for each element ( 0: free spaces, 1: walls, 2: obstacles, 3: doorways, 4: area of interets, 5: boundarys, 6; others)

    """
    floorPlan_img = cv2.imread(img_file)
    fp_zone_img = cv2.imread(img_zone_file)
    fp_grid_w_len = int(floorPlan_img.shape[0]/(width/grid_width))
    fp_grid_l_len = int(floorPlan_img.shape[1]/(length/grid_width))
    
    grid_area = fp_grid_w_len*fp_grid_l_len
    
    grid_w = int(floorPlan_img.shape[0]/fp_grid_w_len)
    grid_l = int(floorPlan_img.shape[1]/fp_grid_l_len)
    
    
    
    fp_grid = np.zeros((grid_w, grid_l))
    fp_grid_zone = np.zeros((grid_w, grid_l))
    
    for i in range(grid_w):
        for j in range(grid_l):
            grid_patch = floorPlan_img[i*fp_grid_w_len:(i+1)*fp_grid_w_len, j*fp_grid_l_len:(j+1)*fp_grid_l_len]
            num_p_max = 0
            for i_color in range(len(baseColor)):
                num_pixel = len(np.where((grid_patch == baseColor[i_color]).all(2))[0])
                if num_pixel > num_p_max:
                    fp_grid[i,j] = i_color + 1
                    num_p_max = num_pixel
                    grid_patch = np.tile(baseColor[i_color],(grid_width,grid_width,1))
    
    for i in range(grid_w):
        for j in range(grid_l):
            grid_patch = fp_zone_img[i*fp_grid_w_len:(i+1)*fp_grid_w_len, j*fp_grid_l_len:(j+1)*fp_grid_l_len]
            num_p_max = 0
            for i_color in range(len(baseColor)):
                num_pixel = len(np.where((grid_patch == baseColor[i_color]).all(2))[0])
                if num_pixel > num_p_max:
                    fp_grid_zone[i,j] = i_color + 1
                    num_p_max = num_pixel
                    grid_patch = np.tile(baseColor[i_color],(grid_width,grid_width,1))
                    
    with open(temp_result_folder + 'fp_grid_groundtruth.pickle','wb') as f:
        pickle.dump(fp_grid_zone, f)
    
    fp_grid_gt = copy.deepcopy(fp_grid_zone)
    fp_grid[fp_grid>=6] = 0
    with open(temp_result_folder + 'fp_grid.pickle','wb') as f:
        pickle.dump(fp_grid, f)
    return fp_grid, fp_grid_gt

def get_fp(baseColor_fn, fp_zone_fn, fp_code_fn, grid_width):
        args = argparser.parse_args()
        base_color_fig_name = args.data_dir + baseColor_fn
        floor_plan_zone_fig_name = args.data_dir + fp_zone_fn
        floor_plan_code_fig_name = args.data_dir + fp_code_fn

        room_width = args.room_width
        room_length = args.room_length

        baseColor_temp_dir = args.temp_result_dir+'baseColor.pickle'
        flag = os.path.exists(baseColor_temp_dir)
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

def A_star_simulation(args, fp_grid):
    """
        get training dataset (using A* algroithm to mimic human behavior)
        Args: 
            args: pre-defined parameters,
            fp_grid: grided floor plan (array)
            grid_width: grid width/cm (float)
            range_b: boundary range (may not use here)

        Returns:
            path_all : list of paths
            bound_slices, b_mid: may not used.
    """
    walk_freq_map = np.zeros(fp_grid.shape)
    # build enviroment for A* algorithm
    fp_env = Env(fp_grid)

    # begin simulation
    path_all = []
    for loop_num in tqdm.tqdm(range(args.path_num)):
        # random select two points as start and end point
        (s_start, s_goal) = fp_env.get_start_end_point()

        # add random small obstacles
        fp_env.add_random_obs(args.random_obstacle_dense)

        # get the shortest path between start point and end point
        astar = AStar(s_start, s_goal, "euclidean", fp_env)
        path, visited = astar.searching(args.doorway_penalty, args.wall_penalty)

        # save shortest paths
        path_all.append(path)

        # mark in the walk frequence map
        if args.save_hotmap:
            for p in path:
                walk_freq_map[p[0],p[1]] += 1

    if args.save_hotmap:
        # plot walk_freq_map
        plt.imshow(walk_freq_map, cmap='hot', interpolation='nearest')
        plt.title("obstacle dense:%.3f, doorway penalty:%d , wall_penalty:%d" % (args.random_obstacle_dense, args.doorway_penalty, args.wall_penalty))
        img_name = "obstacle_dense_" + str(args.random_obstacle_dense) + "doorway_penalty" + str(args.doorway_penalty) + "wall_penalty" + str(args.wall_penalty) +'.png'
        folder_name = args.temp_result_dir+'hotmap/'
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(folder_name + img_name)
        # plt.show()

        with open(args.temp_result_dir+"walk_freq_map.pickle", 'wb') as f:
            pickle.dump(walk_freq_map,f)

    if args.save_temp_result:
        # save simulated paths
        with open(args.temp_result_dir+'path_all.pickle', 'wb') as f:
            pickle.dump(path_all, f)
    
    if args.save_unity3d_result:
        folder_name_unity ='unity3d_result/'
        os.makedirs(folder_name_unity, exist_ok=True)

        # save simulated information for unity 3d simulation
        fp_grid_unity = copy.deepcopy(fp_grid)
        fp_grid_unity[fp_grid_unity==2] = 1
        
        # save obstacles
        obstacle_where = np.where(fp_grid_unity==1)
        obstacle_list = np.zeros((obstacle_where[0].shape[0],2))
        for i in range(obstacle_where[0].shape[0]):
            obstacle_list[i,:] = [obstacle_where[0][i],obstacle_where[1][i]]
            
        obstacle_list = obstacle_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_obstacle.txt", 'w+') as f:
            f.write("numWall,%d" % obstacle_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_obstacle.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, obstacle_list, fmt='%d',delimiter=',')
        
        # save area of interestings
        interest_where = np.where(fp_grid_unity==4)
        
        area_interest_list = np.zeros((interest_where[0].shape[0],2))
        for i in range(interest_where[0].shape[0]):
            area_interest_list[i,:] = [interest_where[0][i],interest_where[1][i]]
        area_interest_list = area_interest_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_interest.txt", 'w+') as f:
            f.write("numInterest,%d" % interest_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_interest.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, area_interest_list, fmt='%d',delimiter=',')
        

    return path_all